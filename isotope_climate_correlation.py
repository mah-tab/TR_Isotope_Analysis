"""
isotope_climate_correlation.py

- Reads isotope data from XLSX (Year + tree series)
- Reads monthly climate data from XLSX (Year, Month + climate vars)
- Computes:
  * mean_d18O per year (mean across all trees)
  * yearly mean climate variables (mean across months)
  * yearly mean T_Delta = T_Max - T_Min
- Correlates (Pearson r + p):
  * mean_d18O vs each climate variable
  * each tree series vs each climate variable
- Also runs the full "reconstruction-style" workflow per climate variable
  (linear transfer function + validation + DW + CI), using mean_d18O as predictor.
- Saves results as XLSX + plots as PNG into OUT_DIR.

Author: Mahtab Arjomandi (adapt)
Date: 2026-01-xx
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from sklearn.model_selection import KFold


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
ISO_XLSX = r"E:\FAU master\Master Thesis\Data\d18o_per_sample_sorted_corrected.xlsx"
CLIM_XLSX = r"E:\FAU master\Master Thesis\Data\climate data\kerman.xlsx"

OUT_DIR = r"E:\FAU master\Master Thesis\R Python outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Tree series columns in isotope file
SAMPLES = ["HNC_24a", "HNC_25a", "HNC_28a", "HNC_53a", "HNC_58b"]

# Climate variables to aggregate as yearly mean over months
CLIMATE_VARS_MEAN = ["T_Mean", "Precip", "RH", "VPD", "es", "ea"]

# Extra computed climate variable
DELTA_NAME = "T_Delta"  # computed as (T_Max - T_Min) monthly then yearly mean

# Calibration period
CAL_START = 1974
CAL_END = 2023

# Validation configs
SPLIT_RATIO = 0.7
K_FOLDS = 5
RANDOM_SEED = 123


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def to_numeric_safe(s: pd.Series) -> pd.Series:
    """Handles comma decimals and coerces non-numeric to NaN."""
    if s.dtype == "object":
        s = s.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def ensure_sorted_by_year(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("Year").reset_index(drop=True)


def calc_re_ce(obs: np.ndarray, pred: np.ndarray, ref_mean: float) -> float:
    """RE/CE as in your R code."""
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denom = np.sum((obs - ref_mean) ** 2)
    if denom == 0 or np.isnan(denom):
        return np.nan
    return 1.0 - (np.sum((obs - pred) ** 2) / denom)


def make_exog(df: pd.DataFrame, x_col: str) -> pd.DataFrame:
    """
    Always returns a 2-column exog matrix: const + x_col
    and guarantees the constant is present (fixes your LOOCV error).
    """
    X = df[[x_col]].astype(float).copy()
    X = sm.add_constant(X, has_constant="add")
    return X


def fit_ols(df_cal: pd.DataFrame, x_col: str, y_col: str):
    """Fit OLS y ~ x with intercept."""
    X = make_exog(df_cal, x_col)
    y = df_cal[y_col].astype(float)
    model = sm.OLS(y, X).fit()
    return model


def observed_vs_fitted_plot(df_cal: pd.DataFrame, out_png: str, title: str) -> None:
    """Observed vs fitted with 1:1 line."""
    plt.figure(figsize=(6, 6))
    plt.scatter(df_cal["Target"], df_cal["Fitted"], alpha=0.8)

    mn = np.nanmin([df_cal["Target"].min(), df_cal["Fitted"].min()])
    mx = np.nanmax([df_cal["Target"].max(), df_cal["Fitted"].max()])
    plt.plot([mn, mx], [mn, mx], linestyle="--")

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Observed Target")
    plt.ylabel("Fitted Target")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def reconstruction_plot(recon: pd.DataFrame, out_png: str, title: str, ylabel: str) -> None:
    """Reconstruction with confidence interval ribbon."""
    plt.figure(figsize=(10, 5))
    plt.plot(recon["Year"], recon["Reconstructed"], linewidth=2)
    plt.fill_between(recon["Year"], recon["CI_lower"], recon["CI_upper"], alpha=0.3)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Year")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------
# 1) Read isotope XLSX and compute yearly mean_d18O
# ------------------------------------------------------------
iso = pd.read_excel(ISO_XLSX)

if "Year" not in iso.columns:
    raise ValueError(f"Isotope file must contain 'Year'. Found: {list(iso.columns)}")

missing_samples = [c for c in SAMPLES if c not in iso.columns]
if missing_samples:
    raise ValueError(f"Missing isotope sample columns: {missing_samples}\nFound: {list(iso.columns)}")

iso["Year"] = to_numeric_safe(iso["Year"]).astype("Int64")
for c in SAMPLES:
    iso[c] = to_numeric_safe(iso[c])

iso = iso.dropna(subset=["Year"]).copy()
iso = ensure_sorted_by_year(iso)

# mean d18O per year across all trees (your requirement)
iso["mean_d18O"] = iso[SAMPLES].mean(axis=1, skipna=True)

iso_chron = iso[["Year", "mean_d18O"]].copy()


# ------------------------------------------------------------
# 2) Read climate XLSX and compute yearly means
# ------------------------------------------------------------
clim = pd.read_excel(CLIM_XLSX)

required = ["Year", "Month", "T_Max", "T_Min"]
missing = [c for c in required if c not in clim.columns]
if missing:
    raise ValueError(f"Climate file missing required columns: {missing}\nFound: {list(clim.columns)}")

clim["Year"] = to_numeric_safe(clim["Year"]).astype("Int64")
clim["Month"] = to_numeric_safe(clim["Month"]).astype("Int64")

# Convert needed columns
for c in set(CLIMATE_VARS_MEAN + ["T_Max", "T_Min"]):
    if c in clim.columns:
        clim[c] = to_numeric_safe(clim[c])

clim = clim.dropna(subset=["Year", "Month"]).copy()
clim = clim.sort_values(["Year", "Month"])

# Monthly delta T
clim[DELTA_NAME] = clim["T_Max"] - clim["T_Min"]

# Yearly mean over months (as requested)
agg = {v: "mean" for v in CLIMATE_VARS_MEAN}
agg[DELTA_NAME] = "mean"

yearly_clim = clim.groupby("Year", as_index=False).agg(agg)
yearly_clim = ensure_sorted_by_year(yearly_clim)


# ------------------------------------------------------------
# 3) Build correlation tables:
#    A) mean_d18O vs climate vars
#    B) each tree series vs climate vars
# ------------------------------------------------------------
targets = CLIMATE_VARS_MEAN + [DELTA_NAME]

# Merge once for correlations (use calibration years and drop NA later per target)
corr_base = pd.merge(
    iso[["Year"] + SAMPLES + ["mean_d18O"]],
    yearly_clim,
    on="Year",
    how="inner"
)

# Restrict to calibration years
corr_base = corr_base[(corr_base["Year"] >= CAL_START) & (corr_base["Year"] <= CAL_END)].copy()

# A) mean_d18O vs each climate var
mean_corr_rows = []
for t in targets:
    sub = corr_base[["mean_d18O", t]].dropna()
    if len(sub) < 3:
        mean_corr_rows.append({"Target": t, "n": len(sub), "r": np.nan, "p": np.nan})
        continue
    r, p = pearsonr(sub["mean_d18O"].astype(float), sub[t].astype(float))
    mean_corr_rows.append({"Target": t, "n": len(sub), "r": r, "p": p})

mean_corr_df = pd.DataFrame(mean_corr_rows).sort_values("Target")

# B) each tree series vs each climate var
tree_corr_rows = []
for tree in SAMPLES:
    for t in targets:
        sub = corr_base[[tree, t]].dropna()
        if len(sub) < 3:
            tree_corr_rows.append({"Tree": tree, "Target": t, "n": len(sub), "r": np.nan, "p": np.nan})
            continue
        r, p = pearsonr(sub[tree].astype(float), sub[t].astype(float))
        tree_corr_rows.append({"Tree": tree, "Target": t, "n": len(sub), "r": r, "p": p})

tree_corr_df = pd.DataFrame(tree_corr_rows).sort_values(["Tree", "Target"])


# ------------------------------------------------------------
# 4) Full R-equivalent reconstruction workflow per climate var
#    using mean_d18O as predictor (Target ~ mean_d18O)
# ------------------------------------------------------------
recon_summary_rows = []

for target_var in targets:
    if target_var not in corr_base.columns:
        continue

    # Build model dataframe
    dat = corr_base[["Year", "mean_d18O", target_var]].rename(columns={target_var: "Target"}).copy()
    dat = dat.dropna(subset=["mean_d18O", "Target"]).copy()
    dat = ensure_sorted_by_year(dat)

    if len(dat) < 10:
        print(f"[SKIP] Not enough data for reconstruction: {target_var} (n={len(dat)})")
        continue

    # Fit transfer function
    mod = fit_ols(dat, x_col="mean_d18O", y_col="Target")
    dat["Fitted"] = mod.predict(make_exog(dat, "mean_d18O"))

    # DW
    dw = float(durbin_watson(mod.resid))

    # 50/50 split
    n = len(dat)
    split_50 = n // 2
    early = dat.iloc[:split_50].copy()
    late = dat.iloc[split_50:].copy()

    m_early = fit_ols(early, "mean_d18O", "Target")
    pred_late = m_early.predict(make_exog(late, "mean_d18O"))

    RE_late = calc_re_ce(late["Target"], pred_late, ref_mean=float(early["Target"].mean()))
    CE_late = calc_re_ce(late["Target"], pred_late, ref_mean=float(late["Target"].mean()))

    m_late = fit_ols(late, "mean_d18O", "Target")
    pred_early = m_late.predict(make_exog(early, "mean_d18O"))

    RE_early = calc_re_ce(early["Target"], pred_early, ref_mean=float(late["Target"].mean()))
    CE_early = calc_re_ce(early["Target"], pred_early, ref_mean=float(early["Target"].mean()))

    # Flexible split 70/30
    split_n = int(np.floor(n * SPLIT_RATIO))
    cal_part = dat.iloc[:split_n].copy()
    val_part = dat.iloc[split_n:].copy()

    m_split = fit_ols(cal_part, "mean_d18O", "Target")
    pred_val = m_split.predict(make_exog(val_part, "mean_d18O"))

    RE_split = calc_re_ce(val_part["Target"], pred_val, ref_mean=float(cal_part["Target"].mean()))
    CE_split = calc_re_ce(val_part["Target"], pred_val, ref_mean=float(val_part["Target"].mean()))

    # LOOCV (fixed! always uses const+predictor)
    loocv_pred = np.full(n, np.nan, dtype=float)
    for i in range(n):
        train = dat.drop(dat.index[i])
        test = dat.iloc[[i]]
        m_i = fit_ols(train, "mean_d18O", "Target")
        loocv_pred[i] = float(m_i.predict(make_exog(test, "mean_d18O")).iloc[0])

    RE_loocv = calc_re_ce(dat["Target"], loocv_pred, ref_mean=float(dat["Target"].mean()))

    # k-fold CV
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    pred_kfold = np.full(n, np.nan, dtype=float)

    for train_idx, test_idx in kf.split(dat):
        train = dat.iloc[train_idx]
        test = dat.iloc[test_idx]
        m_k = fit_ols(train, "mean_d18O", "Target")
        pred_kfold[test_idx] = m_k.predict(make_exog(test, "mean_d18O")).to_numpy()

    k_cor = float(np.corrcoef(dat["Target"].astype(float), pred_kfold.astype(float))[0, 1])
    k_rmse = float(np.sqrt(np.nanmean((dat["Target"].astype(float) - pred_kfold.astype(float)) ** 2)))

    # Reconstruction with confidence intervals (mean CI)
    X_all = make_exog(iso_chron.dropna(subset=["mean_d18O"]).copy(), "mean_d18O")
    pred_frame = mod.get_prediction(X_all).summary_frame(alpha=0.05)

    recon = pd.DataFrame({
        "Year": iso_chron.dropna(subset=["mean_d18O"])["Year"].astype(int).to_numpy(),
        "Reconstructed": pred_frame["mean"].to_numpy(),
        "CI_lower": pred_frame["mean_ci_lower"].to_numpy(),
        "CI_upper": pred_frame["mean_ci_upper"].to_numpy(),
    }).sort_values("Year")

    # Save plots
    tag = target_var.replace(" ", "_")
    obsfit_png = os.path.join(OUT_DIR, f"Calibration_observed_vs_fitted_{tag}.png")
    recon_png = os.path.join(OUT_DIR, f"Reconstruction_{tag}.png")

    observed_vs_fitted_plot(dat, obsfit_png, f"Calibration: observed vs fitted ({tag})")
    reconstruction_plot(recon, recon_png, f"δ18O-based reconstruction – {tag}", ylabel=tag)

    # Save per-target Excel
    xlsx_out = os.path.join(OUT_DIR, f"climate_reconstruction_{tag}.xlsx")
    validation_summary = pd.DataFrame({
        "Method": [
            "50/50 Early→Late",
            "50/50 Late→Early",
            f"{int(SPLIT_RATIO*100)}/{int((1-SPLIT_RATIO)*100)}",
            "LOOCV",
            f"k-fold (k={K_FOLDS})",
        ],
        "RE": [RE_late, RE_early, RE_split, RE_loocv, np.nan],
        "CE": [CE_late, CE_early, CE_split, np.nan, np.nan],
        "Correlation": [np.nan, np.nan, np.nan, np.nan, k_cor],
        "RMSE": [np.nan, np.nan, np.nan, np.nan, k_rmse],
    })

    model_summary = pd.DataFrame({
        "metric": ["n", "intercept", "slope(mean_d18O)", "R2", "adj_R2", "p_slope", "DW"],
        "value": [
            len(dat),
            float(mod.params["const"]),
            float(mod.params["mean_d18O"]),
            float(mod.rsquared),
            float(mod.rsquared_adj),
            float(mod.pvalues["mean_d18O"]),
            dw,
        ]
    })

    with pd.ExcelWriter(xlsx_out, engine="openpyxl") as writer:
        dat.to_excel(writer, sheet_name="calibration_data", index=False)
        model_summary.to_excel(writer, sheet_name="model_summary", index=False)
        validation_summary.to_excel(writer, sheet_name="validation_summary", index=False)
        recon.to_excel(writer, sheet_name="reconstruction", index=False)

    recon_summary_rows.append({
        "Target": tag,
        "n": len(dat),
        "slope": float(mod.params["mean_d18O"]),
        "p_slope": float(mod.pvalues["mean_d18O"]),
        "R2": float(mod.rsquared),
        "DW": dw,
        "RE_50_EarlyLate": RE_late,
        "CE_50_EarlyLate": CE_late,
        "RE_50_LateEarly": RE_early,
        "CE_50_LateEarly": CE_early,
        "RE_split": RE_split,
        "CE_split": CE_split,
        "RE_LOOCV": RE_loocv,
        "kfold_cor": k_cor,
        "kfold_rmse": k_rmse,
        "xlsx_out": xlsx_out,
        "obsfit_png": obsfit_png,
        "recon_png": recon_png,
    })

    print(f"[DONE] Reconstruction for {tag}: {xlsx_out}")

# ------------------------------------------------------------
# 5) Save the correlation tables + global summary (XLSX)
# ------------------------------------------------------------
out_xlsx = os.path.join(OUT_DIR, "climate_isotope_correlation_outputs.xlsx")
with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
    mean_corr_df.to_excel(writer, sheet_name="mean_d18O_vs_climate", index=False)
    tree_corr_df.to_excel(writer, sheet_name="trees_vs_climate", index=False)

    if recon_summary_rows:
        pd.DataFrame(recon_summary_rows).to_excel(writer, sheet_name="reconstruction_summary", index=False)

print(f"\nSaved: {out_xlsx}")
