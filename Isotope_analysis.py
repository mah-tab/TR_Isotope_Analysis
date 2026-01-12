"""
Python equivalent of the R isotope-analysis block (Site A), reading from XLSX:

Input:
- E:\\FAU master\\Master Thesis\\Data\\d18o_per_sample_sorted_corrected.xlsx

Outputs (all inside OUT_DIR):
- isotope_siteA_outputs.xlsx (all tables as separate sheets)
- correlation_matrix_A_heatmap.png
- mean_d18O_chronology_siteA.png
- d18O_distributions_boxplot_siteA.png
- d18O_trends_siteA_facets.png

Includes:
- Inter-series correlations (pairwise complete)
- Mean δ18O chronology
- dplR-equivalent summary stats per series
- Linear trend analysis (d18O ~ Year) for each series + mean

Author: Mahtab Arjomandi
Date: 2026-01-10
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
OUT_DIR = r"E:\FAU master\Master Thesis\R Python outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# UPDATED: read from Excel instead of CSV
ISO_XLSX_PATH = r"E:\FAU master\Master Thesis\Data\d18o_per_sample_sorted_corrected.xlsx"

SAMPLES = ["HNC_24a", "HNC_25a", "HNC_28a", "HNC_53a", "HNC_58b"]

DPI = 300


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def mean_interseries_correlation(corr: pd.DataFrame) -> float:
    """Mean off-diagonal correlation (upper triangle only), ignoring NaNs."""
    vals = corr.to_numpy()
    upper = vals[np.triu_indices_from(vals, k=1)]
    return float(np.nanmean(upper))


def fit_linear_trend(year: pd.Series, y: pd.Series):
    """
    Fit OLS: y ~ Year (like R lm(y ~ Year)).
    Returns slope, std_error, t_value, p_value for the Year term.
    """
    mask = year.notna() & y.notna()
    x = year.loc[mask].astype(float)
    yy = y.loc[mask].astype(float)

    if len(yy) < 3:
        return np.nan, np.nan, np.nan, np.nan

    X = sm.add_constant(x)
    model = sm.OLS(yy, X).fit()

    slope = model.params.get("Year", np.nan)
    se = model.bse.get("Year", np.nan)
    tval = model.tvalues.get("Year", np.nan)
    pval = model.pvalues.get("Year", np.nan)
    return float(slope), float(se), float(tval), float(pval)


def per_series_dplr_equivalent_stats(df_year_indexed: pd.DataFrame) -> pd.DataFrame:
    """
    dplR-equivalent per-series summary table:
    - coverage (n_obs, missing %, first/last year)
    - descriptive stats (mean/std/min/Q1/median/Q3/max)
    """
    years = df_year_indexed.index.to_numpy()
    n_total = len(years)

    rows = []
    for col in df_year_indexed.columns:
        s = pd.to_numeric(df_year_indexed[col], errors="coerce")
        n_obs = int(s.notna().sum())
        n_missing = int(s.isna().sum())

        if n_obs > 0:
            valid_years = years[s.notna().to_numpy()]
            first_year = int(valid_years.min())
            last_year = int(valid_years.max())

            desc = s.describe(percentiles=[0.25, 0.5, 0.75])
            row = {
                "Series": col,
                "n_total_years": n_total,
                "n_obs": n_obs,
                "n_missing": n_missing,
                "missing_pct": 100.0 * n_missing / n_total,
                "first_year": first_year,
                "last_year": last_year,
                "mean": float(desc["mean"]),
                "std": float(desc["std"]),
                "min": float(desc["min"]),
                "Q1": float(desc["25%"]),
                "median": float(desc["50%"]),
                "Q3": float(desc["75%"]),
                "max": float(desc["max"]),
            }
        else:
            row = {
                "Series": col,
                "n_total_years": n_total,
                "n_obs": 0,
                "n_missing": n_total,
                "missing_pct": 100.0,
                "first_year": np.nan,
                "last_year": np.nan,
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "Q1": np.nan,
                "median": np.nan,
                "Q3": np.nan,
                "max": np.nan,
            }

        rows.append(row)

    return pd.DataFrame(rows)


def save_corr_heatmap(corr: pd.DataFrame, out_png: str, title: str) -> None:
    """Plot and save correlation matrix heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, aspect="equal")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)

    # annotate correlation values
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            val = corr.iloc[i, j]
            txt = "NA" if pd.isna(val) else f"{val:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9)

    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# 0) READ DATA (XLSX)
# ------------------------------------------------------------
df = pd.read_excel(ISO_XLSX_PATH)

# Basic validation
if "Year" not in df.columns:
    raise ValueError(f"'Year' column not found. Columns: {list(df.columns)}")

missing_samples = [c for c in SAMPLES if c not in df.columns]
if missing_samples:
    raise ValueError(f"Missing expected sample columns: {missing_samples}. Found: {list(df.columns)}")

# Ensure numeric dtypes
df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
for c in SAMPLES:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Sort by year
df = df.sort_values("Year").reset_index(drop=True)

print("\n--- Data info ---")
print(df.info())
print("\n--- Data summary ---")
print(df.describe(include="all"))

# ------------------------------------------------------------
# 1) SELECT SERIES
# ------------------------------------------------------------
A_data = df[["Year"] + SAMPLES].copy()

# ------------------------------------------------------------
# 2) INTER-SERIES CORRELATIONS (pairwise complete)
# ------------------------------------------------------------
corr = A_data[SAMPLES].corr(method="pearson", min_periods=2)
mean_corr = mean_interseries_correlation(corr)

print("\nCorrelation matrix:\n", corr)
print(f"\nMean inter-series correlation (upper triangle): {mean_corr:.3f}")

corr_png = os.path.join(OUT_DIR, "correlation_matrix_A_heatmap.png")
save_corr_heatmap(corr, corr_png, "Inter-series correlation matrix – Site A (δ18O)")
print(f"Saved: {corr_png}")

# ------------------------------------------------------------
# 3) MEAN δ18O CHRONOLOGY
# ------------------------------------------------------------
df["mean_d18O"] = df[SAMPLES].mean(axis=1, skipna=True)
A_mean = df[["Year", "mean_d18O"]].copy()

mean_png = os.path.join(OUT_DIR, "mean_d18O_chronology_siteA.png")
plt.figure(figsize=(10, 5))
plt.plot(A_mean["Year"], A_mean["mean_d18O"], marker="o", linestyle="-")
plt.title("Mean δ18O chronology – Site A")
plt.xlabel("Year")
plt.ylabel("δ$^{18}$O (‰)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(mean_png, dpi=DPI, bbox_inches="tight")
plt.close()
print(f"Saved: {mean_png}")

# ------------------------------------------------------------
# 4) dplR-EQUIVALENT STATS (Python)
# ------------------------------------------------------------
rwl_like = df[["Year"] + SAMPLES].dropna(subset=["Year"]).set_index("Year")
series_stats = per_series_dplr_equivalent_stats(rwl_like)

mean_desc = df["mean_d18O"].describe(percentiles=[0.25, 0.5, 0.75])
mean_stats = pd.DataFrame({
    "metric": ["count", "mean", "std", "min", "Q1", "median", "Q3", "max"],
    "value": [
        float(mean_desc["count"]),
        float(mean_desc["mean"]),
        float(mean_desc["std"]),
        float(mean_desc["min"]),
        float(mean_desc["25%"]),
        float(mean_desc["50%"]),
        float(mean_desc["75%"]),
        float(mean_desc["max"]),
    ],
})

# Diagnostic distribution plot
box_png = os.path.join(OUT_DIR, "d18O_distributions_boxplot_siteA.png")
plt.figure(figsize=(10, 5))
plt.boxplot([df[s].dropna().values for s in SAMPLES], labels=SAMPLES, vert=True, showfliers=True)
plt.title("δ18O distributions per series – Site A")
plt.xlabel("Series")
plt.ylabel("δ$^{18}$O (‰)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(box_png, dpi=DPI, bbox_inches="tight")
plt.close()
print(f"Saved: {box_png}")

# ------------------------------------------------------------
# 5) OVERALL MEAN δ18O VALUE
# ------------------------------------------------------------
overall_mean = float(df["mean_d18O"].mean(skipna=True))
print(f"\nOverall mean δ18O (Site A): {overall_mean:.2f} ‰")

# ------------------------------------------------------------
# 6) LINEAR TREND ANALYSIS (d18O ~ Year)
# ------------------------------------------------------------
A_long = df[["Year"] + SAMPLES + ["mean_d18O"]].melt(
    id_vars=["Year"], var_name="Series", value_name="d18O"
)

trend_rows = []
for series_name, sub in A_long.groupby("Series"):
    slope, se, tval, pval = fit_linear_trend(sub["Year"], sub["d18O"])
    trend_rows.append({
        "Series": series_name,
        "slope": slope,
        "std_error": se,
        "t_value": tval,
        "p_value": pval,
    })

trend_df = pd.DataFrame(trend_rows)

trend_png = os.path.join(OUT_DIR, "d18O_trends_siteA_facets.png")

series_order = list(trend_df["Series"])
n = len(series_order)
ncols = 3
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4 * nrows))
axes = np.array(axes).reshape(-1)

for i, sname in enumerate(series_order):
    ax = axes[i]
    sub = A_long[A_long["Series"] == sname].dropna(subset=["Year", "d18O"])
    ax.scatter(sub["Year"], sub["d18O"], alpha=0.4)

    if len(sub) >= 3:
        x = sub["Year"].astype(float).values
        y = sub["d18O"].astype(float).values
        model = sm.OLS(y, sm.add_constant(x)).fit()
        xline = np.linspace(x.min(), x.max(), 100)
        yline = model.predict(sm.add_constant(xline))
        ax.plot(xline, yline)

    ax.set_title(sname)
    ax.set_xlabel("Year")
    ax.set_ylabel("δ$^{18}$O (‰)")
    ax.grid(alpha=0.3)

for j in range(i + 1, len(axes)):
    axes[j].axis("off")

fig.suptitle("Linear trends in δ18O – Site A", y=1.02, fontsize=14)
fig.tight_layout()
fig.savefig(trend_png, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {trend_png}")

# ------------------------------------------------------------
# 7) SAVE MEAN CHRONOLOGY (CSV-like equivalent, but as XLSX too)
# ------------------------------------------------------------
mean_out_csv = os.path.join(OUT_DIR, "iso_mean_chronology_siteA.csv")
A_mean.to_csv(mean_out_csv, index=False)
print(f"Saved: {mean_out_csv}")

# ------------------------------------------------------------
# SAVE EVERYTHING AS XLSX (single workbook)
# ------------------------------------------------------------
xlsx_out = os.path.join(OUT_DIR, "isotope_siteA_outputs.xlsx")
with pd.ExcelWriter(xlsx_out, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="data_with_mean", index=False)
    corr.to_excel(writer, sheet_name="correlation_matrix")
    pd.DataFrame({"mean_interseries_corr": [mean_corr]}).to_excel(
        writer, sheet_name="correlation_summary", index=False
    )
    series_stats.to_excel(writer, sheet_name="dplr_equiv_series_stats", index=False)
    mean_stats.to_excel(writer, sheet_name="mean_series_stats", index=False)
    trend_df.to_excel(writer, sheet_name="trend_results", index=False)

print(f"\nSaved Excel workbook: {xlsx_out}")
