"""
Reads a folder of Excel summary tables (one per climate parameter), created in correlation_summary_analysis.py
with y axis being the climate parameters and x axis the stations
and plots a correlation heatmap across stations. [pearson, spearman, or kendall]
parameters to display:

- maximal_calculated_metric_pearson
- optimal_time_window_pearson
(adjust suffix based on corr type)

Changes in this version:
- Drops/ignores stations: "Kerman - 1969" and "jiroft+Baft" (removed from all files + plot)
- Enforces station order exactly as the keys of STATION_DISPLAY (after dropping the two)
- Makes x/y tick labels bold

Author: Mahtab Arjomandi
Date: 01-03-2026
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# CONFIG
# ----------------------------
INPUT_DIR = r"E:\FAU master\Master Thesis\Correlation\Correlation outputs\Narrow Previous Year taken"   # root only
OUTPUT_DIR = r"E:\FAU master\Master Thesis\Correlation\Correlation outputs\Narrow Previous Year taken"
OUTPUT_NAME = "heatmap_station_correlations_KENDALL.png" # -> change based on correlation type

# Which rows to read (Pearson only) -> change [pearson, spearman, kendall]
ROW_METRIC = "maximal_calculated_metric_kendall"
ROW_WINDOW = "optimal_time_window_kendall"

# Desired Y-axis order from BOTTOM -> TOP
INCLUDE_PARAMS = ["T_Mean", "T_Min", "T_Max", "Precip", "RH", "VPD"]

# Station display-name mapping (x-axis only) + desired order (dict order matters in Python 3.7+)
STATION_DISPLAY = {
    "anar": "Anar(1986-2023)",
    "Baft": "Baft(1989-2023)",
    "Bam_1980": "Bam(1980-2023)",
    "Bam_clim": "Bam(1974-2023)",
    "Kerman - 1969": "Kerman(wo ea es 1974-2023)",  # DROP
    "kerman": "Kerman(1974-2023)",
    "kerman_1980": "Kerman(1980-2023)",
    "jiroft_clim": "Jiroft(1990-2023)",
    "rafsanjan_clim": "Rafsanjan(1993-2023)",
    "shahrebabak_clim": "Shahrebabak(1987-2023)",
    "sirjan_clim": "Sirjan(1985-2023)",
    "jiroft+Baft_clim": "Jiroft+Baft(1989-2023)",
    "jiroft+Baft": "Jiroft+Baft(wo some rows1989-2023)",  # DROP
}

# Stations to completely ignore/remove (from all excel files + plot)
DROP_STATIONS = {"Kerman - 1969", "jiroft+Baft"}

# Plot settings (bigger fonts)
FIG_W = 20
FIG_H = 9

ANNOT_FONTSIZE = 11       # text inside squares
XTICK_FONTSIZE = 12       # x-axis station labels
YTICK_FONTSIZE = 14       # y-axis parameter labels
TITLE_FONTSIZE = 20       # title
CBAR_FONTSIZE = 14        # colorbar label/ticks

# Colormap: blue-white-red (negative to positive)
CMAP = "RdBu_r"


# ----------------------------
# HELPERS
# ----------------------------
def safe_param_name_from_filename(path: str) -> str:
    """
    Extract parameter name from:
      summary_correlations_T_mean.xlsx -> T_mean
      summary_correlations_T_Min.xlsx  -> T_Min
    """
    base = os.path.basename(path)
    name = os.path.splitext(base)[0]
    m = re.match(r"summary_correlations_(.+)$", name, flags=re.IGNORECASE)
    return m.group(1) if m else name


def to_float_or_nan(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


# ----------------------------
# MAIN
# ----------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ROOT ONLY: *.xlsx in INPUT_DIR, not recursive
    xlsx_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.xlsx")))
    if not xlsx_files:
        raise FileNotFoundError(f"No .xlsx files found in: {INPUT_DIR}")

    # Enforced station order from STATION_DISPLAY (minus dropped stations)
    desired_station_order = [k for k in STATION_DISPLAY.keys() if k not in DROP_STATIONS]
    desired_station_display = [STATION_DISPLAY[k] for k in desired_station_order]

    all_params = []
    corr_rows = []
    annot_rows = []

    for fp in xlsx_files:
        param = safe_param_name_from_filename(fp)

        if INCLUDE_PARAMS is not None and param not in INCLUDE_PARAMS:
            continue

        df = pd.read_excel(fp)
        if df.shape[1] < 2:
            print(f"[SKIP] {fp} has too few columns.")
            continue

        row_label_col = df.columns[0]
        df = df.copy()
        df[row_label_col] = df[row_label_col].astype(str)

        # station columns from file (drop ignored ones)
        file_stations = [c for c in df.columns[1:] if c not in DROP_STATIONS]

        # Keep only stations we care about and enforce order
        stations = [s for s in desired_station_order if s in file_stations]

        if len(stations) == 0:
            print(f"[SKIP] {fp} has none of the desired stations after dropping columns.")
            continue

        metric_row = df.loc[df[row_label_col].str.strip().str.lower() == ROW_METRIC.lower()]
        window_row = df.loc[df[row_label_col].str.strip().str.lower() == ROW_WINDOW.lower()]

        # If missing rows, fill with NA
        if metric_row.empty:
            metric_vals = {s: np.nan for s in stations}
        else:
            metric_vals = {s: to_float_or_nan(metric_row.iloc[0][s]) for s in stations}

        if window_row.empty:
            window_vals = {s: None for s in stations}
        else:
            window_vals = {s: window_row.iloc[0][s] for s in stations}

        # Build row vectors in desired_station_order length
        corr_vec = []
        annot_vec = []

        for s in desired_station_order:
            if s not in stations:
                corr_vec.append(np.nan)
                annot_vec.append("NA")
                continue

            r = metric_vals.get(s, np.nan)
            w = window_vals.get(s, None)

            if np.isnan(r):
                annot = "NA"
            else:
                w_txt = "NA" if (
                    w is None
                    or (isinstance(w, float) and np.isnan(w))
                    or str(w).strip() == ""
                ) else str(w).strip()
                annot = f"{r:.2f}\n{w_txt}"

            corr_vec.append(r)
            annot_vec.append(annot)

        all_params.append(param)
        corr_rows.append(corr_vec)
        annot_rows.append(annot_vec)

    if not all_params:
        raise RuntimeError("No parameters were loaded. Check INCLUDE_PARAMS or file naming.")

    corr_mat = np.array(corr_rows, dtype=float)
    annot_mat = np.array(annot_rows, dtype=object)

    # ----------------------------
    # FORCE y-axis order (bottom -> top) exactly as INCLUDE_PARAMS
    # ----------------------------
    idx_map = {p: i for i, p in enumerate(all_params)}
    ordered_params = [p for p in INCLUDE_PARAMS if p in idx_map]

    if not ordered_params:
        raise RuntimeError(
            "None of INCLUDE_PARAMS were found in loaded parameters.\n"
            f"Loaded: {all_params}\nWanted: {INCLUDE_PARAMS}"
        )

    order_idx = [idx_map[p] for p in ordered_params]
    corr_mat = corr_mat[order_idx, :]
    annot_mat = annot_mat[order_idx, :]
    all_params = ordered_params

    # Matplotlib y=0 is TOP; to display bottom->top order, flip rows
    corr_mat = np.flipud(corr_mat)
    annot_mat = np.flipud(annot_mat)
    all_params = list(reversed(all_params))

    # ----------------------------
    # PLOT
    # ----------------------------
    plt.figure(figsize=(FIG_W, FIG_H))

    finite_vals = corr_mat[np.isfinite(corr_mat)]
    if finite_vals.size == 0:
        vmin, vmax = -1, 1
    else:
        vmax = float(np.nanmax(np.abs(finite_vals)))
        vmin = -vmax

    im = plt.imshow(corr_mat, aspect="auto", cmap=CMAP, vmin=vmin, vmax=vmax)

    # x ticks: enforced order + bold
    plt.xticks(
        range(len(desired_station_order)),
        desired_station_display,
        rotation=45,
        ha="right",
        fontsize=XTICK_FONTSIZE,
        fontweight="bold"
    )

    # y ticks: bold
    plt.yticks(
        range(len(all_params)),
        all_params,
        fontsize=YTICK_FONTSIZE,
        fontweight="bold"
    )

    plt.title(
        "Kendall: maximal correlation (r) + optimal time window",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold"
    )

    cbar = plt.colorbar(im, fraction=0.03, pad=0.02)
    cbar.set_label("Kendall r (maximal_calculated_metric)", fontsize=CBAR_FONTSIZE, fontweight="bold")
    cbar.ax.tick_params(labelsize=CBAR_FONTSIZE)

    # Cell gridlines
    ax = plt.gca()
    ax.set_xticks(np.arange(-.5, len(desired_station_order), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(all_params), 1), minor=True)
    plt.grid(which="minor", color="white", linestyle="-", linewidth=1)
    plt.tick_params(which="minor", bottom=False, left=False)

    # Annotations
    for i in range(len(all_params)):
        for j in range(len(desired_station_order)):
            txt = annot_mat[i, j]
            if txt == "NA":
                continue

            val = corr_mat[i, j]
            text_color = "white" if (np.isfinite(val) and abs(val) > 0.45 * vmax) else "black"

            plt.text(
                j, i, txt,
                ha="center", va="center",
                fontsize=ANNOT_FONTSIZE,
                color=text_color,
                fontweight="bold"
            )

    plt.tight_layout()

    out_png = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    plt.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close()

    print(f"Saved heatmap to:\n{out_png}")


if __name__ == "__main__":
    main()