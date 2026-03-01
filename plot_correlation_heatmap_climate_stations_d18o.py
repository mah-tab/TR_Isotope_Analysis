"""
plot_station_heatmap_pearson.py

Reads a folder of Excel summary tables (one per climate parameter),
and plots a Pearson-only heatmap across stations.

Each Excel file is assumed to have:
- First row: header with station names in columns (col0 is parameter name)
- First column: row labels such as:
    maximal_calculated_metric_pearson
    optimal_time_window_pearson
    ...
- Cells: values for each station.

Heatmap:
- X = stations
- Y = climate parameters (from filename)
- Cell text = "<corr_value>\n<optimal_time_window>"
- Color = corr_value (Pearson maximal_calculated_metric)

Notes:
- Only reads .xlsx files in INPUT_DIR ROOT (ignores subfolders).
- Forces Y-axis order (bottom -> top) exactly as INCLUDE_PARAMS.

Author: (adapted for Mahtab Arjomandi)
Date: 2026-xx-xx
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
INPUT_DIR = r"E:\FAU master\Master Thesis\Correlation\Correlation outputs\Narrow Next Year taken"   # <-- folder with the Excel tables (root only)
OUTPUT_DIR = r"E:\FAU master\Master Thesis\Correlation\Correlation outputs\Narrow Next Year taken"  # <-- where to save the heatmap
OUTPUT_NAME = "heatmap_station_correlations_PEARSON.png"

# Which rows to read (Pearson only)
ROW_METRIC = "maximal_calculated_metric_pearson"
ROW_WINDOW = "optimal_time_window_pearson"

# Desired Y-axis order from BOTTOM -> TOP
INCLUDE_PARAMS = ["T_mean", "T_Min", "T_Max", "Precip", "RH", "VPD"]

# Plot settings
FIG_W = 18
FIG_H = 8
ANNOT_FONTSIZE = 9
XTICK_FONTSIZE = 10
YTICK_FONTSIZE = 11
TITLE_FONTSIZE = 16

# Colormap: blue-white-red (negative to positive)
CMAP = "RdBu_r"  # diverging


# ----------------------------
# HELPERS
# ----------------------------
def safe_param_name_from_filename(path: str) -> str:
    """
    Extract a parameter name from filenames like:
      summary_correlations_T_mean.xlsx -> T_mean
      summary_correlations_T_Min.xlsx  -> T_Min
    Fallback: filename without extension.
    """
    base = os.path.basename(path)
    name = os.path.splitext(base)[0]

    m = re.match(r"summary_correlations_(.+)$", name, flags=re.IGNORECASE)
    if m:
        return m.group(1)

    return name


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

    all_params = []
    stations_master = None
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

        # First column contains row labels
        row_label_col = df.columns[0]
        df = df.copy()
        df[row_label_col] = df[row_label_col].astype(str)

        # Station columns are everything except first column
        stations = list(df.columns[1:])

        # Ensure consistent station order across all files
        if stations_master is None:
            stations_master = stations
        else:
            common = [s for s in stations_master if s in stations]
            if len(common) == 0:
                print(f"[SKIP] {fp} has no common station columns with previous files.")
                continue
            stations = common

        # Find Pearson rows
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

        # Build row vectors in stations_master order
        corr_vec = []
        annot_vec = []

        for s in stations_master:
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
    stations_master = list(stations_master)

    # ----------------------------
    # FORCE y-axis order (bottom -> top) exactly as INCLUDE_PARAMS
    # ----------------------------
    if INCLUDE_PARAMS is not None:
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

    # symmetric limits around 0
    finite_vals = corr_mat[np.isfinite(corr_mat)]
    if finite_vals.size == 0:
        vmin, vmax = -1, 1
    else:
        vmax = float(np.nanmax(np.abs(finite_vals)))
        vmin = -vmax

    im = plt.imshow(corr_mat, aspect="auto", cmap=CMAP, vmin=vmin, vmax=vmax)

    # Axis ticks/labels
    plt.xticks(
        range(len(stations_master)),
        stations_master,
        rotation=45,
        ha="right",
        fontsize=XTICK_FONTSIZE
    )
    plt.yticks(range(len(all_params)), all_params, fontsize=YTICK_FONTSIZE)

    plt.title(
        "Pearson: maximal correlation (r) + optimal time window",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold"
    )

    # Colorbar
    cbar = plt.colorbar(im, fraction=0.03, pad=0.02)
    cbar.set_label("Pearson r (maximal_calculated_metric)", fontsize=12)

    # Cell gridlines
    ax = plt.gca()
    ax.set_xticks(np.arange(-.5, len(stations_master), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(all_params), 1), minor=True)
    plt.grid(which="minor", color="white", linestyle="-", linewidth=1)
    plt.tick_params(which="minor", bottom=False, left=False)

    # Annotations
    for i in range(len(all_params)):
        for j in range(len(stations_master)):
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