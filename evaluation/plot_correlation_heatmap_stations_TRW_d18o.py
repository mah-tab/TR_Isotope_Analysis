"""
Creates station-comparison heatmaps for climate correlations.

Input:
  1) d18O extracted master Excel:
     summary_correlations_all_climate_params.xlsx

  2) TRW extracted master Excel:
     TRW_summary_correlations_all_climate_params.xlsx

For each proxy:
  - d18O
  - TRW

For each correlation method:
  - pearson
  - spearman
  - kendall

Plots:
  - y-axis: climate variables
  - x-axis: stations
  - cell color: maximal_calculated_metric
  - cell annotation: r value + optimal time window

Drops/ignores stations:
  - Bam_1980
  - Kerman - 1969
  - kerman_1980
  - jiroft+Baft_clim
  - jiroft+Baft

Author: Mahtab Arjomandi
Updated: station-comparison heatmaps for d18O and TRW
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

D18_OUTPUT_DIR = (
    r"E:\FAU master\Master Thesis\Correlation\Correlation outputs"
    r"\Narrow missing removed\d18 climate correlation\python outputs\new raw final"
)

TRW_OUTPUT_DIR = (
    r"E:\FAU master\Master Thesis\Correlation\Correlation outputs"
    r"\Narrow missing removed\TRW climate correlation\python outputs\new raw final"
)

D18_MASTER_XLSX = os.path.join(
    D18_OUTPUT_DIR,
    "summary_correlations_all_climate_params.xlsx"
)

TRW_MASTER_XLSX = os.path.join(
    TRW_OUTPUT_DIR,
    "TRW_summary_correlations_all_climate_params.xlsx"
)

# We use SEASONAL because the attached example shows multi-month optimal windows.
APPROACH_TO_PLOT = "SEASONAL"

CORR_METHODS = ["pearson", "spearman", "kendall"]

# Desired Y-axis order from BOTTOM -> TOP
INCLUDE_PARAMS = ["T_Mean", "T_Min", "T_Max", "Precip", "RH", "VPD"]

# Station order exactly like your old plot, after removing dropped stations.
# Keys are possible internal station names from the Excel files.
# Values are labels shown on x-axis.
STATION_DISPLAY_BASE = {
    "anar": "Anar(1986-2023)",
    "Baft": "Baft(1989-2023)",
    "Bam_1980": "Bam(1980-2023)",  # DROP
    "Bam_clim": "Bam(1974-2023)",
    "Kerman - 1969": "Kerman(wo ea es 1974-2023)",  # DROP
    "kerman": "Kerman(1974-2023)",
    "kerman_1980": "Kerman(1980-2023)",  # DROP
    "jiroft_clim": "Jiroft(1990-2023)",
    "rafsanjan_clim": "Rafsanjan(1993-2023)",
    "shahrebabak_clim": "Shahrebabak(1987-2023)",
    "sirjan_clim": "Sirjan(1985-2023)",
    "jiroft+Baft_clim": "Jiroft+Baft(1989-2023)",  # DROP
    "jiroft+Baft": "Jiroft+Baft(wo some rows1989-2023)",  # DROP
}

# Same order, but also supports TRW filenames with _TRW suffix.
STATION_DISPLAY_TRW = {
    "anar_TRW": "Anar(1986-2023)",
    "Baft_TRW": "Baft(1989-2023)",
    "Bam_1980_TRW": "Bam(1980-2023)",  # DROP
    "Bam_clim_TRW": "Bam(1974-2023)",
    "Kerman - 1969_TRW": "Kerman(wo ea es 1974-2023)",  # DROP
    "kerman_TRW": "Kerman(1974-2023)",
    "kerman_1980_TRW": "Kerman(1980-2023)",  # DROP
    "jiroft_clim_TRW": "Jiroft(1990-2023)",
    "rafsanjan_clim_TRW": "Rafsanjan(1993-2023)",
    "shahrebabak_clim_TRW": "Shahrebabak(1987-2023)",
    "sirjan_clim_TRW": "Sirjan(1985-2023)",
    "jiroft+Baft_clim_TRW": "Jiroft+Baft(1989-2023)",  # DROP
    "jiroft+Baft_TRW": "Jiroft+Baft(wo some rows1989-2023)",  # DROP
}

DROP_STATIONS_BASE = {
    "Bam_1980",
    "Kerman - 1969",
    "kerman_1980",
    "jiroft+Baft_clim",
    "jiroft+Baft",
}

DROP_STATIONS_TRW = {
    "Baft",  # DROP wrong non-TRW Baft column
    "Bam_1980_TRW",
    "Kerman - 1969_TRW",
    "kerman_1980_TRW",
    "jiroft+Baft_clim_TRW",
    "jiroft+Baft_TRW",
    "Bam_1980",
    "Kerman - 1969",
    "kerman_1980",
    "jiroft+Baft_clim",
    "jiroft+Baft",
}

# Plot settings
FIG_W = 18
FIG_H = 8

ANNOT_FONTSIZE = 11
XTICK_FONTSIZE = 12
YTICK_FONTSIZE = 14
TITLE_FONTSIZE = 20
CBAR_FONTSIZE = 14

CMAP = "RdBu_r"

# Fixed color scale.
# Set to None if you want automatic symmetric scaling.
FIXED_VMIN = -0.75
FIXED_VMAX =  0.75


# ============================================================
# HELPERS
# ============================================================

def method_title(method):
    if method.lower() == "pearson":
        return "Pearson"
    if method.lower() == "spearman":
        return "Spearman"
    if method.lower() == "kendall":
        return "Kendall"
    return method


def safe_file_text(x):
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(x))


def normalize_city_for_matching(city):
    """
    Keeps city names as close as possible, only stripping spaces.
    """
    return str(city).strip()


def read_long_all(master_xlsx):
    """
    Reads the long_all sheet from the master Excel created earlier.
    """
    if not os.path.exists(master_xlsx):
        raise FileNotFoundError(f"Master Excel not found:\n{master_xlsx}")

    df = pd.read_excel(master_xlsx, sheet_name="long_all")
    df = df.copy()

    df["city"] = df["city"].astype(str).map(normalize_city_for_matching)
    df["approach"] = df["approach"].astype(str).str.upper().str.strip()
    df["variable"] = df["variable"].astype(str).str.strip()
    df["method"] = df["method"].astype(str).str.lower().str.strip()

    df["maximal_calculated_metric"] = pd.to_numeric(
        df["maximal_calculated_metric"],
        errors="coerce"
    )

    return df


def get_station_order_and_labels(df, station_display, drop_stations):
    """
    Returns station keys and display labels in desired order.
    Only keeps stations present in the dataframe.
    """

    available = set(df["city"].unique())

    station_order = []
    station_labels = []

    for station_key, station_label in station_display.items():
        if station_key in drop_stations:
            continue

        if station_key in available:
            station_order.append(station_key)
            station_labels.append(station_label)

    return station_order, station_labels


def value_to_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def window_to_text(x):
    if x is None:
        return "NA"
    if isinstance(x, float) and np.isnan(x):
        return "NA"
    x = str(x).strip()
    if x == "" or x.lower() == "nan":
        return "NA"
    return x


def choose_text_color(val, vmax_abs):
    if not np.isfinite(val):
        return "black"
    if vmax_abs <= 0:
        return "black"
    return "white" if abs(val) > 0.45 * vmax_abs else "black"


# ============================================================
# MAIN PLOTTING FUNCTION
# ============================================================

def create_station_heatmap(
    master_xlsx,
    output_dir,
    proxy_label,
    output_prefix,
    station_display,
    drop_stations
):
    """
    Creates one heatmap per correlation method for one proxy.
    """

    os.makedirs(output_dir, exist_ok=True)

    df = read_long_all(master_xlsx)

    df = df[
        (df["approach"] == APPROACH_TO_PLOT.upper()) &
        (df["variable"].isin(INCLUDE_PARAMS)) &
        (~df["city"].isin(drop_stations))
    ].copy()

    station_order, station_labels = get_station_order_and_labels(
        df=df,
        station_display=station_display,
        drop_stations=drop_stations
    )

    if len(station_order) == 0:
        raise RuntimeError(
            f"No desired stations were found for {proxy_label}.\n"
            f"Available stations are:\n{sorted(df['city'].unique())}"
        )

    for method in CORR_METHODS:

        sub = df[df["method"] == method].copy()

        if sub.empty:
            print(f"[SKIP] No data for {proxy_label} | {method}")
            continue

        # Matrices in requested y-axis order bottom -> top first
        corr_rows = []
        annot_rows = []
        params_loaded = []

        for param in INCLUDE_PARAMS:

            param_rows = sub[sub["variable"] == param].copy()

            corr_vec = []
            annot_vec = []

            for station in station_order:

                one = param_rows[param_rows["city"] == station]

                if one.empty:
                    r = np.nan
                    w = "NA"
                else:
                    row = one.iloc[0]
                    r = value_to_float(row.get("maximal_calculated_metric", np.nan))
                    w = window_to_text(row.get("optimal_time_window", "NA"))

                corr_vec.append(r)

                if np.isnan(r):
                    annot_vec.append("NA")
                else:
                    annot_vec.append(f"{r:.2f}\n{w}")

            corr_rows.append(corr_vec)
            annot_rows.append(annot_vec)
            params_loaded.append(param)

        corr_mat = np.array(corr_rows, dtype=float)
        annot_mat = np.array(annot_rows, dtype=object)

        # Matplotlib y=0 is top.
        # Flip so INCLUDE_PARAMS stays bottom -> top like the sample figure.
        corr_mat = np.flipud(corr_mat)
        annot_mat = np.flipud(annot_mat)
        y_labels = list(reversed(params_loaded))

        finite_vals = corr_mat[np.isfinite(corr_mat)]

        if finite_vals.size == 0:
            print(f"[SKIP] All values are NA for {proxy_label} | {method}")
            continue

        if FIXED_VMIN is None or FIXED_VMAX is None:
            vmax_abs = float(np.nanmax(np.abs(finite_vals)))
            vmin = -vmax_abs
            vmax = vmax_abs
        else:
            vmin = FIXED_VMIN
            vmax = FIXED_VMAX
            vmax_abs = max(abs(FIXED_VMIN), abs(FIXED_VMAX))

        # ----------------------------
        # Plot
        # ----------------------------
        plt.figure(figsize=(FIG_W, FIG_H))

        im = plt.imshow(
            corr_mat,
            aspect="auto",
            cmap=CMAP,
            vmin=vmin,
            vmax=vmax
        )

        plt.xticks(
            range(len(station_order)),
            station_labels,
            rotation=45,
            ha="right",
            fontsize=XTICK_FONTSIZE,
            fontweight="bold"
        )

        plt.yticks(
            range(len(y_labels)),
            y_labels,
            fontsize=YTICK_FONTSIZE,
            fontweight="bold"
        )

        plt.title(
            f"{method_title(method)}: {proxy_label} maximal correlation (r) + optimal time window",
            fontsize=TITLE_FONTSIZE,
            fontweight="bold"
        )

        cbar = plt.colorbar(im, fraction=0.03, pad=0.02)
        cbar.set_label(
            f"{method_title(method)} r (maximal_calculated_metric)",
            fontsize=CBAR_FONTSIZE,
            fontweight="bold"
        )
        cbar.ax.tick_params(labelsize=CBAR_FONTSIZE)

        ax = plt.gca()

        # White cell gridlines
        ax.set_xticks(np.arange(-0.5, len(station_order), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(y_labels), 1), minor=True)

        plt.grid(
            which="minor",
            color="white",
            linestyle="-",
            linewidth=1
        )

        plt.tick_params(
            which="minor",
            bottom=False,
            left=False
        )

        # Cell annotations
        for i in range(len(y_labels)):
            for j in range(len(station_order)):

                txt = annot_mat[i, j]

                if txt == "NA":
                    continue

                val = corr_mat[i, j]
                text_color = choose_text_color(val, vmax_abs)

                plt.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=ANNOT_FONTSIZE,
                    color=text_color,
                    fontweight="bold"
                )

        plt.tight_layout()

        out_png = os.path.join(
            output_dir,
            f"{output_prefix}_{APPROACH_TO_PLOT.lower()}_station_heatmap_{method}.png"
        )

        plt.savefig(
            out_png,
            dpi=400,
            bbox_inches="tight",
            facecolor="white"
        )

        plt.close()

        print(f"Saved heatmap:\n{out_png}")


# ============================================================
# RUN
# ============================================================

def main():

    print("\nCreating d18O station heatmaps...\n")

    create_station_heatmap(
        master_xlsx=D18_MASTER_XLSX,
        output_dir=D18_OUTPUT_DIR,
        proxy_label="d18O",
        output_prefix="d18O_climate",
        station_display=STATION_DISPLAY_BASE,
        drop_stations=DROP_STATIONS_BASE
    )

    print("\nCreating TRW station heatmaps...\n")

    create_station_heatmap(
        master_xlsx=TRW_MASTER_XLSX,
        output_dir=TRW_OUTPUT_DIR,
        proxy_label="TRW",
        output_prefix="TRW_climate",
        station_display=STATION_DISPLAY_TRW,
        drop_stations=DROP_STATIONS_TRW
    )

    print("\nDone. All heatmaps saved.")


if __name__ == "__main__":
    main()