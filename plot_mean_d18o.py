"""
Creates 4 plots:

1) All individual δ18O samples + yearly mean
2) Mean ± standard deviation (shaded band)
3) All individual δ18O samples + yearly mean, with missing values breaking the lines
4) Manually corrected mean chronology ± standard deviation, with missing values breaking the line and shade

Author: Mahtab Arjomandi
Date: 01.03.2026
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Inputs
# -----------------------------
# D18O_XLSX = r"E:\FAU master\Master Thesis\Data\d18o_per_sample_sorted_corrected_missing.xlsx"
# D18O_XLSX = r"E:\FAU master\Master Thesis\Data\d18o Data\d18o_per_sample_sorted_cleaned.xlsx"
D18O_XLSX = r"E:\FAU master\Master Thesis\Data\d18o Data\new\Henza_O_corrected_final.xlsx"

# Additional manually corrected mean chronology file
MEAN_CHRON_XLSX = r"E:\FAU master\Master Thesis\Data\d18o Data\new\Henza_mean_chron_final.xlsx"

# OUT_DIR = r"E:\FAU master\Master Thesis\Plots"
# OUT_DIR = r"E:\FAU master\Master Thesis\Results\d18o new narrow missing removed"
OUT_DIR = r"E:\FAU master\Master Thesis\Results\d18o new narrow missing removed\new_raw_final"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PNG_1 = os.path.join(OUT_DIR, "d18o_per_year_samples_plus_mean.png")
OUT_PNG_2 = os.path.join(OUT_DIR, "d18o_mean_plus_minus_std.png")

OUT_PNG_1_REMOVED_MISSING = os.path.join(
    OUT_DIR,
    "d18o_per_year_samples_plus_mean_removed_missing.png"
)

OUT_PNG_2_REMOVED_MISSING = os.path.join(
    OUT_DIR,
    "d18o_mean_plus_minus_std_removed_missing.png"
)

# Additional output for manually corrected mean chronology
OUT_PNG_MANUAL_MEAN_CHRON_GAPS = os.path.join(
    OUT_DIR,
    "d18o_manual_mean_chron_plus_minus_std_gaps.png"
)

SAMPLES = ["HNC_24a", "HNC_25a", "HNC_28a", "HNC_53a", "HNC_58b"]

COLOR_MAP = {
    "HNC_24a": "C0",
    "HNC_25a": "C1",
    "HNC_28a": "C2",
    "HNC_53a": "C3",
    "HNC_58b": "C4",
}

MEAN_LABEL = "Mean (5 samples)"
MEAN_COLOR = "0.25"
MEAN_LINEWIDTH = 3.0
MEAN_MARKERSIZE = 6

YLABEL = "δ$^{18}$O (‰)"

XLABEL_FONTSIZE = 14
YLABEL_FONTSIZE = 14
TICK_FONTSIZE = 12


def main():
    df = pd.read_excel(D18O_XLSX).sort_values("Year")

    # OPTIONAL: remove outliers
    df.loc[df["Year"].isin([2016, 2017]), "HNC_25a"] = pd.NA

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

    for s in SAMPLES:
        df[s] = pd.to_numeric(df[s], errors="coerce")

    df = df.dropna(subset=["Year"]).copy()
    df["Year"] = df["Year"].astype(int)

    df["mean_d18O"] = df[SAMPLES].mean(axis=1, skipna=True)
    df["std_d18O"] = df[SAMPLES].std(axis=1, skipna=True)

    df["mean_d18O_complete"] = df[SAMPLES].mean(axis=1, skipna=False)
    df["std_d18O_complete"] = df[SAMPLES].std(axis=1, skipna=False)

    year_min = int(df["Year"].min())
    year_max = int(df["Year"].max())
    xticks = list(range(year_min, year_max + 1, 3))

    # ============================================================
    # PLOT 1
    # ============================================================
    plt.figure(figsize=(10, 6))

    for sample in SAMPLES:
        mask = df[sample].notna()

        plt.plot(
            df.loc[mask, "Year"],
            df.loc[mask, sample],
            marker="o",
            linestyle="-",
            label=sample,
            color=COLOR_MAP[sample]
        )

    mask_m = df["mean_d18O"].notna()

    plt.plot(
        df.loc[mask_m, "Year"],
        df.loc[mask_m, "mean_d18O"],
        marker="o",
        linestyle="-",
        color=MEAN_COLOR,
        linewidth=MEAN_LINEWIDTH,
        markersize=MEAN_MARKERSIZE,
        label=MEAN_LABEL,
        zorder=5
    )

    plt.xlabel("Year", fontsize=XLABEL_FONTSIZE)
    plt.ylabel(YLABEL, fontsize=YLABEL_FONTSIZE)
    plt.title("δ$^{18}$O per Year (Tree-Ring Samples) + Mean")
    plt.legend(title="Series", fontsize=8.5, loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.xticks(xticks)
    plt.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    plt.tight_layout()
    plt.savefig(OUT_PNG_1, dpi=500)
    plt.close()
    print(f"Saved: {OUT_PNG_1}")

    # ============================================================
    # PLOT 2
    # ============================================================
    plt.figure(figsize=(10, 6))

    plt.fill_between(
        df["Year"],
        df["mean_d18O"] - df["std_d18O"],
        df["mean_d18O"] + df["std_d18O"],
        color="0.8",
        alpha=0.6,
        label="±STD"
    )

    plt.plot(
        df["Year"],
        df["mean_d18O"],
        marker="o",
        linestyle="-",
        color=MEAN_COLOR,
        linewidth=MEAN_LINEWIDTH,
        markersize=MEAN_MARKERSIZE,
        label=MEAN_LABEL,
        zorder=5
    )

    plt.xlabel("Year", fontsize=XLABEL_FONTSIZE)
    plt.ylabel(YLABEL, fontsize=YLABEL_FONTSIZE)
    plt.title("δ$^{18}$O per Year — Mean ± Standard Deviation")
    plt.legend(title="Series", fontsize=9, loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.xticks(xticks)
    plt.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    plt.tight_layout()
    plt.savefig(OUT_PNG_2, dpi=500)
    plt.close()
    print(f"Saved: {OUT_PNG_2}")

    # ============================================================
    # PLOT 3
    # ============================================================
    plt.figure(figsize=(10, 6))

    for sample in SAMPLES:
        plt.plot(
            df["Year"],
            df[sample],
            marker="o",
            linestyle="-",
            label=sample,
            color=COLOR_MAP[sample]
        )

    plt.plot(
        df["Year"],
        df["mean_d18O_complete"],
        marker="o",
        linestyle="-",
        color=MEAN_COLOR,
        linewidth=MEAN_LINEWIDTH,
        markersize=MEAN_MARKERSIZE,
        label=MEAN_LABEL,
        zorder=5
    )

    plt.xlabel("Year", fontsize=XLABEL_FONTSIZE)
    plt.ylabel(YLABEL, fontsize=YLABEL_FONTSIZE)
    plt.title("δ$^{18}$O per Year (Tree-Ring Samples) + Mean")
    plt.legend(title="Series", fontsize=8.5, loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.xticks(xticks)
    plt.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    plt.tight_layout()
    plt.savefig(OUT_PNG_1_REMOVED_MISSING, dpi=500)
    plt.close()
    print(f"Saved: {OUT_PNG_1_REMOVED_MISSING}")

    # ============================================================
    # PLOT 4
    # ============================================================
    mean_chron_df = pd.read_excel(MEAN_CHRON_XLSX).sort_values("Year")

    mean_chron_df["Year"] = pd.to_numeric(
        mean_chron_df["Year"],
        errors="coerce"
    )

    if "Mean_d18O" in mean_chron_df.columns:
        mean_col = "Mean_d18O"
    elif "O18_raw" in mean_chron_df.columns:
        mean_col = "O18_raw"
    else:
        raise ValueError(
            "Could not find a mean δ18O column. Expected 'Mean_d18O' or 'O18_raw'."
        )

    mean_chron_df[mean_col] = pd.to_numeric(
        mean_chron_df[mean_col],
        errors="coerce"
    )

    mean_chron_df = mean_chron_df.dropna(subset=["Year"]).copy()
    mean_chron_df["Year"] = mean_chron_df["Year"].astype(int)

    possible_std_cols = [
        "std_d18O",
        "STD",
        "Std",
        "std",
        "Standard deviation",
        "standard_deviation",
        "SD",
        "sd"
    ]

    std_col = None

    for col in possible_std_cols:
        if col in mean_chron_df.columns:
            std_col = col
            break

    if std_col is not None:
        mean_chron_df[std_col] = pd.to_numeric(
            mean_chron_df[std_col],
            errors="coerce"
        )
    else:
        mean_chron_df = mean_chron_df.merge(
            df[["Year", "std_d18O_complete"]],
            on="Year",
            how="left"
        )
        std_col = "std_d18O_complete"

    mean_chron_df["manual_mean_lower"] = mean_chron_df[mean_col] - mean_chron_df[std_col]
    mean_chron_df["manual_mean_upper"] = mean_chron_df[mean_col] + mean_chron_df[std_col]

    plt.figure(figsize=(10, 6))

    plt.fill_between(
        mean_chron_df["Year"],
        mean_chron_df["manual_mean_lower"],
        mean_chron_df["manual_mean_upper"],
        color="0.8",
        alpha=0.6,
        label="±STD"
    )

    plt.plot(
        mean_chron_df["Year"],
        mean_chron_df[mean_col],
        marker="o",
        linestyle="-",
        color=MEAN_COLOR,
        linewidth=MEAN_LINEWIDTH,
        markersize=MEAN_MARKERSIZE,
        label="Mean chronology",
        zorder=5
    )

    mean_chron_year_min = int(mean_chron_df["Year"].min())
    mean_chron_year_max = int(mean_chron_df["Year"].max())
    mean_chron_xticks = list(range(mean_chron_year_min, mean_chron_year_max + 1, 3))

    plt.xlabel("Year", fontsize=XLABEL_FONTSIZE)
    plt.ylabel(YLABEL, fontsize=YLABEL_FONTSIZE)
    plt.title("δ$^{18}$O per Year — Mean Chronology ± Standard Deviation")
    plt.legend(title="Series", fontsize=9, loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.xticks(mean_chron_xticks)
    plt.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    plt.tight_layout()
    plt.savefig(OUT_PNG_MANUAL_MEAN_CHRON_GAPS, dpi=500)
    plt.close()

    print(f"Saved: {OUT_PNG_MANUAL_MEAN_CHRON_GAPS}")


if __name__ == "__main__":
    main()