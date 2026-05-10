"""
Creates 3 plots:

1) All individual δ18O samples + yearly mean
2) Mean ± standard deviation (shaded band)
3) All individual δ18O samples + yearly mean, with missing values breaking the lines

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
D18O_XLSX = r"E:\FAU master\Master Thesis\Data\d18o Data\d18o_per_sample_sorted_cleaned.xlsx"

# OUT_DIR = r"E:\FAU master\Master Thesis\Plots"
OUT_DIR = r"E:\FAU master\Master Thesis\Results\d18o new narrow missing removed"
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


def main():
    df = pd.read_excel(D18O_XLSX).sort_values("Year")

    # OPTIONAL: remove outliers
    df.loc[df["Year"].isin([2016, 2017]), "HNC_25a"] = pd.NA

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

    for s in SAMPLES:
        df[s] = pd.to_numeric(df[s], errors="coerce")

    df = df.dropna(subset=["Year"]).copy()
    df["Year"] = df["Year"].astype(int)

    # ---- compute statistics for original plots
    # This calculates the mean/std from available values.
    # Example: if one sample is missing, the mean is calculated from the remaining samples.
    df["mean_d18O"] = df[SAMPLES].mean(axis=1, skipna=True)
    df["std_d18O"] = df[SAMPLES].std(axis=1, skipna=True)

    # ---- compute statistics for removed-missing plots
    # This only calculates mean/std when all sample values exist.
    # If any sample is missing in that year, mean/std become NaN,
    # which makes Matplotlib break the line and shaded band.
    df["mean_d18O_complete"] = df[SAMPLES].mean(axis=1, skipna=False)
    df["std_d18O_complete"] = df[SAMPLES].std(axis=1, skipna=False)

    year_min = int(df["Year"].min())
    year_max = int(df["Year"].max())
    xticks = list(range(year_min, year_max + 1, 3))

    # ============================================================
    # PLOT 1: All samples + mean
    # Original version
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

    plt.xlabel("Year")
    plt.ylabel(YLABEL)
    plt.title("δ$^{18}$O per Year (Tree-Ring Samples) + Mean")
    plt.legend(title="Series", fontsize=8.5, loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.xticks(xticks)
    plt.tight_layout()
    plt.savefig(OUT_PNG_1, dpi=500)
    plt.close()
    print(f"Saved: {OUT_PNG_1}")

    # ============================================================
    # PLOT 2: Mean ± Std
    # Original version
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

    plt.xlabel("Year")
    plt.ylabel(YLABEL)
    plt.title("δ$^{18}$O per Year — Mean ± Standard Deviation")
    plt.legend(title="Series", fontsize=9, loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.xticks(xticks)
    plt.tight_layout()
    plt.savefig(OUT_PNG_2, dpi=500)
    plt.close()
    print(f"Saved: {OUT_PNG_2}")

    # ============================================================
    # PLOT 3: All samples + mean
    # Removed-missing version
    # Missing values break the sample lines.
    # Mean line also breaks if any sample is missing in that year.
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

    plt.xlabel("Year")
    plt.ylabel(YLABEL)
    plt.title("δ$^{18}$O per Year (Tree-Ring Samples) + Mean")
    plt.legend(title="Series", fontsize=8.5, loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.xticks(xticks)
    plt.tight_layout()
    plt.savefig(OUT_PNG_1_REMOVED_MISSING, dpi=500)
    plt.close()
    print(f"Saved: {OUT_PNG_1_REMOVED_MISSING}")



if __name__ == "__main__":
    main()