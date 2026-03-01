"""
Creates TWO plots:

1) All individual δ18O samples + yearly mean
2) Mean ± standard deviation (shaded band)

Author: Mahtab Arjomandi
Date: 01.03.2026
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Inputs
# -----------------------------
D18O_XLSX = r"E:\FAU master\Master Thesis\Data\d18o_per_sample_sorted_corrected_missing.xlsx"

OUT_DIR = r"E:\FAU master\Master Thesis\Plots"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PNG_1 = os.path.join(OUT_DIR, "d18o_per_year_samples_plus_mean.png")
OUT_PNG_2 = os.path.join(OUT_DIR, "d18o_mean_plus_minus_std.png")

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

    # ---- compute statistics
    df["mean_d18O"] = df[SAMPLES].mean(axis=1, skipna=True)
    df["std_d18O"] = df[SAMPLES].std(axis=1, skipna=True)

    year_min = int(df["Year"].min())
    year_max = int(df["Year"].max())
    xticks = list(range(year_min, year_max + 1, 3))

    # ============================================================
    # PLOT 1: All samples + mean
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
    plt.legend(title="Series")
    plt.grid(True, alpha=0.3)
    plt.xticks(xticks)
    plt.tight_layout()
    plt.savefig(OUT_PNG_1, dpi=500)
    plt.close()
    print(f"Saved: {OUT_PNG_1}")

    # ============================================================
    # PLOT 2: Mean ± Std (shaded)
    # ============================================================
    plt.figure(figsize=(10, 6))

    # shaded region
    plt.fill_between(
        df["Year"],
        df["mean_d18O"] - df["std_d18O"],
        df["mean_d18O"] + df["std_d18O"],
        color="0.8",           # light grey
        alpha=0.6,
        label="±STD"
    )

    # mean line
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
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(xticks)
    plt.tight_layout()
    plt.savefig(OUT_PNG_2, dpi=500)
    plt.close()
    print(f"Saved: {OUT_PNG_2}")


if __name__ == "__main__":
    main()