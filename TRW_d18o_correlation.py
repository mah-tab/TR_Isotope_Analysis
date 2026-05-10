"""
Correlation analysis between tree-ring width and mean δ18O.

This script:
1) Reads TRW chronology Excel file
2) Calculates Spearman correlation
3) Calculates Pearson correlation as a secondary check
4) Creates and saves a correlation scatter plot with a fitted trend line
5) Saves the correlation results as a text file

Input Excel structure:
Column 1 = Year
Column 2 = Tree-ring width in mm
Column 3 = Mean δ18O

Author: Mahtab Arjomandi
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr


# -----------------------------
# Input / Output paths
# -----------------------------
INPUT_XLSX = r"E:\FAU master\Master Thesis\Data\Tree Ring Width Chronology\TRW_chronology.xlsx"

OUT_DIR = r"E:\FAU master\Master Thesis\Results\TRW"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PNG = os.path.join(
    OUT_DIR,
    "trw_mean_d18o_correlation_plot.png"
)

OUT_TXT = os.path.join(
    OUT_DIR,
    "trw_mean_d18o_correlation_results.txt"
)


# -----------------------------
# Plot settings
# -----------------------------
POINT_COLOR = "darkgreen"
LINE_COLOR = "deepskyblue"

DPI = 600

XLABEL = "Tree-Ring Width (mm)"
YLABEL = "Mean δ$^{18}$O (‰)"


def main():
    # -----------------------------
    # Read data
    # -----------------------------
    df = pd.read_excel(INPUT_XLSX)

    # Use first three columns regardless of original column names
    df = df.iloc[:, :3].copy()
    df.columns = ["Year", "TRW", "mean_d18O"]

    # Convert to numeric
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["TRW"] = pd.to_numeric(df["TRW"], errors="coerce")
    df["mean_d18O"] = pd.to_numeric(df["mean_d18O"], errors="coerce")

    # Keep only complete paired values
    df_corr = df.dropna(subset=["Year", "TRW", "mean_d18O"]).copy()

    # -----------------------------
    # Correlation calculations
    # -----------------------------
    x = df_corr["TRW"]
    y = df_corr["mean_d18O"]

    spearman_rho, spearman_p = spearmanr(x, y)
    pearson_r, pearson_p = pearsonr(x, y)

    n = len(df_corr)

    print("Correlation between Tree-Ring Width and Mean δ18O")
    print("-------------------------------------------------")
    print(f"n = {n}")
    print(f"Spearman rho = {spearman_rho:.4f}")
    print(f"Spearman p-value = {spearman_p:.4f}")
    print()
    print(f"Pearson r = {pearson_r:.4f}")
    print(f"Pearson p-value = {pearson_p:.4f}")

    # -----------------------------
    # Save results as text file
    # -----------------------------
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("Correlation between Tree-Ring Width and Mean δ18O\n")
        f.write("-------------------------------------------------\n")
        f.write(f"Input file: {INPUT_XLSX}\n")
        f.write(f"Number of complete year pairs: n = {n}\n\n")

        f.write("Primary correlation: Spearman\n")
        f.write(f"Spearman rho = {spearman_rho:.4f}\n")
        f.write(f"Spearman p-value = {spearman_p:.4f}\n\n")

        f.write("Secondary correlation: Pearson\n")
        f.write(f"Pearson r = {pearson_r:.4f}\n")
        f.write(f"Pearson p-value = {pearson_p:.4f}\n\n")

        f.write("Interpretation:\n")
        if spearman_p < 0.05:
            f.write(
                "There is a statistically significant monotonic correlation "
                "between tree-ring width and mean δ18O.\n"
            )
        else:
            f.write(
                "There is no statistically significant monotonic correlation "
                "between tree-ring width and mean δ18O.\n"
            )

    print(f"\nSaved results: {OUT_TXT}")

    # -----------------------------
    # Linear trend line for visualization
    # -----------------------------
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept

    # -----------------------------
    # Correlation plot
    # -----------------------------
    plt.figure(figsize=(8, 6))

    plt.scatter(
        x,
        y,
        color=POINT_COLOR,
        s=55,
        alpha=0.85,
        label="Yearly data points"
    )

    plt.plot(
        x_line,
        y_line,
        color=LINE_COLOR,
        alpha=0.4,
        linewidth=2.5,
        label="Linear trend"
    )

    # Spearman annotation box only
    annotation_text = (
        f"Spearman ρ = {spearman_rho:.3f}, p = {spearman_p:.3f}\n"
        f"n = {n}"
    )

    # Spearman box: top left
    plt.text(
        0.02,
        0.98,
        annotation_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.8
        )
    )

    plt.xlabel(XLABEL, fontsize=12)
    plt.ylabel(YLABEL, fontsize=12)
    plt.title("Correlation Between Tree-Ring Width and Mean δ$^{18}$O")

    # Normal legend: bottom left
    plt.legend(loc="lower left", fontsize=9)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(OUT_PNG, dpi=DPI)
    plt.close()

    print(f"Saved plot: {OUT_PNG}")


if __name__ == "__main__":
    main()