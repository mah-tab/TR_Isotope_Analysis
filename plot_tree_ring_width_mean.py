"""
Creates FOUR plots from TRW chronology data:

1) Tree-ring width per year
2) Mean δ18O per year
3) Tree-ring width and mean δ18O as two side-by-side subplots
4) Tree-ring width and mean δ18O on the same plot with two y-axes

Input Excel structure:
Column 1 = Year
Column 2 = Tree-ring width
Column 3 = Mean δ18O

Author: Mahtab Arjomandi
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Input / Output paths
# -----------------------------
INPUT_XLSX = r"E:\FAU master\Master Thesis\Data\Tree Ring Width Chronology\TRW_chronology.xlsx"

OUT_DIR = r"E:\FAU master\Master Thesis\Results\TRW"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_TRW = os.path.join(OUT_DIR, "tree_ring_width_per_year.png")
OUT_D18O = os.path.join(OUT_DIR, "mean_d18o_per_year.png")
OUT_SUBPLOTS = os.path.join(OUT_DIR, "trw_and_mean_d18o_subplots.png")
OUT_DUAL_AXIS = os.path.join(OUT_DIR, "trw_and_mean_d18o_dual_axis.png")


# -----------------------------
# Plot settings
# -----------------------------
TRW_COLOR = "darkgreen"
D18O_COLOR = "deepskyblue"

TRW_LABEL = "Tree-Ring Width"
D18O_LABEL = "Mean δ$^{18}$O"

TRW_YLABEL = "Tree-Ring Width (mm)"
D18O_YLABEL = "Mean δ$^{18}$O (‰)"

DPI = 600

# Font sizes
YLABEL_FONTSIZE = 13
YTICK_FONTSIZE = 11


def main():
    # -----------------------------
    # Read data
    # -----------------------------
    df = pd.read_excel(INPUT_XLSX)

    # Use the first three columns, regardless of their original Excel names
    df = df.iloc[:, :3].copy()
    df.columns = ["Year", "TRW", "mean_d18O"]

    # Convert columns to numeric
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["TRW"] = pd.to_numeric(df["TRW"], errors="coerce")
    df["mean_d18O"] = pd.to_numeric(df["mean_d18O"], errors="coerce")

    # Remove rows without year
    df = df.dropna(subset=["Year"]).copy()
    df["Year"] = df["Year"].astype(int)

    # Sort by year
    df = df.sort_values("Year")

    # X-axis ticks
    year_min = int(df["Year"].min())
    year_max = int(df["Year"].max())

    # Used for plots 1, 2, and 4
    xticks = list(range(year_min, year_max + 1, 3))

    # Used only for plot 3, the side-by-side subplots
    xticks_subplots = list(range(year_min, year_max + 1, 5))

    # ============================================================
    # PLOT 1: Tree-ring width per year
    # ============================================================
    plt.figure(figsize=(10, 6))

    plt.plot(
        df["Year"],
        df["TRW"],
        marker="o",
        linestyle="-",
        color=TRW_COLOR,
        label=TRW_LABEL
    )

    plt.xlabel("Year")
    plt.ylabel(TRW_YLABEL, fontsize=YLABEL_FONTSIZE)
    plt.yticks(fontsize=YTICK_FONTSIZE)
    plt.title("Tree-Ring Width per Year")
    #plt.legend(loc="upper left", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.xticks(xticks)
    plt.tight_layout()
    plt.savefig(OUT_TRW, dpi=DPI)
    plt.close()

    print(f"Saved: {OUT_TRW}")

    # ============================================================
    # PLOT 2: Mean δ18O per year
    # ============================================================
    plt.figure(figsize=(10, 6))

    plt.plot(
        df["Year"],
        df["mean_d18O"],
        marker="o",
        linestyle="-",
        color=D18O_COLOR,
        label=D18O_LABEL
    )

    plt.xlabel("Year")
    plt.ylabel(D18O_YLABEL, fontsize=YLABEL_FONTSIZE)
    plt.yticks(fontsize=YTICK_FONTSIZE)
    plt.title("Mean δ$^{18}$O per Year")
    #plt.legend(loc="upper left", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.xticks(xticks)
    plt.tight_layout()
    plt.savefig(OUT_D18O, dpi=DPI)
    plt.close()

    print(f"Saved: {OUT_D18O}")

    # ============================================================
    # PLOT 3: TRW and mean δ18O as two subplots next to each other
    # Uses x-axis ticks every 5 years
    # ============================================================
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6), sharex=False)

    # Left subplot: TRW
    axes[0].plot(
        df["Year"],
        df["TRW"],
        marker="o",
        linestyle="-",
        color=TRW_COLOR,
        label=TRW_LABEL
    )

    axes[0].set_xlabel("Year")
    axes[0].set_ylabel(TRW_YLABEL, fontsize=YLABEL_FONTSIZE)
    axes[0].tick_params(axis="y", labelsize=YTICK_FONTSIZE)
    axes[0].set_title("Tree-Ring Width per Year")
    #axes[0].legend(loc="upper left", fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(xticks_subplots)

    # Right subplot: mean δ18O
    axes[1].plot(
        df["Year"],
        df["mean_d18O"],
        marker="o",
        linestyle="-",
        color=D18O_COLOR,
        label=D18O_LABEL
    )

    axes[1].set_xlabel("Year")
    axes[1].set_ylabel(D18O_YLABEL, fontsize=YLABEL_FONTSIZE)
    axes[1].tick_params(axis="y", labelsize=YTICK_FONTSIZE)
    axes[1].set_title("Mean δ$^{18}$O per Year")
    #axes[1].legend(loc="upper left", fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(xticks_subplots)

    fig.suptitle("Tree-Ring Width and Mean δ$^{18}$O per Year", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_SUBPLOTS, dpi=DPI)
    plt.close(fig)

    print(f"Saved: {OUT_SUBPLOTS}")

    # ============================================================
    # PLOT 4: TRW and mean δ18O on same plot with two y-axes
    # ============================================================
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left y-axis: TRW
    line1, = ax1.plot(
        df["Year"],
        df["TRW"],
        marker="o",
        linestyle="-",
        color=TRW_COLOR,
        label=TRW_LABEL
    )

    ax1.set_xlabel("Year")
    ax1.set_ylabel(TRW_YLABEL, color=TRW_COLOR, fontsize=YLABEL_FONTSIZE)
    ax1.tick_params(axis="y", labelcolor=TRW_COLOR, labelsize=YTICK_FONTSIZE)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(xticks)

    # Right y-axis: mean δ18O
    ax2 = ax1.twinx()

    line2, = ax2.plot(
        df["Year"],
        df["mean_d18O"],
        marker="o",
        linestyle="-",
        color=D18O_COLOR,
        label=D18O_LABEL
    )

    ax2.set_ylabel(D18O_YLABEL, color=D18O_COLOR, fontsize=YLABEL_FONTSIZE)
    ax2.tick_params(axis="y", labelcolor=D18O_COLOR, labelsize=YTICK_FONTSIZE)

    # Combined legend
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]

    ax1.legend(
        lines,
        labels,
        loc="upper left",
        fontsize=9
    )

    plt.title("Tree-Ring Width and Mean δ$^{18}$O per Year")
    fig.tight_layout()
    fig.savefig(OUT_DUAL_AXIS, dpi=DPI)
    plt.close(fig)

    print(f"Saved: {OUT_DUAL_AXIS}")


if __name__ == "__main__":
    main()