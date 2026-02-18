import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# -----------------------------
# Inputs
# -----------------------------
files = {
    "d18o": r"E:\FAU master\Master Thesis\Data\d18o_per_sample_sorted_corrected_narrow_next_year.xlsx",
    "amount": r"E:\FAU master\Master Thesis\Data\amount_per_sample_sorted_corrected.xlsx",
    "oxygen_percentage": r"E:\FAU master\Master Thesis\Data\oxygen_percentage_per_sample_sorted_corrected.xlsx",
}

out_dir = r"E:\FAU master\Master Thesis\Plots"
os.makedirs(out_dir, exist_ok=True)

samples = ["HNC_24a", "HNC_25a", "HNC_28a", "HNC_53a", "HNC_58b"]

plot_meta = {
    "d18o": {
        "ylabel": "δ$^{18}$O (‰)",
        "title": "δ$^{18}$O per Year (Tree-Ring Samples)",
        "filename": "d18o_per_year_samples.png",
        "subplot_filename": "d18o_per_year_subplots_separate.png",
        "subplot_title": "δ$^{18}$O per Year — Separate Sample Subplots",
    },
    "amount": {
        "ylabel": "Amount",
        "title": "Amount per Year (Tree-Ring Samples)",
        "filename": "amount_per_year_samples.png",
        "subplot_filename": "amount_per_year_subplots_separate.png",
        "subplot_title": "Amount per Year — Separate Sample Subplots",
    },
    "oxygen_percentage": {
        "ylabel": "%O",
        "title": "%O per Year (Tree-Ring Samples)",
        "filename": "oxygen_percentage_per_year_samples.png",
        "subplot_filename": "oxygen_percentage_per_year_subplots_separate.png",
        "subplot_title": "%O per Year — Separate Sample Subplots",
    },
}

# fixed color mapping (consistent across plots)
color_map = {
    "HNC_24a": "C0",
    "HNC_25a": "C1",
    "HNC_28a": "C2",
    "HNC_53a": "C3",
    "HNC_58b": "C4",
}

def plot_variable(excel_path: str, key: str) -> None:
    df = pd.read_excel(excel_path).sort_values("Year")

    # =================================================
    # OPTIONAL: DROP OUTLIERS (HNC_25a in 2016 & 2017)
    # Comment / uncomment this line as needed
    # =================================================
    df.loc[df["Year"].isin([2016, 2017]), "HNC_25a"] = pd.NA
    # =================================================

    # determine tick positions every 3 years
    year_min = int(df["Year"].min())
    year_max = int(df["Year"].max())
    xticks = list(range(year_min, year_max + 1, 3))

    # -----------------------------
    # Plot 1: All samples in one plot
    # -----------------------------
    plt.figure(figsize=(10, 6))

    for sample in samples:
        mask = df[sample].notna()
        x = df.loc[mask, "Year"]
        y = df.loc[mask, sample]

        plt.plot(
            x, y,
            marker="o",
            linestyle="-",
            label=sample,
            color=color_map[sample]
        )

    plt.xlabel("Year")
    plt.ylabel(plot_meta[key]["ylabel"])
    plt.title(plot_meta[key]["title"])
    plt.legend(title="Sample")
    plt.grid(True, alpha=0.3)
    plt.xticks(xticks)
    plt.tight_layout()

    out_path = os.path.join(out_dir, plot_meta[key]["filename"])
    plt.savefig(out_path, dpi=500)
    plt.close()
    print(f"Saved: {out_path}")

    # -----------------------------
    # Plot 2: Separate subplots (2 cols x 3 rows)
    # - 5 samples get their own subplot
    # - 6th (bottom-right) is legend panel
    # -----------------------------
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 10), sharex=False)
    axes = axes.flatten()

    # x-ticks every 5 years for subplots
    xticks_sub = list(range(year_min, year_max + 1, 5))

    for i, sample in enumerate(samples):
        ax = axes[i]
        mask = df[sample].notna()
        x = df.loc[mask, "Year"]
        y = df.loc[mask, sample]

        ax.plot(
            x, y,
            marker="o",
            linestyle="-",
            color=color_map[sample],
            label=sample
        )

        ax.set_title(sample)
        ax.set_ylabel(plot_meta[key]["ylabel"])
        ax.set_xlabel("Year")                 # ← show on ALL subplots
        ax.set_xticks(xticks_sub)
        ax.grid(True, alpha=0.3)

    # Legend panel (6th subplot: bottom-right)
    legend_ax = axes[5]
    legend_ax.axis("off")
    legend_handles = [
        Line2D(
            [0], [0],
            color=color_map[s],
            marker="o",
            linestyle="-",
            linewidth=2.5,
            markersize=8,
            label=s
        )
        for s in samples
    ]

    legend_ax.legend(
        handles=legend_handles,
        title="Sample (Color Codes)",
        loc="center",
        frameon=False,
        fontsize=12,          # bigger legend text
        title_fontsize=13,
        markerscale=1.4       # bigger markers in legend
    )

    fig.suptitle(plot_meta[key]["subplot_title"], fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    sub_out_path = os.path.join(out_dir, plot_meta[key]["subplot_filename"])
    fig.savefig(sub_out_path, dpi=500)
    plt.close(fig)
    print(f"Saved: {sub_out_path}")



# -----------------------------
# Generate all plots
# -----------------------------
for key, path in files.items():
    plot_variable(path, key)
