import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# -----------------------------
# Inputs
# -----------------------------
# files = {
#     "d18o": r"E:\FAU master\Master Thesis\Data\d18o_per_sample_sorted_corrected_narrow_next_year.xlsx",
#     "amount": r"E:\FAU master\Master Thesis\Data\amount_per_sample_sorted_corrected.xlsx",
#     "oxygen_percentage": r"E:\FAU master\Master Thesis\Data\oxygen_percentage_per_sample_sorted_corrected.xlsx",
# }

data_files = {
    "d18o": r"E:\FAU master\Master Thesis\Data\d18o Data\new\Henza_O_corrected_final.xlsx",
    "oxygen_percentage": r"E:\FAU master\Master Thesis\Data\d18o Data\oxygen_percentage_per_sample_sorted_corrected.xlsx",
    "amount": r"E:\FAU master\Master Thesis\Data\d18o Data\amount_per_sample_sorted_corrected.xlsx",
}

# out_dir = r"E:\FAU master\Master Thesis\Plots"
out_dir = r"E:\FAU master\Master Thesis\Results\d18o new narrow missing removed\new_raw_final"
os.makedirs(out_dir, exist_ok=True)

samples = ["HNC_24a", "HNC_25a", "HNC_28a", "HNC_53a", "HNC_58b"]

plot_meta = {
    "d18o": {
        "ylabel": "δ$^{18}$O (‰)",
        "title": "δ$^{18}$O per Year (Tree-Ring Samples)",
        "filename": "d18o_per_year_samples.png",
        "subplot_filename": "d18o_per_year_subplots_separate.png",
        "subplot_title": "δ$^{18}$O per Year — Separate Sample Subplots",
        "subplot_missing_filename": "d18o_per_year_subplots_missing_gaps.png",
        "subplot_missing_title": "δ$^{18}$O per Year — Separate Sample Subplots with Missing-Value Gaps",
    },
    "amount": {
        "ylabel": "Amount",
        "title": "Amount per Year (Tree-Ring Samples)",
        "filename": "amount_per_year_samples.png",
        "subplot_filename": "amount_per_year_subplots_separate.png",
        "subplot_title": "Amount per Year — Separate Sample Subplots",
        "subplot_missing_filename": "amount_per_year_subplots_missing_gaps.png",
        "subplot_missing_title": "Amount per Year — Separate Sample Subplots with Missing-Value Gaps",
    },
    "oxygen_percentage": {
        "ylabel": "%O",
        "title": "%O per Year (Tree-Ring Samples)",
        "filename": "oxygen_percentage_per_year_samples.png",
        "subplot_filename": "oxygen_percentage_per_year_subplots_separate.png",
        "subplot_title": "%O per Year — Separate Sample Subplots",
        "subplot_missing_filename": "oxygen_percentage_per_year_subplots_missing_gaps.png",
        "subplot_missing_title": "%O per Year — Separate Sample Subplots with Missing-Value Gaps",
    },
}

# fixed color mapping, consistent across plots
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
    # This plot closes gaps, as in your original code
    # -----------------------------
    plt.figure(figsize=(10, 6))

    for sample in samples:
        mask = df[sample].notna()
        x = df.loc[mask, "Year"]
        y = df.loc[mask, sample]

        plt.plot(
            x,
            y,
            marker="o",
            linestyle="-",
            label=sample,
            color=color_map[sample],
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
    # Plot 2: Separate subplots, 2 cols x 3 rows
    # This plot also closes gaps, as in your original code
    # - 5 samples get their own subplot
    # - 6th bottom-right subplot is legend panel
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
            x,
            y,
            marker="o",
            linestyle="-",
            color=color_map[sample],
            label=sample,
        )

        ax.set_title(sample)
        ax.set_ylabel(plot_meta[key]["ylabel"])
        ax.set_xlabel("Year")
        ax.set_xticks(xticks_sub)
        ax.grid(True, alpha=0.3)

    # Legend panel, 6th subplot: bottom-right
    legend_ax = axes[5]
    legend_ax.axis("off")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=color_map[s],
            marker="o",
            linestyle="-",
            linewidth=2.5,
            markersize=8,
            label=s,
        )
        for s in samples
    ]

    legend_ax.legend(
        handles=legend_handles,
        title="Sample (Color Codes)",
        loc="center",
        frameon=False,
        fontsize=12,
        title_fontsize=13,
        markerscale=1.4,
    )

    fig.suptitle(plot_meta[key]["subplot_title"], fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    sub_out_path = os.path.join(out_dir, plot_meta[key]["subplot_filename"])
    fig.savefig(sub_out_path, dpi=500)
    plt.close(fig)
    print(f"Saved: {sub_out_path}")

    # -----------------------------
    # Plot 3: Separate subplots with missing-value gaps
    # This is the additional new plot.
    # - Keeps full Year axis
    # - Does not plot NA / missing values
    # - Shows scatter dots where values exist
    # - Breaks lines wherever values are missing
    # -----------------------------
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 10), sharex=False)
    axes = axes.flatten()

    for i, sample in enumerate(samples):
        ax = axes[i]

        # Full x-axis, including years with missing values
        x_all = df["Year"]

        # Convert to numeric so missing values become NaN.
        # Matplotlib automatically breaks lines at NaN values.
        y_all = pd.to_numeric(df[sample], errors="coerce")

        ax.plot(
            x_all,
            y_all,
            linestyle="-",
            color=color_map[sample],
            label=sample,
        )

        # Scatter dots only where real sample values exist
        mask = y_all.notna()

        ax.scatter(
            df.loc[mask, "Year"],
            y_all.loc[mask],
            color=color_map[sample],
            s=30,
        )

        ax.set_title(sample)
        ax.set_ylabel(plot_meta[key]["ylabel"])
        ax.set_xlabel("Year")
        ax.set_xticks(xticks_sub)
        ax.grid(True, alpha=0.3)

    # Legend panel, 6th subplot: bottom-right
    legend_ax = axes[5]
    legend_ax.axis("off")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=color_map[s],
            marker="o",
            linestyle="-",
            linewidth=2.5,
            markersize=8,
            label=s,
        )
        for s in samples
    ]

    legend_ax.legend(
        handles=legend_handles,
        title="Sample (Color Codes)",
        loc="center",
        frameon=False,
        fontsize=12,
        title_fontsize=13,
        markerscale=1.4,
    )

    fig.suptitle(plot_meta[key]["subplot_missing_title"], fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    missing_sub_out_path = os.path.join(
        out_dir,
        plot_meta[key]["subplot_missing_filename"],
    )

    fig.savefig(missing_sub_out_path, dpi=500)
    plt.close(fig)
    print(f"Saved: {missing_sub_out_path}")


# -----------------------------
# Generate all plots
# -----------------------------
for key, path in data_files.items():
    plot_variable(path, key)