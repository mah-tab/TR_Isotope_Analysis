import os
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Paths (Windows)
# -----------------------------
data_files = {
    "d18o": r"E:\FAU master\Master Thesis\Data\d18o_per_sample_sorted_corrected_narrow_next_year.xlsx",
    "oxygen_percentage": r"E:\FAU master\Master Thesis\Data\oxygen_percentage_per_sample_sorted_corrected.xlsx",
    "amount": r"E:\FAU master\Master Thesis\Data\amount_per_sample_sorted_corrected.xlsx",
}

output_dir = r"E:\FAU master\Master Thesis\Plots"
os.makedirs(output_dir, exist_ok=True)

# Where to save statistics
stats_output_dir = r"E:\FAU master\Master Thesis\Data"
os.makedirs(stats_output_dir, exist_ok=True)

# -----------------------------
# Styling
# -----------------------------
box_color = "#00BDD6"
median_color = "#FF5E69"

def compute_and_save_stats(df: pd.DataFrame, sample_cols: list, var_key: str) -> None:
    """
    Computes min, Q1, median, mean, Q3, max per sample column (ignoring NaNs),
    prints them, and saves them to an Excel file.
    Output format:
      rows = [min, Q1, median, mean, Q3, max]
      cols = sample names
    """
    stats_index = ["min", "Q1", "median", "mean", "Q3", "max"]
    stats_df = pd.DataFrame(index=stats_index, columns=sample_cols, dtype="float64")

    for s in sample_cols:
        vals = pd.to_numeric(df[s], errors="coerce").dropna()
        if vals.empty:
            continue

        stats_df.loc["min", s] = vals.min()
        stats_df.loc["Q1", s] = vals.quantile(0.25)
        stats_df.loc["median", s] = vals.median()
        stats_df.loc["mean", s] = vals.mean()
        stats_df.loc["Q3", s] = vals.quantile(0.75)
        stats_df.loc["max", s] = vals.max()

    # ---- print nicely ----
    print("\n" + "=" * 70)
    print(f"Statistics per sample for: {var_key}")
    print(stats_df.round(6))
    print("=" * 70 + "\n")

    # ---- save to excel ----
    out_name = f"{var_key}_statistics_per_sample.xlsx"  # (fixed to .xlsx)
    out_path = os.path.join(stats_output_dir, out_name)
    stats_df.to_excel(out_path, index=True)
    print(f"Saved statistics: {out_path}")

def plot_boxplot_from_excel(excel_path: str, title: str, ylabel: str, out_png: str, var_key: str):
    df = pd.read_excel(excel_path)

    # Expect first column to be Year, remaining columns are samples
    sample_cols = [c for c in df.columns if c.lower() != "year"]
    if not sample_cols:
        raise ValueError(f"No sample columns found in: {excel_path}. Columns: {list(df.columns)}")

    # -----------------------------
    # NEW SECTION: statistics per sample (does not affect plotting)
    # -----------------------------
    compute_and_save_stats(df, sample_cols, var_key)

    # Collect values per sample, dropping NaNs (missing years stay out of the distribution)
    data = [pd.to_numeric(df[c], errors="coerce").dropna().values for c in sample_cols]

    plt.figure(figsize=(10, 5))

    plt.boxplot(
        data,
        labels=sample_cols,
        vert=True,
        patch_artist=True,
        showfliers=True,
        boxprops=dict(facecolor=box_color, edgecolor=box_color, linewidth=1.5),
        whiskerprops=dict(color=box_color, linewidth=1.5),
        capprops=dict(color=box_color, linewidth=1.5),
        medianprops=dict(color=median_color, linewidth=2.5),
        flierprops=dict(
            marker="o",
            markersize=5,
            markerfacecolor="none",   # empty dots
            markeredgecolor="black",  # outline color for outliers
            linestyle="none"
        ),
    )

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Sample")
    plt.xticks(rotation=0)
    plt.tight_layout()

    out_path = os.path.join(output_dir, out_png)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

# -----------------------------
# Make the 3 plots (+ stats exports)
# -----------------------------
plot_boxplot_from_excel(
    data_files["d18o"],
    title="d18O per sample (1974–2023)",
    ylabel="d18O (VSMOW)",
    out_png="d18o_samples_boxplot.png",
    var_key="d18o"
)

plot_boxplot_from_excel(
    data_files["oxygen_percentage"],
    title="%O per sample (1974–2023)",
    ylabel="%O",
    out_png="oxygen_percentage_samples_boxplot.png",
    var_key="oxygen_percentage"
)

plot_boxplot_from_excel(
    data_files["amount"],
    title="Amount per sample (1974–2023)",
    ylabel="Amount",
    out_png="amount_samples_boxplot.png",
    var_key="amount"
)
