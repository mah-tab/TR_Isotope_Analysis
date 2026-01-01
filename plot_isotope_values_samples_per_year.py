import os
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Inputs
# -----------------------------
files = {
    "d18o": r"E:\FAU master\Master Thesis\Data\d18o_per_sample_sorted.xlsx",
    "amount": r"E:\FAU master\Master Thesis\Data\amount_per_sample_sorted.xlsx",
    "oxygen_percentage": r"E:\FAU master\Master Thesis\Data\oxygen_percentage_per_sample_sorted.xlsx",
}

out_dir = r"E:\FAU master\Master Thesis\Plots"
os.makedirs(out_dir, exist_ok=True)

samples = ["HNC_24a", "HNC_25a", "HNC_28a", "HNC_53a", "HNC_58b"]

plot_meta = {
    "d18o": {
        "ylabel": "δ$^{18}$O (‰)",
        "title": "δ$^{18}$O per Year (Tree-Ring Samples)",
        "filename": "d18o_per_year_samples.png",
    },
    "amount": {
        "ylabel": "Amount",
        "title": "Amount per Year (Tree-Ring Samples)",
        "filename": "amount_per_year_samples.png",
    },
    "oxygen_percentage": {
        "ylabel": "%O",
        "title": "%O per Year (Tree-Ring Samples)",
        "filename": "oxygen_percentage_per_year_samples.png",
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

    # determine tick positions every 3 years
    year_min = int(df["Year"].min())
    year_max = int(df["Year"].max())
    xticks = list(range(year_min, year_max + 1, 3))

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
# Generate all plots
# -----------------------------
for key, path in files.items():
    plot_variable(path, key)
