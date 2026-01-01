import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Path
# -----------------------------
data_path = r"E:\FAU master\Master Thesis\Data\d18o_per_sample_sorted.xlsx"

# -----------------------------
# Read data
# -----------------------------
df = pd.read_excel(data_path)

samples = ["HNC_24a", "HNC_25a", "HNC_28a", "HNC_53a", "HNC_58b"]

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10, 6))

for sample in samples:
    # keep only years where this sample has a value
    mask = df[sample].notna()
    years = df.loc[mask, "Year"]
    values = df.loc[mask, sample]

    plt.plot(
        years,
        values,
        marker="o",
        linestyle="-",
        label=sample
    )

# -----------------------------
# Formatting
# -----------------------------
plt.xlabel("Year")
plt.ylabel("δ$^{18}$O (‰)")
plt.title("δ$^{18}$O Time Series per Tree-Ring Sample")
plt.legend(title="Sample")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
