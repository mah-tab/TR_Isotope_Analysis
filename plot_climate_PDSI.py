"""
monthly_mean_pdsi.py

Creates a monthly mean PDSI graph for Baft.

Data structure:
Year | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec

The script:
1. Reads the PDSI Excel file
2. Assigns the year and month column names
3. Converts values to numeric
4. Keeps only the period 1989-2023 to match the Baft climate data
5. Calculates the mean PDSI for each month over all available years
6. Plots the monthly mean PDSI as a blue line
7. Saves the plot as PNG with 600 dpi
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Input and output paths
# -----------------------------
input_file = r"E:\FAU master\Master Thesis\Data\climate data\pdsi.xlsx"

output_dir = r"E:\FAU master\Master Thesis\Plots\climate"
os.makedirs(output_dir, exist_ok=True)

output_png = os.path.join(output_dir, "Baft_PDSI.png")

# -----------------------------
# Read Excel file
# The file has no header row
# -----------------------------
df = pd.read_excel(input_file, header=None)

# -----------------------------
# Assign column names
# First column = Year
# Following 12 columns = January to December
# -----------------------------
month_columns = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

df.columns = ["Year"] + month_columns

# -----------------------------
# Convert columns to numeric
# This also handles comma decimals if present
# -----------------------------
for col in df.columns:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

# -----------------------------
# Keep only the period 1989-2023
# This matches the available period of Baft_clim.xlsx
# -----------------------------
df = df[
    (df["Year"] >= 1989) &
    (df["Year"] <= 2023)
].copy()

print(
    f"PDSI period used: "
    f"{int(df['Year'].min())}-{int(df['Year'].max())}"
)

# -----------------------------
# Calculate monthly averages over all years
# For example: mean January PDSI across 1989-2023
# -----------------------------
monthly_mean = df[month_columns].mean(
    axis=0,
    skipna=True
)

# -----------------------------
# Colors
# -----------------------------
pdsi_color = "#1f77b4"      # blue

# -----------------------------
# Create plot
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 6))

# -----------------------------
# Plot monthly mean PDSI
# -----------------------------
ax.plot(
    range(1, 13),
    monthly_mean.values,
    color=pdsi_color,
    marker="o",
    linestyle="-",
    linewidth=2.5,
    markersize=6
)

ax.set_xlabel(
    "Month",
    fontsize=15,
    fontweight="bold",
    color="black"
)

ax.set_ylabel(
    "PDSI",
    fontsize=15,
    fontweight="bold",
    color="black"
)

# -----------------------------
# Axis ticks
# -----------------------------
ax.tick_params(
    axis="x",
    labelsize=14,
    labelcolor="black"
)

ax.tick_params(
    axis="y",
    labelsize=14,
    labelcolor="black"
)

ax.set_xticks(range(1, 13))

ax.set_xticklabels(
    month_columns,
    fontsize=14,
    color="black"
)

# -----------------------------
# Title and grid
# -----------------------------
plt.title(
    "Monthly Mean PDSI — Baft",
    fontsize=15,
    fontweight="bold"
)

ax.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

# -----------------------------
# Save figure
# -----------------------------
plt.savefig(
    output_png,
    dpi=600,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {output_png}")