"""
amberothermic_climate_graph.py

Creates an amberothermic-style climate graph for one climate station.

Data structure:
Year | Month | T_Mean | T_Max | T_Min | Precip | ...

The script:
1. Reads the climate Excel file
2. Converts comma decimals to numeric values
3. Averages each month over all available years
4. Plots:
   - Precipitation as blue bars on the left y-axis
   - T_mean, T_max, T_min as red line plots on the right y-axis
5. Saves the plot as PNG with 600 dpi
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Input and output paths
# -----------------------------
input_file = r"E:\FAU master\Master Thesis\Data\climate data\Baft_clim.xlsx"

output_dir = r"E:\FAU master\Master Thesis\Plots\climate"
os.makedirs(output_dir, exist_ok=True)

output_png = os.path.join(output_dir, "amberothermic_Baft_clim.png")

# -----------------------------
# Read Excel file
# -----------------------------
df = pd.read_excel(input_file)

# -----------------------------
# Convert columns to numeric
# This also handles comma decimals like "12,34"
# -----------------------------
needed_cols = ["Year", "Month", "T_Mean", "T_Max", "T_Min", "Precip"]

for col in needed_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Keep only valid months
df = df[df["Month"].between(1, 12)]

# -----------------------------
# Calculate monthly averages over all years
# For example: mean January precipitation across all years
# -----------------------------
monthly_mean = (
    df.groupby("Month", as_index=False)
    .agg({
        "Precip": "mean",
        "T_Mean": "mean",
        "T_Max": "mean",
        "T_Min": "mean"
    })
)

# Make sure months 1–12 exist and are in correct order
all_months = pd.DataFrame({"Month": range(1, 13)})
monthly_mean = all_months.merge(monthly_mean, on="Month", how="left")

month_labels = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

# -----------------------------
# Colors
# -----------------------------
precip_color = "#1f77b4"      # blue
tmean_color = "#d62728"       # red
tmax_color = "#b22222"        # darker/paler red distinction
tmin_color = "#ff7f7f"        # pale red

# -----------------------------
# Create plot
# -----------------------------
fig, ax_precip = plt.subplots(figsize=(8, 6))

# -----------------------------
# Left y-axis: precipitation bars
# -----------------------------
bars = ax_precip.bar(
    monthly_mean["Month"],
    monthly_mean["Precip"],
    color=precip_color,
    alpha=0.75,
    label="Precipitation"
)

ax_precip.set_xlabel("Month", fontsize=13, fontweight="bold")
ax_precip.set_ylabel("Precipitation (mm)", color=precip_color, fontsize=13, fontweight="bold")
ax_precip.tick_params(axis="y", labelcolor=precip_color)

ax_precip.set_xticks(range(1, 13))
ax_precip.set_xticklabels(month_labels, fontsize=11)

# Give precipitation axis a clean range
precip_max = monthly_mean["Precip"].max()
ax_precip.set_ylim(0, precip_max * 1.25)

# -----------------------------
# Right y-axis: temperature lines
# -----------------------------
ax_temp = ax_precip.twinx()

line_tmean, = ax_temp.plot(
    monthly_mean["Month"],
    monthly_mean["T_Mean"],
    color=tmean_color,
    marker="o",
    linestyle="-",
    linewidth=2.5,
    markersize=6,
    label="T-mean"
)

line_tmax, = ax_temp.plot(
    monthly_mean["Month"],
    monthly_mean["T_Max"],
    color=tmax_color,
    marker="s",
    linestyle="--",
    linewidth=2,
    markersize=5,
    label="T-max"
)

line_tmin, = ax_temp.plot(
    monthly_mean["Month"],
    monthly_mean["T_Min"],
    color=tmin_color,
    marker="s",
    linestyle=":",
    linewidth=2,
    markersize=5,
    label="T-min"
)

ax_temp.set_ylabel("Temperature (°C)", color=tmean_color, fontsize=13, fontweight="bold")
ax_temp.tick_params(axis="y", labelcolor=tmean_color)

# Give temperature axis a clean range
temp_min = monthly_mean[["T_Min", "T_Mean", "T_Max"]].min().min()
temp_max = monthly_mean[["T_Min", "T_Mean", "T_Max"]].max().max()
ax_temp.set_ylim(temp_min - 3, temp_max + 3)

# -----------------------------
# Title, grid, and legend
# -----------------------------
plt.title("Amberothermic Climate Graph — Baft", fontsize=15, fontweight="bold")

ax_precip.grid(axis="y", alpha=0.3)

# Combined legend from both y-axes
handles = [bars, line_tmean, line_tmax, line_tmin]
labels = ["Precipitation", "T-mean", "T-max", "T-min"]

ax_precip.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=4,
    frameon=False,
    fontsize=11
)

plt.tight_layout()

# -----------------------------
# Save figure
# -----------------------------
plt.savefig(output_png, dpi=600, bbox_inches="tight")
plt.close()

print(f"Saved: {output_png}")