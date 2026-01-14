import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Paths
# -----------------------------
in_file = r"E:\FAU master\Master Thesis\Data\climate data\kerman.xlsx"
out_dir = r"E:\FAU master\Master Thesis\Plots"
os.makedirs(out_dir, exist_ok=True)

# -----------------------------
# Read climate data
# -----------------------------
df = pd.read_excel(in_file)

# Basic checks
required_cols = ["Year", "Month", "T_Mean", "T_Max", "T_Min", "Precip", "RH", "VPD", "es", "ea"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in climate file: {missing}\nFound columns: {list(df.columns)}")

# Ensure numeric (robust to comma-decimals just in case)
def to_numeric_safe(s):
    if s.dtype == "object":
        s = s.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

df["Year"] = to_numeric_safe(df["Year"]).astype("Int64")
df["Month"] = to_numeric_safe(df["Month"]).astype("Int64")

for c in ["T_Mean", "T_Max", "T_Min", "Precip", "RH", "VPD", "es", "ea"]:
    df[c] = to_numeric_safe(df[c])

# Drop rows without Year
df = df.dropna(subset=["Year"]).copy()
df = df.sort_values(["Year", "Month"])

# -----------------------------
# Compute yearly means (over months)
# -----------------------------
df["T_Delta"] = df["T_Max"] - df["T_Min"]  # monthly Tmax - Tmin

yearly = df.groupby("Year", as_index=False).agg(
    T_Mean=("T_Mean", "mean"),
    Precip=("Precip", "mean"),   # requested: yearly mean over months
    RH=("RH", "mean"),
    VPD=("VPD", "mean"),
    es=("es", "mean"),
    ea=("ea", "mean"),
    T_Delta=("T_Delta", "mean"),
)

# Keep only years with some data (optional safety)
yearly = yearly.dropna(subset=["Year"]).copy()

# X tick positions every 3 years
year_min = int(yearly["Year"].min())
year_max = int(yearly["Year"].max())
#xticks_3y = list(range(year_min, year_max + 1, 3))
# X tick positions
xticks_3y = list(range(year_min, year_max + 1, 3))  # for single plots
xticks_5y = list(range(year_min, year_max + 1, 5))  # for subplots


# -----------------------------
# Plot settings
# -----------------------------
plot_specs = [
    ("T_Mean",  "Annual mean T_Mean",            "T_Mean (°C)",          "climate_T_Mean_means.png",   "red"),
    ("Precip",  "Annual mean Precip",            "Precip (mean)",        "climate_Precip_means.png",   "blue"),
    ("RH",      "Annual mean RH",                "RH (%)",               "climate_RH_means.png",       "green"),
    ("VPD",     "Annual mean VPD",               "VPD",                  "climate_VPD_means.png",      "purple"),
    ("es",      "Annual mean es",                "es",                   "climate_es_means.png",       "brown"),
    ("ea",      "Annual mean ea",                "ea",                   "climate_ea_means.png",       "gray"),
    ("T_Delta", "Annual mean (T_Max − T_Min)",   "T_Max − T_Min (°C)",   "climate_TmaxTmin_means.png", "orange"),
]

# -----------------------------
# 1) Separate plots (one per variable)
# -----------------------------
for col, title, ylabel, filename, color in plot_specs:
    plt.figure(figsize=(10, 5))
    plt.plot(yearly["Year"], yearly[col], marker="o", linestyle="-", color=color, label=col)

    plt.title(title, fontsize=14)
    plt.xlabel("Year")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.xticks(xticks_3y, rotation=0)
    plt.tight_layout()

    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

# -----------------------------
# 2) Combined plot: 7 subplots + legend bottom-right
#    (2 columns x 4 rows; last axis used for legend)
# -----------------------------
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(14, 14))
axes = axes.flatten()

handles_for_legend = []
labels_for_legend = []

for i, (col, title, ylabel, _, color) in enumerate(plot_specs):
    ax = axes[i]
    line = ax.plot(yearly["Year"], yearly[col], marker="o", linestyle="-", color=color, label=col)[0]

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.set_xticks(xticks_5y)

    handles_for_legend.append(line)
    labels_for_legend.append(col)

# Use the last (8th) subplot for legend (since we only need 7 plots)
legend_ax = axes[len(plot_specs)]
legend_ax.axis("off")
legend_ax.legend(
    handles_for_legend,
    labels_for_legend,
    loc="lower right",
    frameon=True,
    fontsize=14
)

fig.suptitle("Kerman climate: yearly means (over months) – 1974–2023", y=0.995, fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.98])

combined_path = os.path.join(out_dir, "climate_all_variables_yearly_means_subplots.png")
fig.savefig(combined_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {combined_path}")
