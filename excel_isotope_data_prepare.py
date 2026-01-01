import os
import pandas as pd

input_path = r"E:\FAU master\Master Thesis\All files and Data\Shared with all\Shared with all\d18O_2511_Mahtab Arjomandi.xlsx"
output_dir = r"E:\FAU master\Master Thesis\Data"

samples = ["HNC_24a", "HNC_25a", "HNC_28a", "HNC_53a", "HNC_58b"]
years = list(range(1974, 2024))  # 1974..2023

# Read excel (your headers start on row 2)
df = pd.read_excel(input_path, sheet_name=0, header=1)

# Extract sample name from Identifier 1: e.g. "HNC_28a_1974" -> "HNC_28a"
df["sample"] = df["Identifier 1"].astype(str).str.split("_").str[0:2].str.join("_")

# Keep only relevant rows and ensure year is numeric
df = df[df["sample"].isin(samples)].copy()
df["year"] = pd.to_numeric(df["year"], errors="coerce")

os.makedirs(output_dir, exist_ok=True)

# Variable -> (input column name, output filename)
exports = {
    "d18o": ("d18O_VSMOV", "d18o_per_sample_sorted.xlsx"),
    "amount": ("Amount", "amount_per_sample_sorted.xlsx"),
    "oxygen_percentage": ("%O", "oxygen_percentage_per_sample_sorted.xlsx"),
}

for _, (value_col, out_name) in exports.items():
    if value_col not in df.columns:
        raise ValueError(
            f"Column '{value_col}' not found in the Excel file. "
            f"Available columns: {list(df.columns)}"
        )

    # Reshape to wide table (will error if duplicates exist for same year+sample)
    wide = df.pivot(index="year", columns="sample", values=value_col)

    # Enforce year range + sample order
    wide = wide.reindex(years)[samples]

    # Final table: Year + sample columns
    out = wide.reset_index().rename(columns={"year": "Year"})

    # Save (NaN becomes blank cells in Excel)
    out_path = os.path.join(output_dir, out_name)
    out.to_excel(out_path, index=False)
    print(f"Saved: {out_path}")
