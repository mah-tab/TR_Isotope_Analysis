import pandas as pd
from pathlib import Path


# ============================================================
# Paths
# ============================================================
base_path = Path(r"E:\FAU master\Master Thesis\Data\d18o Data\new")

sample_input_csv = base_path / "Henza_O.csv"
sample_output_xlsx = base_path / "Henza_O_corrected.xlsx"

chron_input_csv = base_path / "henza_chron.csv"
chron_output_csv = base_path / "Henza_mean_chron.csv"
chron_output_xlsx = base_path / "Henza_mean_chron.xlsx"


# ============================================================
# Function to correct d18O values
# ============================================================
def correct_d18o_value(value):
    """
    Corrects d18O values where the decimal point was lost.

    Examples:
    3531049315 -> 35.31049315
    385178677  -> 38.5178677
    37084306   -> 37.084306

    If the value is already a normal decimal, e.g. 35.31049315,
    it is kept unchanged.
    """

    if pd.isna(value):
        return pd.NA

    value_str = str(value).strip()

    if value_str == "" or value_str.upper() == "NA":
        return pd.NA

    # Replace comma decimal separator if present
    value_str = value_str.replace(",", ".")

    # If value already has decimal point, keep it as float
    if "." in value_str:
        try:
            return float(value_str)
        except ValueError:
            return pd.NA

    # Remove possible non-numeric characters
    value_str = "".join(ch for ch in value_str if ch.isdigit())

    if len(value_str) < 3:
        return pd.NA

    # Insert decimal point after first two digits
    corrected_str = value_str[:2] + "." + value_str[2:]

    try:
        return float(corrected_str)
    except ValueError:
        return pd.NA


# ============================================================
# Part 1: Correct raw per-sample d18O file
# ============================================================
print("Processing raw sample file...")

sample_df = pd.read_csv(sample_input_csv, sep=";", dtype=str)

# Remove completely empty columns, often caused by trailing semicolons
sample_df = sample_df.dropna(axis=1, how="all")

# Clean column names
sample_df.columns = sample_df.columns.str.strip()

sample_corrected = sample_df.copy()

for col in sample_corrected.columns:
    if col.lower() == "year":
        sample_corrected[col] = pd.to_numeric(sample_corrected[col], errors="coerce").astype("Int64")
    else:
        sample_corrected[col] = sample_corrected[col].apply(correct_d18o_value)

# Calculate annual mean and number of available trees
tree_columns = [col for col in sample_corrected.columns if col.lower() != "year"]

sample_corrected["Mean_d18O"] = sample_corrected[tree_columns].mean(axis=1, skipna=True)
sample_corrected["N_trees"] = sample_corrected[tree_columns].count(axis=1)

# Save sample corrected file
with pd.ExcelWriter(sample_output_xlsx, engine="openpyxl") as writer:
    sample_corrected.to_excel(writer, sheet_name="Corrected_values", index=False)
    sample_df.to_excel(writer, sheet_name="Original_raw", index=False)

print(f"Saved corrected sample file:\n{sample_output_xlsx}")


# ============================================================
# Part 2: Correct mean chronology file
# ============================================================
print("Processing mean chronology file...")

chron_df = pd.read_csv(chron_input_csv, sep=";", dtype=str)

# Remove completely empty columns
chron_df = chron_df.dropna(axis=1, how="all")

# Clean column names
chron_df.columns = chron_df.columns.str.strip()

# Remove unnamed index column if present
unnamed_cols = [col for col in chron_df.columns if col.lower().startswith("unnamed") or col == ""]
chron_df = chron_df.drop(columns=unnamed_cols, errors="ignore")

chron_corrected = chron_df.copy()

for col in chron_corrected.columns:
    col_lower = col.lower()

    if col_lower == "year":
        chron_corrected[col] = pd.to_numeric(chron_corrected[col], errors="coerce").astype("Int64")

    elif col_lower in ["n_raw", "n", "n_trees", "samples", "n_samples"]:
        chron_corrected[col] = pd.to_numeric(chron_corrected[col], errors="coerce").astype("Int64")

    else:
        # This corrects O18_raw / mean d18O columns
        chron_corrected[col] = chron_corrected[col].apply(correct_d18o_value)

# Rename columns to clearer names if present
chron_corrected = chron_corrected.rename(
    columns={
        "O18_raw": "Mean_d18O",
        "n_raw": "N_samples"
    }
)

# Save chronology as CSV and Excel
chron_corrected.to_csv(chron_output_csv, index=False, sep=";")

with pd.ExcelWriter(chron_output_xlsx, engine="openpyxl") as writer:
    chron_corrected.to_excel(writer, sheet_name="Mean_chronology", index=False)
    chron_df.to_excel(writer, sheet_name="Original_raw", index=False)

print(f"Saved corrected mean chronology CSV:\n{chron_output_csv}")
print(f"Saved corrected mean chronology Excel:\n{chron_output_xlsx}")

print("Done.")