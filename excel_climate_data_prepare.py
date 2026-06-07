import os
import glob
import pandas as pd

'''
Sort years, filter and Save climate data into excel files
'''
# ---------------------------------------
# Paths
# ---------------------------------------
in_dir = r"E:\FAU master\Master Thesis\All files and Data\kerman climate- r format"
out_dir = r"E:\FAU master\Master Thesis\Data\climate data"
os.makedirs(out_dir, exist_ok=True)

# Additional SPEI climate file
spei_file = r"E:\FAU master\Master Thesis\Data\d18o Data\new\Baft-clim_with_SPEIS.csv"
spei_out_dir = r"E:\FAU master\Master Thesis\Data\d18o Data\new"
os.makedirs(spei_out_dir, exist_ok=True)

YEAR_MIN, YEAR_MAX = 1974, 2023
columns_order = ["Year", "Month", "T_Mean", "T_Max", "T_Min", "Precip", "RH", "VPD", "es", "ea"]

# ---------------------------------------
# Utilities
# ---------------------------------------
def looks_like_xlsx(file_path: str) -> bool:
    """Return True if file is actually an XLSX (zip) based on magic bytes 'PK'."""
    with open(file_path, "rb") as f:
        sig = f.read(2)
    return sig == b"PK"

def to_numeric_comma_decimal(series: pd.Series) -> pd.Series:
    """Convert strings with comma decimals (e.g. '5,88276') to float; missing stays NaN."""
    s = series.replace({"": pd.NA, "NA": pd.NA, "NaN": pd.NA, "nan": pd.NA, "None": pd.NA})
    s = s.astype("string").str.strip()
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

def read_table_any_format(path: str) -> pd.DataFrame:
    """
    Reads either:
      - real CSV (various encodings/delimiters)
      - Excel mistakenly named .csv (starts with PK)
    """
    if looks_like_xlsx(path):
        df = pd.read_excel(path, sheet_name=0, dtype=str)
        return df

    # Otherwise treat as CSV with encoding fallback and delimiter sniffing
    encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err = None

    for enc in encodings_to_try:
        try:
            df = pd.read_csv(
                path,
                sep=None,               # sniff delimiter (tab/semicolon/comma)
                engine="python",
                dtype=str,
                encoding=enc,
            )
            return df
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Could not read {path}. Last error: {last_err}")

# ---------------------------------------
# Process all files
# ---------------------------------------
paths = sorted(glob.glob(os.path.join(in_dir, "*.csv")))
if not paths:
    raise FileNotFoundError(f"No .csv files found in: {in_dir}")

for p in paths:
    try:
        df = read_table_any_format(p)

        # Clean column names
        df.columns = [str(c).strip() for c in df.columns]

        # Drop any unnamed/empty columns
        df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", na=False)]

        # Ensure required columns exist (create missing as empty)
        for col in columns_order:
            if col not in df.columns:
                df[col] = pd.NA

        # Keep only required columns in order
        df = df[columns_order].copy()

        # Convert Year/Month
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df["Month"] = pd.to_numeric(df["Month"], errors="coerce")

        # Filter years 1974..2023
        df = df[(df["Year"] >= YEAR_MIN) & (df["Year"] <= YEAR_MAX)].copy()

        # Convert measurement columns (handles comma decimals too)
        value_cols = [c for c in columns_order if c not in ("Year", "Month")]
        for c in value_cols:
            df[c] = to_numeric_comma_decimal(df[c])

        # Sort
        df = df.sort_values(["Year", "Month"]).reset_index(drop=True)

        # Save as Excel
        base = os.path.splitext(os.path.basename(p))[0]
        out_path = os.path.join(out_dir, f"{base}.xlsx")
        df.to_excel(out_path, index=False)

        print(f"Saved: {out_path}")

    except Exception as e:
        print(f"[FAILED] {p}\n  Reason: {e}\n")


# ---------------------------------------
# Additionally process SPEI climate file
# ---------------------------------------
try:
    spei_df = read_table_any_format(spei_file)

    # Clean column names
    spei_df.columns = [str(c).strip() for c in spei_df.columns]

    # Drop any unnamed/empty columns, e.g. row index column from R
    spei_df = spei_df.loc[:, ~spei_df.columns.str.contains(r"^Unnamed", na=False)]
    spei_df = spei_df.loc[:, spei_df.columns != ""]

    # Convert Year/Month
    spei_df["Year"] = pd.to_numeric(spei_df["Year"], errors="coerce")
    spei_df["Month"] = pd.to_numeric(spei_df["Month"], errors="coerce")

    # Filter years 1974..2023
    spei_df = spei_df[
        (spei_df["Year"] >= YEAR_MIN) &
        (spei_df["Year"] <= YEAR_MAX)
    ].copy()

    # Convert all columns except Year/Month to numeric
    spei_value_cols = [c for c in spei_df.columns if c not in ("Year", "Month")]

    for c in spei_value_cols:
        spei_df[c] = to_numeric_comma_decimal(spei_df[c])

    # Sort
    spei_df = spei_df.sort_values(["Year", "Month"]).reset_index(drop=True)

    # Save as Excel in the same folder as the SPEI CSV
    spei_base = os.path.splitext(os.path.basename(spei_file))[0]
    spei_out_path = os.path.join(spei_out_dir, f"{spei_base}.xlsx")

    spei_df.to_excel(spei_out_path, index=False)

    print(f"Saved SPEI climate file: {spei_out_path}")

except Exception as e:
    print(f"[FAILED] {spei_file}\n  Reason: {e}\n")