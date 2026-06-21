"""
Reads all summary_*.txt files from old station summaries and new Baft summaries,
extracts MONTHLY and SEASONAL correlation summary values for all climate parameters,
and saves the output as Excel files.

Handles:
  1) Old station format:
       summary_Kerman.txt
     containing pearson, spearman, kendall in one file.

  2) New Baft format:
       summary_pearson_Baft.txt
       summary_spearmann_Baft.txt
       summary_kendall_Baft.txt
     each containing one correlation method.

Extracted values:
  - analysed_years
  - maximal_calculated_metric
  - lower_ci
  - upper_ci
  - optimal_time_window
  - optimal_time_window_length

Outputs:
  1) One master Excel file with all variables, stations, methods, and approaches.
  2) Optional old-style Excel files, one per parameter.

Author: Mahtab Arjomandi
Updated for Baft_new split summaries
"""

import os
import re
import glob
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_INPUT_DIR = (
    r"E:\FAU master\Master Thesis\Correlation\Correlation outputs"
    r"\Narrow missing removed\d18 climate correlation\All_summary_texts"
)

BAFT_NEW_DIR = (
    r"E:\FAU master\Master Thesis\Correlation\Correlation outputs"
    r"\Narrow missing removed\d18 climate correlation\All_summary_texts\Baft_new"
)

OUTPUT_DIR = (
    r"E:\FAU master\Master Thesis\Correlation\Correlation outputs"
    r"\Narrow missing removed\d18 climate correlation\python outputs\new raw final"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

MASTER_OUTPUT_XLSX = os.path.join(
    OUTPUT_DIR,
    "summary_correlations_all_climate_params.xlsx"
)

# Make old-style one-file-per-parameter outputs too
WRITE_ONE_FILE_PER_PARAM = True

# Correlation methods
CORR_METHODS = ["pearson", "spearman", "kendall"]

# Climate parameters to keep.
# SPEIs are intentionally ignored.
# Include both older variable names and new Baft variable names.
PARAMS_TO_KEEP = [
    "T_Mean",
    "T_Max",
    "T_Min",
    "Precip",
    "RH",
    "VP",
    "VPD",
    "PET",
    "BAL",
    "es",
    "ea",
]

# Case-insensitive canonical parameter names
PARAM_CANONICAL = {p.lower(): p for p in PARAMS_TO_KEEP}

# Row order for old-style parameter sheets
OLD_STYLE_ROW_KEYS = []
for method in CORR_METHODS:
    OLD_STYLE_ROW_KEYS.extend([
        f"analysed_years_{method}",
        f"maximal_calculated_metric_{method}",
        f"lower_ci_{method}",
        f"upper_ci_{method}",
        f"optimal_time_window_{method}",
        f"optimal_time_window_length_{method}",
    ])


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_method(method):
    """
    Normalizes method spelling.

    The Baft file name may say 'spearmann',
    but inside the summary block it is usually 'spearman'.
    """
    method = str(method).strip().lower()

    if method == "spearmann":
        return "spearman"

    return method


def normalize_variable(var):
    """
    Normalizes variable names to the chosen canonical names.
    Returns None if the variable should be ignored.
    """

    var = str(var).strip()

    # Ignore SPEI variables completely
    if re.match(r"^SPEI\d+$", var, flags=re.IGNORECASE):
        return None

    key = var.lower()

    if key in PARAM_CANONICAL:
        return PARAM_CANONICAL[key]

    # Ignore anything not in PARAMS_TO_KEEP
    return None


def city_from_filename(path):
    """
    Extracts city/station name from filenames.

    Examples:
      summary_kerman.txt            -> kerman
      summary_Kerman.txt            -> Kerman
      summary_pearson_Baft.txt      -> Baft
      summary_spearmann_Baft.txt    -> Baft
      summary_kendall_Baft.txt      -> Baft
    """

    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]

    # remove summary_ prefix
    stem = re.sub(r"^summary_", "", stem, flags=re.IGNORECASE)

    parts = stem.split("_")

    # New Baft style: pearson_Baft / spearmann_Baft / kendall_Baft
    if len(parts) >= 2 and normalize_method(parts[0]) in CORR_METHODS:
        city = "_".join(parts[1:])
        return city

    # possible alternative style: Baft_pearson
    if len(parts) >= 2 and normalize_method(parts[-1]) in CORR_METHODS:
        city = "_".join(parts[:-1])
        return city

    return stem


def is_baft_city(city):
    """Detects Baft names, case-insensitive."""
    return str(city).strip().lower() == "baft"


def read_text_file(path):
    """Reads a txt file robustly."""
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        return f.read()


def safe_sheet_name(name):
    """
    Excel sheet names must be <= 31 characters and cannot contain:
    : \\ / ? * [ ]
    """
    name = re.sub(r"[:\\/*?\[\]]", "_", str(name))
    return name[:31]


# ============================================================
# FILE COLLECTION
# ============================================================

def collect_summary_files():
    """
    Collects all summary files.

    Old folder:
      - uses summary_*.txt directly in BASE_INPUT_DIR
      - skips Baft files from this folder

    New Baft folder:
      - uses summary_*.txt from BAFT_NEW_DIR
      - forces city = Baft
    """

    records = []

    # ----------------------------
    # Old station summaries
    # ----------------------------
    old_paths = sorted(
        glob.glob(os.path.join(BASE_INPUT_DIR, "summary_*.txt"))
    )

    for path in old_paths:
        city = city_from_filename(path)

        # Ignore old Baft in the base path
        if is_baft_city(city):
            print(f"Skipping old Baft file from base folder: {path}")
            continue

        records.append({
            "path": path,
            "city": city,
            "source": "old_station"
        })

    # ----------------------------
    # New Baft summaries
    # ----------------------------
    baft_paths = sorted(
        glob.glob(os.path.join(BAFT_NEW_DIR, "summary_*.txt"))
    )

    for path in baft_paths:
        records.append({
            "path": path,
            "city": "Baft",
            "source": "baft_new"
        })

    if not records:
        raise FileNotFoundError(
            "No summary_*.txt files found in:\n"
            f"{BASE_INPUT_DIR}\n"
            f"or:\n{BAFT_NEW_DIR}"
        )

    return records


# ============================================================
# PARSING SUMMARY BLOCKS
# ============================================================

HEADER_RE = re.compile(
    r"=+\s*\n"
    r"\s*(MONTHLY|SEASONAL\s*\(1[–-]8\))\s*:\s*"
    r"([^\|\n]+?)"
    r"\s*\|\s*"
    r"(pearson|spearman|spearmann|kendall)"
    r"\s*\n=+",
    flags=re.IGNORECASE
)


def normalize_approach(raw_approach):
    """Normalizes approach text to MONTHLY or SEASONAL."""
    raw_approach = str(raw_approach).strip().upper()

    if raw_approach.startswith("MONTHLY"):
        return "MONTHLY"

    if raw_approach.startswith("SEASONAL"):
        return "SEASONAL"

    return raw_approach


def line_value(block, key):
    """
    Extracts the value after a key from a summary block.

    Example:
      5   maximal_calculated_metric       -0.366
      10  optimal_time_window             Jan* - Jun

    Returns the rest of the line after the key.
    """

    pattern = re.compile(
        r"\b" + re.escape(key) + r"\b\s+(.+?)\s*$",
        flags=re.IGNORECASE
    )

    for line in block.splitlines():
        m = pattern.search(line)
        if m:
            value = m.group(1).strip()
            return value

    return None


def numeric_line_value(block, key):
    """
    Extracts the first numeric value after a key.
    Returns None if missing.
    """

    value = line_value(block, key)

    if value is None:
        return None

    m = re.search(
        r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
        value
    )

    if not m:
        return None

    return m.group(0)


def parse_summary_block(block):
    """
    Parses all relevant values from one MONTHLY or SEASONAL block.
    """

    if block is None:
        return {
            "analysed_years": None,
            "maximal_calculated_metric": None,
            "lower_ci": None,
            "upper_ci": None,
            "optimal_time_window": None,
            "optimal_time_window_length": None,
            "all_insignificant": None,
        }

    all_insignificant = bool(
        re.search(
            r"All calculations are insignificant",
            block,
            flags=re.IGNORECASE
        )
    )

    if all_insignificant:
        return {
            "analysed_years": line_value(block, "analysed_years"),
            "maximal_calculated_metric": None,
            "lower_ci": None,
            "upper_ci": None,
            "optimal_time_window": None,
            "optimal_time_window_length": None,
            "all_insignificant": True,
        }

    return {
        "analysed_years": line_value(block, "analysed_years"),
        "maximal_calculated_metric": numeric_line_value(block, "maximal_calculated_metric"),
        "lower_ci": numeric_line_value(block, "lower_ci"),
        "upper_ci": numeric_line_value(block, "upper_ci"),
        "optimal_time_window": line_value(block, "optimal_time_window"),
        "optimal_time_window_length": numeric_line_value(block, "optimal_time_window_length"),
        "all_insignificant": False,
    }


def parse_file(path, city, source):
    """
    Parses one summary txt file.

    Returns a list of records:
      city, source, approach, variable, method, values...
    """

    text = read_text_file(path)

    matches = list(HEADER_RE.finditer(text))

    records = []

    if not matches:
        print(f"WARNING: No summary headers found in file: {path}")
        return records

    for i, m in enumerate(matches):
        raw_approach = m.group(1)
        raw_var = m.group(2)
        raw_method = m.group(3)

        approach = normalize_approach(raw_approach)
        variable = normalize_variable(raw_var)
        method = normalize_method(raw_method)

        # Ignore SPEI and any variable not in PARAMS_TO_KEEP
        if variable is None:
            continue

        block_start = m.end()
        block_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[block_start:block_end]

        values = parse_summary_block(block)

        record = {
            "city": city,
            "source": source,
            "file_name": os.path.basename(path),
            "approach": approach,
            "variable": variable,
            "method": method,
        }

        record.update(values)
        records.append(record)

    return records


# ============================================================
# DATAFRAME + EXCEL HELPERS
# ============================================================

def build_long_dataframe(file_records):
    """
    Parses all files and builds one long dataframe.
    """

    all_records = []

    for rec in file_records:
        path = rec["path"]
        city = rec["city"]
        source = rec["source"]

        print(f"Reading: {path}")
        parsed = parse_file(path, city, source)
        all_records.extend(parsed)

    if not all_records:
        raise RuntimeError("No valid records were extracted from the summary files.")

    df = pd.DataFrame(all_records)

    # Convert numeric columns
    numeric_cols = [
        "maximal_calculated_metric",
        "lower_ci",
        "upper_ci",
        "optimal_time_window_length",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Order columns
    ordered_cols = [
        "city",
        "source",
        "file_name",
        "approach",
        "variable",
        "method",
        "analysed_years",
        "maximal_calculated_metric",
        "lower_ci",
        "upper_ci",
        "optimal_time_window",
        "optimal_time_window_length",
        "all_insignificant",
    ]

    df = df[ordered_cols]

    # Apply categorical-like sorting
    method_order = {m: i for i, m in enumerate(CORR_METHODS)}
    approach_order = {"MONTHLY": 0, "SEASONAL": 1}
    var_order = {v: i for i, v in enumerate(PARAMS_TO_KEEP)}

    df["_city_order"] = df["city"].str.lower()
    df["_method_order"] = df["method"].map(method_order)
    df["_approach_order"] = df["approach"].map(approach_order)
    df["_var_order"] = df["variable"].map(var_order)

    df = df.sort_values(
        by=["_city_order", "_var_order", "_approach_order", "_method_order"],
        na_position="last"
    )

    df = df.drop(columns=["_city_order", "_method_order", "_approach_order", "_var_order"])

    return df


def make_pivot(df, approach, value_col):
    """
    Makes a pivot table:
      rows = variable
      columns = city_method
      values = selected value_col
    """

    sub = df[df["approach"] == approach].copy()

    if sub.empty:
        return pd.DataFrame()

    sub["city_method"] = sub["city"].astype(str) + "_" + sub["method"].astype(str)

    pivot = sub.pivot_table(
        index="variable",
        columns="city_method",
        values=value_col,
        aggfunc="first"
    )

    # Reorder variables
    ordered_vars = [v for v in PARAMS_TO_KEEP if v in pivot.index]
    other_vars = [v for v in pivot.index if v not in ordered_vars]
    pivot = pivot.reindex(ordered_vars + other_vars)

    pivot = pivot.reset_index()

    return pivot


def build_old_style_table(df, variable, approach):
    """
    Builds an old-style table for one parameter and one approach.

    Rows:
      analysed_years_pearson
      maximal_calculated_metric_pearson
      lower_ci_pearson
      upper_ci_pearson
      optimal_time_window_pearson
      optimal_time_window_length_pearson
      ...

    Columns:
      first column = variable name
      then one column per city
    """

    sub = df[
        (df["variable"] == variable) &
        (df["approach"] == approach)
    ].copy()

    cities = sorted(sub["city"].unique(), key=lambda x: x.lower())

    out = pd.DataFrame(index=OLD_STYLE_ROW_KEYS, columns=cities)

    for _, row in sub.iterrows():
        city = row["city"]
        method = row["method"]

        out.loc[f"analysed_years_{method}", city] = row["analysed_years"]
        out.loc[f"maximal_calculated_metric_{method}", city] = row["maximal_calculated_metric"]
        out.loc[f"lower_ci_{method}", city] = row["lower_ci"]
        out.loc[f"upper_ci_{method}", city] = row["upper_ci"]
        out.loc[f"optimal_time_window_{method}", city] = row["optimal_time_window"]
        out.loc[f"optimal_time_window_length_{method}", city] = row["optimal_time_window_length"]

    out.insert(0, variable, out.index)

    return out


def write_master_excel(df):
    """
    Writes the master Excel workbook with:
      - long_all
      - monthly/seasonal pivot sheets
    """

    with pd.ExcelWriter(MASTER_OUTPUT_XLSX, engine="openpyxl") as writer:

        df.to_excel(writer, sheet_name="long_all", index=False)

        for approach in ["MONTHLY", "SEASONAL"]:
            for value_col, short_name in [
                ("maximal_calculated_metric", "metric"),
                ("lower_ci", "lower_ci"),
                ("upper_ci", "upper_ci"),
                ("optimal_time_window", "window"),
                ("optimal_time_window_length", "win_length"),
                ("analysed_years", "years"),
            ]:
                pivot = make_pivot(df, approach, value_col)

                sheet_name = safe_sheet_name(
                    f"{approach.lower()}_{short_name}"
                )

                pivot.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\nMaster Excel saved to:\n{MASTER_OUTPUT_XLSX}")


def write_old_style_param_excels(df):
    """
    Writes one Excel file per parameter, similar to the old script,
    but now with two sheets:
      - MONTHLY
      - SEASONAL
    """

    variables = [
        v for v in PARAMS_TO_KEEP
        if v in set(df["variable"].unique())
    ]

    for variable in variables:

        output_path = os.path.join(
            OUTPUT_DIR,
            f"summary_correlations_{variable}.xlsx"
        )

        monthly_table = build_old_style_table(
            df=df,
            variable=variable,
            approach="MONTHLY"
        )

        seasonal_table = build_old_style_table(
            df=df,
            variable=variable,
            approach="SEASONAL"
        )

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            monthly_table.to_excel(writer, sheet_name="MONTHLY", index=False)
            seasonal_table.to_excel(writer, sheet_name="SEASONAL", index=False)

        print(f"Old-style parameter Excel saved: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nCollecting summary files...\n")

    file_records = collect_summary_files()

    print("\nFiles to parse:")
    for rec in file_records:
        print(f"  {rec['city']} | {rec['source']} | {rec['path']}")

    print("\nParsing files...\n")

    df = build_long_dataframe(file_records)

    # Save raw extracted table as CSV too
    csv_path = os.path.join(
        OUTPUT_DIR,
        "summary_correlations_all_climate_params_long.csv"
    )

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"\nLong CSV saved to:\n{csv_path}")

    # Master Excel
    write_master_excel(df)

    # Old-style one file per parameter
    if WRITE_ONE_FILE_PER_PARAM:
        write_old_style_param_excels(df)

    print("\nDone.")
    print(f"All outputs saved in:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()