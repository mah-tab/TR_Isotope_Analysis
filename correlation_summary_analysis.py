"""
Reads all summary_*.txt files in a folder, extracts correlation params for a given parameter, e.g (SEASONAL (1–8): T_Mean)
for pearson/spearman/kendall:
  - maximal_calculated_metric
  - optimal_time_window

Writes them to an Excel table:
Row0: Parameter, <city1>, <city2>, ...
Col0: maximal_calculated_metric_pearson, optimal_time_window_pearson, ...
Cells: extracted values (or NA if missing / insignificant)

Author: Mahtab Arjomandi
Date: 2026-03-01
"""

import os
import re
import glob
import pandas as pd


# ----------------------------
# CONFIG
# ----------------------------
INPUT_DIR = r"E:\FAU master\Master Thesis\Correlation\Correlation outputs\Narrow Next Year taken\All_summary_texts"  # <-- change to your folder
OUTPUT_DIR = r"E:\FAU master\Master Thesis\Correlation\Correlation outputs\Narrow Next Year taken\\"

PARAM_NAME_IN_FILE = "T_Mean"     # exactly as in your summaries
PARAM_NAME_IN_TABLE = "T_Mean"    # how you want it shown in the table header
# # Select parameter from :
#     T_Mean
#     T_Max
#     T_Min
#     Precip
#     RH
#     VPD
#     es
#     ea

CORR_METHODS = ["pearson", "spearman", "kendall"]

# make sure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_XLSX = os.path.join(
    OUTPUT_DIR,
    f"summary_correlations_{PARAM_NAME_IN_TABLE}.xlsx"
)

# output row labels (first column)
ROW_KEYS = [
    "maximal_calculated_metric_pearson",
    "optimal_time_window_pearson",
    "maximal_calculated_metric_spearman",
    "optimal_time_window_spearman",
    "maximal_calculated_metric_kendall",
    "optimal_time_window_kendall",
]


# ----------------------------
# PARSING HELPERS
# ----------------------------
def city_from_filename(path: str) -> str:
    """
    Extracts city name from filenames like:
      summary_Kerman.txt -> Kerman
      summary_baft.txt   -> baft
    """
    base = os.path.basename(path)
    m = re.match(r"summary_(.+?)\.txt$", base, flags=re.IGNORECASE)
    return m.group(1) if m else os.path.splitext(base)[0]


def extract_seasonal_block(text: str, var_name: str, method: str) -> str | None:
    """
    Returns the substring block for:
      ==============================
      SEASONAL (1–8): <var> | <method>
      ==============================
      ...
    until the next "====" header or end of file.

    If not found, returns None.
    """
    # Be tolerant: hyphen might be '-' or '–'
    # and there might be different spacing.
    pattern = (
        r"=+\s*\n"
        r"SEASONAL\s*\(1[–-]8\)\s*:\s*"
        + re.escape(var_name) +
        r"\s*\|\s*" + re.escape(method) +
        r"\s*\n=+\s*\n"
    )

    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return None

    start = m.end()

    # next header begins with "====" and then a label (MONTHLY/SEASONAL)
    nxt = re.search(r"\n=+\s*\n(?:MONTHLY|SEASONAL)", text[start:], flags=re.IGNORECASE)
    end = start + nxt.start() if nxt else len(text)

    return text[start:end]


def parse_metric_and_window(block: str) -> tuple[str | None, str | None]:
    """
    From a seasonal block, extracts:
      maximal_calculated_metric  <value>
      optimal_time_window        <value>

    If the block contains "All calculations are insignificant", returns (None, None).
    """
    if block is None:
        return None, None

    if "All calculations are insignificant" in block:
        return None, None

    # Example lines:
    # 5   maximal_calculated_metric                                                           -0.323
    # 10        optimal_time_window                                                      Jan* - Feb*
    metric = None
    window = None

    m1 = re.search(r"maximal_calculated_metric\s+([-+]?\d*\.?\d+)", block)
    if m1:
        metric = m1.group(1)

    m2 = re.search(r"optimal_time_window\s+(.+)", block)
    if m2:
        # take rest of the line, strip whitespace
        window = m2.group(1).strip()

    return metric, window


def set_value(table_dict: dict, city: str, row_key: str, value):
    """Helper to set table cell; keep None as NA."""
    if city not in table_dict:
        table_dict[city] = {rk: None for rk in ROW_KEYS}
    table_dict[city][row_key] = value


# ----------------------------
# MAIN
# ----------------------------
def main():
    paths = sorted(glob.glob(os.path.join(INPUT_DIR, "summary_*.txt")))
    if not paths:
        raise FileNotFoundError(f"No files found like summary_*.txt in: {INPUT_DIR}")

    # table_dict[city][row_key] = value
    table_dict = {}

    for p in paths:
        city = city_from_filename(p)

        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        for method in CORR_METHODS:
            block = extract_seasonal_block(text, PARAM_NAME_IN_FILE, method)
            metric, window = parse_metric_and_window(block)

            set_value(table_dict, city, f"maximal_calculated_metric_{method}", metric)
            set_value(table_dict, city, f"optimal_time_window_{method}", window)

    # Build DataFrame in the exact layout you asked:
    # First column is the row keys, columns are cities, first row is parameter label.
    cities = sorted(table_dict.keys())

    df = pd.DataFrame(index=ROW_KEYS, columns=cities)
    for city in cities:
        for rk in ROW_KEYS:
            df.loc[rk, city] = table_dict[city].get(rk, None)

    # Put the parameter name in an extra "Parameter" column above the row keys,
    # by writing a first column that contains the row keys and then a top header row.
    out = df.copy()
    out.insert(0, PARAM_NAME_IN_TABLE, out.index)  # first column: row labels

    # Write to Excel
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="SEASONAL_Tmean", index=False)

    print(f"Done.\nSaved Excel to:\n{OUTPUT_XLSX}")


if __name__ == "__main__":
    main()