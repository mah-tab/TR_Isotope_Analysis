"""
Creates one-column Baft heatmaps for climate/PDSI/best-SPEI correlations.

For each proxy:
  - d18O
  - TRW

For each correlation method:
  - pearson
  - spearman
  - kendall

Rows:
  T_Mean, T_Min, T_Max, Precip, RH, VPD, PDSI, Best SPEI

The script:
  1) Reads Baft summary txt files.
  2) Extracts seasonal maximal_calculated_metric and optimal_time_window.
  3) Reads SPEI1-SPEI10 best-correlation CSVs.
  4) Finds the single strongest absolute SPEI correlation for each method.
  5) Plots one heatmap per method and proxy.
  6) Saves PNGs at 600 dpi and also saves CSVs with the plotted values.

Author: Mahtab Arjomandi
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

D18_BAFT_DIR = (
    r"E:\FAU master\Master Thesis\Correlation\Correlation outputs"
    r"\Narrow missing removed\d18 climate correlation\All_summary_texts\Baft_new"
)

TRW_BAFT_DIR = (
    r"E:\FAU master\Master Thesis\Correlation\Correlation outputs"
    r"\Narrow missing removed\TRW climate correlation\All_summary_texts\Baft_new"
)

D18_OUTPUT_DIR = (
    r"E:\FAU master\Master Thesis\Correlation\Correlation outputs"
    r"\Narrow missing removed\d18 climate correlation\python outputs\new raw final"
)

TRW_OUTPUT_DIR = (
    r"E:\FAU master\Master Thesis\Correlation\Correlation outputs"
    r"\Narrow missing removed\TRW climate correlation\python outputs\new raw final"
)

os.makedirs(D18_OUTPUT_DIR, exist_ok=True)
os.makedirs(TRW_OUTPUT_DIR, exist_ok=True)

CORR_METHODS = ["pearson", "spearman", "kendall"]

# Plot the seasonal/1-8 month results, like your station-comparison heatmaps
APPROACH_TO_USE = "SEASONAL"

# Desired y-axis order from BOTTOM to TOP
BASE_PARAMS = ["T_Mean", "T_Min", "T_Max", "Precip", "RH", "VPD"]

# Correlation color scale
COR_SCALE_MIN = -0.75
COR_SCALE_MAX =  0.75

CMAP = "RdBu_r"

# Figure settings
FIG_W = 5.5
FIG_H = 9.0

ANNOT_FONTSIZE = 13
XTICK_FONTSIZE = 13
YTICK_FONTSIZE = 15
TITLE_FONTSIZE = 17
CBAR_FONTSIZE = 13

STATION_LABEL = "Baft(1989-2023)"


# ============================================================
# GENERAL HELPERS
# ============================================================

def normalize_method(method):
    method = str(method).strip().lower()
    if method == "spearmann":
        return "spearman"
    return method


def method_title(method):
    method = normalize_method(method)

    if method == "pearson":
        return "Pearson"
    if method == "spearman":
        return "Spearman"
    if method == "kendall":
        return "Kendall"

    return method


def method_file_aliases(method):
    method = normalize_method(method)

    if method == "spearman":
        return ["spearman", "spearmann"]

    return [method]


def safe_file_text(x):
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(x))


def to_float_or_nan(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def clean_window_text(x):
    if x is None:
        return "NA"

    if isinstance(x, float) and np.isnan(x):
        return "NA"

    x = str(x).strip()

    if x == "" or x.lower() == "nan":
        return "NA"

    return x


def choose_text_color(value):
    if not np.isfinite(value):
        return "black"

    return "white" if abs(value) >= 0.33 else "black"


# ============================================================
# SUMMARY TXT PARSER
# ============================================================

HEADER_RE = re.compile(
    r"=+\s*\n"
    r"\s*(MONTHLY|SEASONAL\s*\(1\s*[–-]\s*8\))\s*:\s*"
    r"([^\|\n]+?)"
    r"\s*\|\s*"
    r"(pearson|spearman|spearmann|kendall)"
    r"\s*\n=+",
    flags=re.IGNORECASE
)


def read_text_file(path):
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        return f.read()


def normalize_approach(raw_approach):
    raw_approach = str(raw_approach).strip().upper()

    if raw_approach.startswith("MONTHLY"):
        return "MONTHLY"

    if raw_approach.startswith("SEASONAL"):
        return "SEASONAL"

    return raw_approach


def normalize_variable_from_header(raw_variable):
    """
    Handles both:
      MONTHLY: T_Mean | spearman
      MONTHLY: mean δ18O vs PDSI | pearson
      MONTHLY: mean TRW vs PDSI | spearman
    """

    raw_variable = str(raw_variable).strip()

    # If the header is "mean δ18O vs PDSI", take the part after "vs"
    if re.search(r"\bvs\b", raw_variable, flags=re.IGNORECASE):
        parts = re.split(r"\bvs\b", raw_variable, flags=re.IGNORECASE)
        raw_variable = parts[-1].strip()

    # Clean spaces
    raw_variable = raw_variable.strip()

    # Ignore SPEI blocks inside summary txts
    if re.match(r"^SPEI\d+$", raw_variable, flags=re.IGNORECASE):
        return None

    allowed = BASE_PARAMS + ["PDSI"]

    allowed_map = {v.lower(): v for v in allowed}

    key = raw_variable.lower()

    if key in allowed_map:
        return allowed_map[key]

    return None


def line_value(block, key):
    """
    Extracts the value after a row label such as:
      maximal_calculated_metric
      lower_ci
      upper_ci
      optimal_time_window
    """

    pattern = re.compile(
        r"\b" + re.escape(key) + r"\b\s+(.+?)\s*$",
        flags=re.IGNORECASE
    )

    for line in block.splitlines():
        m = pattern.search(line)
        if m:
            return m.group(1).strip()

    return None


def numeric_line_value(block, key):
    value = line_value(block, key)

    if value is None:
        return np.nan

    m = re.search(
        r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
        value
    )

    if not m:
        return np.nan

    return to_float_or_nan(m.group(0))


def parse_summary_block(block):
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
            "maximal_calculated_metric": np.nan,
            "lower_ci": np.nan,
            "upper_ci": np.nan,
            "optimal_time_window": "NA",
            "optimal_time_window_length": np.nan,
            "all_insignificant": True,
        }

    return {
        "analysed_years": line_value(block, "analysed_years"),
        "maximal_calculated_metric": numeric_line_value(block, "maximal_calculated_metric"),
        "lower_ci": numeric_line_value(block, "lower_ci"),
        "upper_ci": numeric_line_value(block, "upper_ci"),
        "optimal_time_window": clean_window_text(line_value(block, "optimal_time_window")),
        "optimal_time_window_length": numeric_line_value(block, "optimal_time_window_length"),
        "all_insignificant": False,
    }


def parse_all_summary_txts(input_dir):
    """
    Reads all .txt files in the Baft_new folder and extracts records.

    Returns:
      records[(variable, method, approach)] = parsed values
    """

    txt_files = sorted(glob.glob(os.path.join(input_dir, "*.txt")))

    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in:\n{input_dir}")

    records = {}

    for path in txt_files:

        text = read_text_file(path)
        matches = list(HEADER_RE.finditer(text))

        if not matches:
            print(f"[WARNING] No summary headers found in:\n{path}")
            continue

        for i, m in enumerate(matches):

            raw_approach = m.group(1)
            raw_variable = m.group(2)
            raw_method = m.group(3)

            approach = normalize_approach(raw_approach)
            variable = normalize_variable_from_header(raw_variable)
            method = normalize_method(raw_method)

            if variable is None:
                continue

            block_start = m.end()
            block_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            block = text[block_start:block_end]

            parsed = parse_summary_block(block)
            parsed["source_file"] = os.path.basename(path)

            key = (variable, method, approach)

            # If duplicates exist, keep the later one only if the old value is missing.
            if key not in records:
                records[key] = parsed
            else:
                old_r = records[key].get("maximal_calculated_metric", np.nan)
                new_r = parsed.get("maximal_calculated_metric", np.nan)

                if np.isnan(old_r) and np.isfinite(new_r):
                    records[key] = parsed

    return records


def get_summary_record(records, variable, method, approach=APPROACH_TO_USE):
    """
    Gets the selected summary record.
    Falls back to MONTHLY if SEASONAL is missing.
    """

    method = normalize_method(method)

    key = (variable, method, approach)

    if key in records:
        return records[key]

    fallback_key = (variable, method, "MONTHLY")

    if fallback_key in records:
        return records[fallback_key]

    return {
        "analysed_years": None,
        "maximal_calculated_metric": np.nan,
        "lower_ci": np.nan,
        "upper_ci": np.nan,
        "optimal_time_window": "NA",
        "optimal_time_window_length": np.nan,
        "all_insignificant": None,
        "source_file": None,
    }


# ============================================================
# SPEI BEST CSV PARSER
# ============================================================

def format_spei_label(value):
    """
    Converts SPEI7 or 7 to SPEI07.
    """

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "SPEI_NA"

    txt = str(value).strip()

    m = re.search(r"(\d+)", txt)

    if not m:
        return txt

    num = int(m.group(1))

    return f"SPEI{num:02d}"


def month_label_from_row(row):
    if "Month_label" in row.index and not pd.isna(row["Month_label"]):
        return str(row["Month_label"]).strip().upper()

    if "Month" in row.index and not pd.isna(row["Month"]):
        try:
            m = int(row["Month"])
            labels = [
                "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"
            ]
            if 1 <= m <= 12:
                return labels[m - 1]
        except Exception:
            pass

    if "optimal_time_window" in row.index and not pd.isna(row["optimal_time_window"]):
        return str(row["optimal_time_window"]).strip()

    return "NA"


def find_best_spei_csv(input_dir, response_prefix, method):
    """
    Finds files like:
      mean_d18O_SPEI1to10_best_correlations_spearman.csv
      mean_TRW_SPEI1to10_best_correlations_pearson.csv
    """

    method = normalize_method(method)
    aliases = method_file_aliases(method)

    csv_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))

    candidates = []

    for path in csv_files:

        base = os.path.basename(path).lower()

        if response_prefix.lower() not in base:
            continue

        if "spei1to10_best_correlations" not in base:
            continue

        if any(alias in base for alias in aliases):
            candidates.append(path)

    if not candidates:
        return None

    return candidates[0]


def read_best_spei(input_dir, response_prefix, method):
    """
    Reads the best SPEI CSV and returns:
      correlation, SPEI label, month/window, source file
    """

    path = find_best_spei_csv(
        input_dir=input_dir,
        response_prefix=response_prefix,
        method=method
    )

    if path is None:
        print(f"[WARNING] No best-SPEI CSV found for {response_prefix} | {method}")
        return {
            "row_label": "Best SPEI",
            "variable": "Best SPEI",
            "maximal_calculated_metric": np.nan,
            "optimal_time_window": "NA",
            "spei_label": "SPEI_NA",
            "source_file": None,
        }

    df = pd.read_csv(path)

    if df.empty:
        print(f"[WARNING] Empty best-SPEI CSV:\n{path}")
        return {
            "row_label": "Best SPEI",
            "variable": "Best SPEI",
            "maximal_calculated_metric": np.nan,
            "optimal_time_window": "NA",
            "spei_label": "SPEI_NA",
            "source_file": os.path.basename(path),
        }

    # Prefer the row explicitly marked Strongest_absolute.
    if "Type" in df.columns:
        mask = df["Type"].astype(str).str.lower().str.contains("strongest_absolute")
        if mask.any():
            best = df.loc[mask].iloc[0]
        else:
            best = df.iloc[(df["Correlation"].abs()).idxmax()]
    else:
        best = df.iloc[(df["Correlation"].abs()).idxmax()]

    if "SPEI" in df.columns:
        spei_label = format_spei_label(best["SPEI"])
    elif "SPEI_number" in df.columns:
        spei_label = format_spei_label(best["SPEI_number"])
    else:
        spei_label = "SPEI_NA"

    r_value = to_float_or_nan(best.get("Correlation", np.nan))
    month_window = month_label_from_row(best)

    return {
        "row_label": f"Best SPEI ({spei_label})",
        "variable": "Best SPEI",
        "maximal_calculated_metric": r_value,
        "optimal_time_window": month_window,
        "spei_label": spei_label,
        "source_file": os.path.basename(path),
    }


# ============================================================
# HEATMAP PLOTTING
# ============================================================

def build_plot_rows(records, input_dir, response_prefix, method):
    """
    Builds the rows for one proxy/method heatmap.
    """

    rows = []

    for variable in BASE_PARAMS + ["PDSI"]:

        rec = get_summary_record(
            records=records,
            variable=variable,
            method=method,
            approach=APPROACH_TO_USE
        )

        rows.append({
            "row_label": variable,
            "variable": variable,
            "method": method,
            "maximal_calculated_metric": rec["maximal_calculated_metric"],
            "lower_ci": rec["lower_ci"],
            "upper_ci": rec["upper_ci"],
            "optimal_time_window": rec["optimal_time_window"],
            "optimal_time_window_length": rec["optimal_time_window_length"],
            "source_file": rec["source_file"],
        })

    best_spei = read_best_spei(
        input_dir=input_dir,
        response_prefix=response_prefix,
        method=method
    )

    rows.append({
        "row_label": best_spei["row_label"],
        "variable": best_spei["variable"],
        "method": method,
        "maximal_calculated_metric": best_spei["maximal_calculated_metric"],
        "lower_ci": np.nan,
        "upper_ci": np.nan,
        "optimal_time_window": best_spei["optimal_time_window"],
        "optimal_time_window_length": np.nan,
        "source_file": best_spei["source_file"],
        "spei_label": best_spei["spei_label"],
    })

    return rows


def plot_one_column_heatmap(rows, output_dir, proxy_label, file_proxy, method):
    """
    Saves the one-column Baft heatmap for one proxy and one correlation method.
    """

    df = pd.DataFrame(rows)

    # Save the data used for this plot
    csv_path = os.path.join(
        output_dir,
        f"Baft_heatmap_{file_proxy}_with_PDSI_SPEI_{method}_values.csv"
    )

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Bottom-to-top order was created in rows.
    y_labels_bottom_to_top = list(df["row_label"])
    corr_values_bottom_to_top = df["maximal_calculated_metric"].apply(to_float_or_nan).to_numpy()

    # Flip for matplotlib because first row is shown at the top
    y_labels = list(reversed(y_labels_bottom_to_top))
    corr_values = np.flipud(corr_values_bottom_to_top.reshape(-1, 1))

    # Build annotations
    annotations_bottom_to_top = []

    for _, row in df.iterrows():

        r = to_float_or_nan(row["maximal_calculated_metric"])
        w = clean_window_text(row["optimal_time_window"])

        if np.isnan(r):
            annotations_bottom_to_top.append("NA")
        else:
            annotations_bottom_to_top.append(f"{r:.2f}\n{w}")

    annotations = np.flipud(np.array(annotations_bottom_to_top, dtype=object).reshape(-1, 1))

    plt.figure(figsize=(FIG_W, FIG_H))

    im = plt.imshow(
        corr_values,
        aspect="auto",
        cmap=CMAP,
        vmin=COR_SCALE_MIN,
        vmax=COR_SCALE_MAX
    )

    plt.xticks(
        [0],
        [STATION_LABEL],
        rotation=45,
        ha="right",
        fontsize=XTICK_FONTSIZE,
        fontweight="bold"
    )

    plt.yticks(
        range(len(y_labels)),
        y_labels,
        fontsize=YTICK_FONTSIZE,
        fontweight="bold"
    )

    plt.title(
        f"{method_title(method)}: Baft {proxy_label} with PDSI + best SPEI",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold"
    )

    cbar = plt.colorbar(im, fraction=0.12, pad=0.05)
    cbar.set_label(
        f"{method_title(method)} r",
        fontsize=CBAR_FONTSIZE,
        fontweight="bold"
    )
    cbar.ax.tick_params(labelsize=CBAR_FONTSIZE)

    ax = plt.gca()

    # Gridlines
    ax.set_xticks(np.arange(-0.5, 1.5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_labels), 1), minor=True)

    plt.grid(
        which="minor",
        color="white",
        linestyle="-",
        linewidth=1.5
    )

    plt.tick_params(
        which="minor",
        bottom=False,
        left=False
    )

    # Annotations
    for i in range(len(y_labels)):

        txt = annotations[i, 0]

        if txt == "NA":
            continue

        val = corr_values[i, 0]

        plt.text(
            0,
            i,
            txt,
            ha="center",
            va="center",
            fontsize=ANNOT_FONTSIZE,
            color=choose_text_color(val),
            fontweight="bold"
        )

    plt.tight_layout()

    out_png = os.path.join(
        output_dir,
        f"Baft_heatmap_{file_proxy}_with_PDSI_SPEI_{method}.png"
    )

    plt.savefig(
        out_png,
        dpi=600,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close()

    print(f"Saved heatmap:\n{out_png}")
    print(f"Saved values CSV:\n{csv_path}")


# ============================================================
# RUNNER
# ============================================================

def run_proxy_analysis(input_dir, output_dir, proxy_label, file_proxy, response_prefix):
    """
    Runs all three correlation methods for one proxy.
    """

    os.makedirs(output_dir, exist_ok=True)

    print("\n============================================================")
    print(f"Reading summary txt files for {proxy_label}")
    print("============================================================")
    print(input_dir)

    records = parse_all_summary_txts(input_dir)

    best_spei_summary = []

    for method in CORR_METHODS:

        print("\n------------------------------------------------------------")
        print(f"{proxy_label} | {method_title(method)}")
        print("------------------------------------------------------------")

        rows = build_plot_rows(
            records=records,
            input_dir=input_dir,
            response_prefix=response_prefix,
            method=method
        )

        # Print chosen best SPEI
        best_row = rows[-1]

        print(
            f"Best SPEI for {proxy_label} | {method_title(method)}: "
            f"{best_row.get('spei_label', 'SPEI_NA')} | "
            f"r = {best_row['maximal_calculated_metric']:.3f} | "
            f"month/window = {best_row['optimal_time_window']}"
        )

        best_spei_summary.append({
            "proxy": proxy_label,
            "method": method,
            "best_spei": best_row.get("spei_label", "SPEI_NA"),
            "correlation": best_row["maximal_calculated_metric"],
            "month_or_window": best_row["optimal_time_window"],
            "source_file": best_row["source_file"],
        })

        plot_one_column_heatmap(
            rows=rows,
            output_dir=output_dir,
            proxy_label=proxy_label,
            file_proxy=file_proxy,
            method=method
        )

    best_summary_df = pd.DataFrame(best_spei_summary)

    best_summary_path = os.path.join(
        output_dir,
        f"Baft_heatmap_{file_proxy}_best_SPEI_summary.csv"
    )

    best_summary_df.to_csv(
        best_summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nSaved best SPEI summary:\n{best_summary_path}")


def main():

    run_proxy_analysis(
        input_dir=D18_BAFT_DIR,
        output_dir=D18_OUTPUT_DIR,
        proxy_label="d18O",
        file_proxy="d18o",
        response_prefix="mean_d18O"
    )

    run_proxy_analysis(
        input_dir=TRW_BAFT_DIR,
        output_dir=TRW_OUTPUT_DIR,
        proxy_label="TRW",
        file_proxy="TRW",
        response_prefix="mean_TRW"
    )

    print("\nAll Baft one-column heatmaps finished.")


if __name__ == "__main__":
    main()