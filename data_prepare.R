#Author: Mahtab Arjomandi, 06.01.2026
#------------------------------------------------------------
# Prepare δ18O Excel file for dendro-isotope R analysis
# ------------------------------------------------------------

# ------------------------------------------------------------
# Prepare δ18O Excel file for dendro-isotope R analysis
# (input and outputs in the same directory)
# ------------------------------------------------------------

library(readxl)
library(dplyr)

# -----------------------------
# Paths (Windows-safe)
# -----------------------------
data_dir <- "E:/FAU master/Master Thesis/Data"

in_file  <- file.path(data_dir, "d18o_per_sample_sorted_corrected.xlsx")
out_iso  <- file.path(data_dir, "d18o_siteA_wide.csv")
out_mean <- file.path(data_dir, "iso_mean_chronology_siteA.csv")

# -----------------------------
# Read Excel
# -----------------------------
A.Oxy <- read_excel(in_file)

# -----------------------------
# Basic checks
# -----------------------------
stopifnot("Year" %in% colnames(A.Oxy))

# Ensure correct types
A.Oxy <- A.Oxy %>%
  mutate(
    Year = as.integer(Year),
    across(-Year, as.numeric)
  )

# -----------------------------
# Save wide isotope table
# (for correlations, trends, rwl analysis)
# -----------------------------
write.csv(A.Oxy, out_iso, row.names = FALSE)

# -----------------------------
# Create mean δ18O chronology
# (for climate reconstruction)
# -----------------------------
sample_cols <- setdiff(colnames(A.Oxy), "Year")

A_mean <- A.Oxy %>%
  mutate(
    mean_d18O = rowMeans(select(., all_of(sample_cols)), na.rm = TRUE)
  ) %>%
  select(Year, mean_d18O)

write.csv(A_mean, out_mean, row.names = FALSE)

# -----------------------------
# Confirmation
# -----------------------------
cat("Conversion complete.\n")
cat("Files written:\n")
cat(" -", out_iso, "\n")
cat(" -", out_mean, "\n")

