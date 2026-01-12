# Author: Mahtab Arjomandi, 06.01.2026
#------------------------------------------------------------
# Prepare δ18O + Climate data for dendro-isotope R analysis
# - Isotope: wide CSV + mean chronology
# - Climate: one CSV per variable (Year, Target)
#------------------------------------------------------------

library(readxl)
library(dplyr)
library(readr)

# -----------------------------
# Paths (Windows-safe)
# -----------------------------
data_dir  <- "E:/FAU master/Master Thesis/Data"
clim_dir  <- "E:/FAU master/Master Thesis/Data/climate data"

# Isotope input + outputs
iso_in_file  <- file.path(data_dir, "d18o_per_sample_sorted_corrected.xlsx")
iso_out_wide <- file.path(data_dir, "d18o_siteA_wide.csv")
iso_out_mean <- file.path(data_dir, "iso_mean_chronology_siteA.csv")

# Climate input
clim_in_file <- file.path(clim_dir, "kerman.xlsx")

# Climate outputs (directory)
clim_out_dir <- file.path(data_dir, "climate_targets_R")
dir.create(clim_out_dir, showWarnings = FALSE, recursive = TRUE)

# ============================================================
# PART 1 — Isotope data preparation
# ============================================================

A.Oxy <- read_excel(iso_in_file)

stopifnot("Year" %in% colnames(A.Oxy))

A.Oxy <- A.Oxy %>%
  mutate(
    Year = as.integer(Year),
    across(-Year, as.numeric)
  )

# Save wide isotope table
write.csv(A.Oxy, iso_out_wide, row.names = FALSE)

# Mean δ18O chronology
sample_cols <- setdiff(colnames(A.Oxy), "Year")

A_mean <- A.Oxy %>%
  mutate(mean_d18O = rowMeans(select(., all_of(sample_cols)), na.rm = TRUE)) %>%
  select(Year, mean_d18O)

write.csv(A_mean, iso_out_mean, row.names = FALSE)

# ============================================================
# PART 2 — Climate data preparation (one file per variable)
# ============================================================

clim_monthly <- read_excel(clim_in_file) %>%
  select(-matches("^Unnamed")) %>%
  mutate(
    Year  = as.integer(Year),
    Month = as.integer(Month),
    across(-c(Year, Month), as.numeric)
  ) %>%
  filter(Year >= 1974, Year <= 2023)

# ---- Helper functions ----
annual_mean <- function(df, var) {
  df %>%
    group_by(Year) %>%
    summarise(Target = mean(.data[[var]], na.rm = TRUE), .groups = "drop")
}

annual_sum <- function(df, var) {
  df %>%
    group_by(Year) %>%
    summarise(Target = sum(.data[[var]], na.rm = TRUE), .groups = "drop")
}

# ---- Variables to export ----
climate_means <- c("T_Mean", "T_Max", "T_Min", "RH", "VPD", "es", "ea")
climate_sums  <- c("Precip")

# ---- Save mean-based variables ----
for (var in climate_means) {
  out <- annual_mean(clim_monthly, var) %>%
    mutate(Target = ifelse(is.nan(Target), NA, Target))
  
  out_file <- file.path(clim_out_dir, paste0("climate_", var, "_annual_mean.csv"))
  write_csv(out, out_file)
}

# ---- Save sum-based variables ----
for (var in climate_sums) {
  out <- annual_sum(clim_monthly, var) %>%
    mutate(Target = ifelse(is.nan(Target), NA, Target))
  
  out_file <- file.path(clim_out_dir, paste0("climate_", var, "_annual_sum.csv"))
  write_csv(out, out_file)
}

# ============================================================
# Confirmation
# ============================================================

cat("Conversion complete.\n\n")

cat("Isotope files written:\n")
cat(" -", iso_out_wide, "\n")
cat(" -", iso_out_mean, "\n\n")

cat("Climate target files written to:\n")
cat(" -", clim_out_dir, "\n")

