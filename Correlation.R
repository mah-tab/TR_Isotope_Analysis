############################################################################################################
# mean d18O vs monthly & seasonal climate correlations
# Pearson, Spearman, Kendall
# SHOW plots in R + SAVE plots as PNG (non-empty)
############################################################################################################

install.packages("dendroTools")
install.packages("readxl")

library(dendroTools)
library(ggplot2)
library(readxl)

# -----------------------------
# Paths
# -----------------------------
d18o_path    <- "E:/FAU master/Master Thesis/Correlation/data - next.xlsx"
climate_path <- "E:/FAU master/Master Thesis/Correlation/sirjan_clim.xlsx"

out_dir <- "E:/FAU master/Master Thesis/Correlation/Correlation outputs/Narrow Next Year taken/sirjan_clim"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# -----------------------------
# Read mean d18O data
# -----------------------------
data_crn1 <- read_excel(d18o_path)
colnames(data_crn1)[1:2] <- c("Year", "mean_d18O")

data_crn1 <- as.data.frame(data_crn1)
rownames(data_crn1) <- data_crn1$Year
data_crn1$Year <- NULL

response <- data_crn1[, 1, drop = FALSE]

# -----------------------------
# Read climate data
# -----------------------------
climate_data <- read_excel(climate_path)
climate_data <- as.data.frame(climate_data)

climate_vars <- setdiff(colnames(climate_data), c("Year", "Month"))
cor_methods <- c("pearson", "spearman", "kendall")

# -----------------------------
# Helper: show plot + save plot (robust)
# -----------------------------
save_two_plots <- function(res_obj, prefix, out_dir) {
  
  # ---- show in R (interactive viewer)
  plot(res_obj, type = 1)
  plot(res_obj, type = 2)
  
  # ---- save type 1
  p1 <- plot(res_obj, type = 1)
  png(file.path(out_dir, paste0(prefix, "_type1.png")),
      width = 1600, height = 1000, res = 160)
  if (inherits(p1, "ggplot")) {
    print(p1)
  } else {
    # If plot() already drew something (base), just call again inside device
    plot(res_obj, type = 1)
  }
  dev.off()
  
  # ---- save type 2
  p2 <- plot(res_obj, type = 2)
  png(file.path(out_dir, paste0(prefix, "_type2.png")),
      width = 1600, height = 1000, res = 160)
  if (inherits(p2, "ggplot")) {
    print(p2)
  } else {
    plot(res_obj, type = 2)
  }
  dev.off()
}

# -----------------------------
# Loop over climate variables
# -----------------------------
for (v in climate_vars) {
  
  # Precip -> sum, everything else -> mean
  agg_fun <- ifelse(tolower(v) %in% c("precip", "precipitation", "ppt"),
                    "sum", "mean")
  
  for (cm in cor_methods) {
    
    # Safe tag for filenames
    safe_v <- gsub("[^A-Za-z0-9_]+", "_", v)
    
    # -----------------------------
    # MONTHLY (single months)
    # -----------------------------
    res_monthly <- monthly_response(
      response = response,
      lower_limit = 1, upper_limit = 1,
      env_data = climate_data[, c("Year", "Month", v)],
      fixed_width = 0,
      method = "cor", cor_method = cm,
      row_names_subset = TRUE,
      remove_insignificant = FALSE,
      previous_year = TRUE,
      reference_window = "start",
      alpha = 0.05,
      boot = TRUE, seed = 123,
      tidy_env_data = TRUE, boot_n = 100,
      month_interval = c(-1, 12)
    )
    
    cat("\n==============================\n")
    cat("MONTHLY:", v, "|", cm, "\n")
    cat("==============================\n")
    print(summary(res_monthly))
    
    # show + save plots (FIXED)
    prefix_m <- paste0("MONTHLY_mean_d18O_vs_", safe_v, "_", cm)
    save_two_plots(res_monthly, prefix_m, out_dir)
    
    # save results
    write.csv(res_monthly$calculations,
              file = file.path(out_dir, paste0(prefix_m, "_results.csv")),
              row.names = FALSE)
    
    # -----------------------------
    # SEASONAL (1–8 months)
    # -----------------------------
    res_season <- monthly_response(
      response = response,
      lower_limit = 1, upper_limit = 8,
      env_data = climate_data[, c("Year", "Month", v)],
      fixed_width = 0,
      method = "cor", cor_method = cm,
      row_names_subset = TRUE,
      remove_insignificant = FALSE,
      previous_year = TRUE,
      reference_window = "start",
      alpha = 0.05,
      aggregate_function = agg_fun,
      boot = TRUE, seed = 123,
      tidy_env_data = TRUE, boot_n = 100,
      month_interval = c(-1, 12)
    )
    
    cat("\n==============================\n")
    cat("SEASONAL (1–8):", v, "|", cm, "\n")
    cat("==============================\n")
    print(summary(res_season))
    
    # show + save plots (FIXED)
    prefix_s <- paste0("SEASONAL1to8_mean_d18O_vs_", safe_v, "_", cm)
    save_two_plots(res_season, prefix_s, out_dir)
    
    # save results
    write.csv(res_season$calculations,
              file = file.path(out_dir, paste0(prefix_s, "_results.csv")),
              row.names = FALSE)
  }
}

cat("\nFinished.\nOutputs saved in:\n", out_dir, "\n")

