# Author Parisa.Z.Foroozan and Mahtab Arjomandi 16.12.2025
############################################################
# Oxygen isotope analyses – Study Site A / Henza
# - Inter-series correlations
# - Mean δ18O chronology
# - dplR statistics (rwl format)
# - Linear trend analysis
# - Climate reconstruction framework
############################################################

library(dplR)
library(utils)
library(dplyr)
library(tidyr)
library(broom)
library(ggplot2)
library(readr)
library(lmtest)

#-----------------------------------------------------------
# Output folder
#-----------------------------------------------------------

out_dir <- "E:/FAU master/Master Thesis/Results/d18o new narrow missing removed/new_raw_final/R_analyse_for_AN_MA"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

#-----------------------------------------------------------
# 1. Read oxygen isotope data
#-----------------------------------------------------------

A.Oxy <- read.csv(
  "E:/FAU master/Master Thesis/Data/d18o Data/new/Henza_O.csv",
  sep = ";",
  header = TRUE,
  stringsAsFactors = FALSE,
  check.names = FALSE
)

# Remove completely empty columns caused by trailing semicolons
A.Oxy <- A.Oxy[, colSums(!is.na(A.Oxy) & A.Oxy != "") > 0]

# Clean column names
names(A.Oxy) <- trimws(names(A.Oxy))

# Check column names
print(names(A.Oxy))

# Define isotope series
columns_to_correlate_A <- c("HNC_24a", "HNC_25a", "HNC_28a", "HNC_53a", "HNC_58b")

# Check that all required columns exist
missing_cols <- setdiff(c("Year", columns_to_correlate_A), names(A.Oxy))

if (length(missing_cols) > 0) {
  stop(
    paste(
      "These required columns are missing from the CSV:",
      paste(missing_cols, collapse = ", ")
    )
  )
}

# Convert Year to numeric
A.Oxy$Year <- as.numeric(A.Oxy$Year)

# Convert isotope columns to numeric
A.Oxy[columns_to_correlate_A] <- lapply(
  A.Oxy[columns_to_correlate_A],
  function(x) as.numeric(as.character(x))
)

# Remove rows without year
A.Oxy <- A.Oxy %>%
  filter(!is.na(Year))

# Optional: remove outliers / problematic years if needed
# Uncomment if you want to exclude HNC_25a in 2016 and 2017
# A.Oxy$HNC_25a[A.Oxy$Year %in% c(2016, 2017)] <- NA

# Select isotope data
A_data <- A.Oxy[, c("Year", columns_to_correlate_A)]

str(A_data)
summary(A.Oxy)

#-----------------------------------------------------------
# 2. Inter-series correlations
#-----------------------------------------------------------

cor_matrix_A <- cor(
  A_data[, -1],
  use = "pairwise.complete.obs"
)

print(cor_matrix_A)

write.csv(
  cor_matrix_A,
  file.path(out_dir, "correlation_matrix_A.csv"),
  row.names = TRUE
)

# Mean inter-series correlation, off-diagonal only
mean_interseries_corr <- function(cor_matrix) {
  upper_vals <- cor_matrix[upper.tri(cor_matrix)]
  mean(upper_vals, na.rm = TRUE)
}

mean_corr_A <- mean_interseries_corr(cor_matrix_A)

cat(
  "Mean inter-series correlation (Site A):",
  round(mean_corr_A, 3),
  "\n"
)

#-----------------------------------------------------------
# 3. Mean δ18O chronology
#-----------------------------------------------------------

A.Oxy$mean_d18O <- rowMeans(
  A.Oxy[, columns_to_correlate_A],
  na.rm = TRUE
)

A_mean <- A.Oxy[, c("Year", "mean_d18O")]

# Plot mean chronology
p_mean_chron <- ggplot(A_mean, aes(x = Year, y = mean_d18O)) +
  geom_line() +
  geom_point() +
  labs(
    title = expression(paste("Mean δ"^18, "O chronology – Site A")),
    x = "Year",
    y = expression(delta^18 * "O (‰)")
  ) +
  theme_minimal()

print(p_mean_chron)

ggsave(
  filename = file.path(out_dir, "mean_d18O_chronology_siteA.png"),
  plot = p_mean_chron,
  width = 10,
  height = 6,
  dpi = 500
)

#-----------------------------------------------------------
# 4. Convert to rwl format and dplR statistics
#-----------------------------------------------------------

# Individual series in rwl format
A.Oxy_raw <- A.Oxy[, c("Year", columns_to_correlate_A)]

A.Oxy_rwl <- data.frame(
  A.Oxy_raw[, -1],
  row.names = A.Oxy_raw$Year
)

rwl_path <- file.path(out_dir, "A.Oxy_rwl.rwl")

write.rwl(A.Oxy_rwl, fname = rwl_path)

A.Oxy_rwl <- read.rwl(rwl_path)

# dplR summaries
rwl_report_A <- rwl.report(A.Oxy_rwl)
print(rwl_report_A)

rwl_stats_A <- rwl.stats(A.Oxy_rwl)
print(rwl_stats_A)

stats_A <- rwi.stats(A.Oxy_rwl)
print(stats_A)

write.csv(
  rwl_stats_A,
  file.path(out_dir, "rwl_stats_A.csv"),
  row.names = TRUE
)

write.csv(
  stats_A,
  file.path(out_dir, "rwi_stats_A.csv"),
  row.names = TRUE
)

# Mean chronology also in rwl-style, single series
A_mean_rwl <- data.frame(
  A = A_mean$mean_d18O,
  row.names = A_mean$Year
)

mean_rwl_stats_A <- rwl.stats(A_mean_rwl)
print(mean_rwl_stats_A)

#-----------------------------------------------------------
# 5. Overall mean δ18O value
#-----------------------------------------------------------

mean_A <- mean(A.Oxy$mean_d18O, na.rm = TRUE)

cat(
  "Overall mean δ18O (Site A):",
  round(mean_A, 2),
  "‰\n"
)

#-----------------------------------------------------------
# 6. Linear trend analysis
#-----------------------------------------------------------

# Long format including all series + mean
A_long <- A.Oxy %>%
  select(Year, all_of(columns_to_correlate_A), mean_d18O) %>%
  pivot_longer(
    cols = -Year,
    names_to = "Series",
    values_to = "d18O"
  )

# Linear model: d18O ~ Year for each series
trend_A <- A_long %>%
  drop_na(d18O) %>%
  group_by(Series) %>%
  do(tidy(lm(d18O ~ Year, data = .))) %>%
  filter(term == "Year") %>%
  select(
    Series,
    slope = estimate,
    std_error = std.error,
    t_value = statistic,
    p_value = p.value
  )

print(trend_A)

write.csv(
  trend_A,
  file.path(out_dir, "d18O_trend_results_siteA.csv"),
  row.names = FALSE
)

# Visualisation of trends
p_trends <- ggplot(A_long, aes(x = Year, y = d18O)) +
  geom_point(alpha = 0.4) +
  geom_smooth(method = "lm", se = FALSE) +
  facet_wrap(~ Series, scales = "free_y") +
  labs(
    title = expression(paste("Linear trends in δ"^18, "O – Site A")),
    x = "Year",
    y = expression(delta^18 * "O (‰)")
  ) +
  theme_minimal()

print(p_trends)

ggsave(
  filename = file.path(out_dir, "linear_trends_d18O_siteA.png"),
  plot = p_trends,
  width = 12,
  height = 8,
  dpi = 500
)

#-----------------------------------------------------------
# 7. Save mean chronology for reconstruction
#-----------------------------------------------------------

write.csv(
  A_mean,
  file.path(out_dir, "iso_mean_chronology_siteA.csv"),
  row.names = FALSE
)

############################################################
# Climate processing and climate–δ18O correlations
# for Study Site A / Henza
############################################################

############################################################
# δ18O-based climate reconstruction – Study Site A
# - Linear transfer function
# - 50/50 split validation
# - Flexible split validation, e.g. 70/30
# - LOOCV
# - k-fold CV
# - Full reconstruction with confidence intervals
############################################################

#-----------------------------------------------------------
# 8. Read chronology and climate target data
#-----------------------------------------------------------

# Use the chronology created above
iso_chron <- read_csv(
  file.path(out_dir, "iso_mean_chronology_siteA.csv"),
  show_col_types = FALSE
)

# Climate target file
# This file must contain columns: Year, Target
climate <- read_csv(
  "E:/FAU master/Master Thesis/Data/climate_targets_R/climate_Precip_annual_sum.csv",
  show_col_types = FALSE
)

# Merge by common years
dat <- inner_join(iso_chron, climate, by = "Year")

#-----------------------------------------------------------
# 9. Define calibration period
#-----------------------------------------------------------

cal_years <- 1974:2023

cal_dat <- dat %>%
  filter(Year %in% cal_years) %>%
  drop_na()

# Stop if there are not enough overlapping years
if (nrow(cal_dat) < 10) {
  stop("Not enough overlapping non-NA years between isotope chronology and climate target.")
}

#-----------------------------------------------------------
# 10. Linear transfer function
#-----------------------------------------------------------

mod <- lm(Target ~ mean_d18O, data = cal_dat)
print(summary(mod))

# Observed vs fitted
cal_dat <- cal_dat %>%
  mutate(Fitted = predict(mod))

p_calibration <- ggplot(cal_dat, aes(Target, Fitted)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(
    title = "Calibration: observed vs fitted",
    x = "Observed climate target",
    y = "Fitted climate target"
  ) +
  theme_minimal()

print(p_calibration)

ggsave(
  filename = file.path(out_dir, "calibration_observed_vs_fitted.png"),
  plot = p_calibration,
  width = 8,
  height = 6,
  dpi = 500
)

#-----------------------------------------------------------
# Helper function
#-----------------------------------------------------------

calc_re_ce <- function(obs, pred, ref_mean) {
  1 - sum((obs - pred)^2, na.rm = TRUE) /
    sum((obs - ref_mean)^2, na.rm = TRUE)
}

#-----------------------------------------------------------
# 11. 50 / 50 split validation
#-----------------------------------------------------------

n <- nrow(cal_dat)
split_50 <- floor(n / 2)

early <- cal_dat[1:split_50, ]
late  <- cal_dat[(split_50 + 1):n, ]

# Early to Late
m_early <- lm(Target ~ mean_d18O, data = early)
pred_late <- predict(m_early, newdata = late)

RE_late <- calc_re_ce(late$Target, pred_late, mean(early$Target, na.rm = TRUE))
CE_late <- calc_re_ce(late$Target, pred_late, mean(late$Target, na.rm = TRUE))

# Late to Early
m_late <- lm(Target ~ mean_d18O, data = late)
pred_early <- predict(m_late, newdata = early)

RE_early <- calc_re_ce(early$Target, pred_early, mean(late$Target, na.rm = TRUE))
CE_early <- calc_re_ce(early$Target, pred_early, mean(early$Target, na.rm = TRUE))

cat("50/50 split:\n")
cat("Early → Late: RE =", round(RE_late, 3), " CE =", round(CE_late, 3), "\n")
cat("Late → Early: RE =", round(RE_early, 3), " CE =", round(CE_early, 3), "\n\n")

#-----------------------------------------------------------
# 12. Flexible split, default 70 / 30
#-----------------------------------------------------------

split_ratio <- 0.7

split_n <- floor(n * split_ratio)

cal_part <- cal_dat[1:split_n, ]
val_part <- cal_dat[(split_n + 1):n, ]

m_split <- lm(Target ~ mean_d18O, data = cal_part)
pred_val <- predict(m_split, newdata = val_part)

RE_split <- calc_re_ce(val_part$Target, pred_val, mean(cal_part$Target, na.rm = TRUE))
CE_split <- calc_re_ce(val_part$Target, pred_val, mean(val_part$Target, na.rm = TRUE))

cat(paste0(round(split_ratio * 100), "/", round((1 - split_ratio) * 100), " split:\n"))
cat("RE =", round(RE_split, 3), " CE =", round(CE_split, 3), "\n\n")

#-----------------------------------------------------------
# 13. Leave-one-out cross-validation, LOOCV
#-----------------------------------------------------------

loocv_pred <- numeric(n)

for (i in seq_len(n)) {
  m <- lm(Target ~ mean_d18O, data = cal_dat[-i, ])
  loocv_pred[i] <- predict(m, newdata = cal_dat[i, ])
}

RE_loocv <- calc_re_ce(cal_dat$Target, loocv_pred, mean(cal_dat$Target, na.rm = TRUE))

cat("LOOCV:\n")
cat("RE =", round(RE_loocv, 3), "\n\n")

#-----------------------------------------------------------
# 14. k-fold cross-validation
#-----------------------------------------------------------

set.seed(123)
k <- 5

folds <- cut(seq_len(n), breaks = k, labels = FALSE)
pred_kfold <- numeric(n)

for (f in seq_len(k)) {
  train_idx <- which(folds != f)
  test_idx  <- which(folds == f)
  
  m <- lm(Target ~ mean_d18O, data = cal_dat[train_idx, ])
  pred_kfold[test_idx] <- predict(m, newdata = cal_dat[test_idx, ])
}

k_cor <- cor(cal_dat$Target, pred_kfold, use = "complete.obs")
k_rmse <- sqrt(mean((cal_dat$Target - pred_kfold)^2, na.rm = TRUE))

cat("k-fold CV, k =", k, ":\n")
cat("Correlation =", round(k_cor, 3), " RMSE =", round(k_rmse, 3), "\n\n")

#-----------------------------------------------------------
# 15. Durbin–Watson test for residual autocorrelation
#-----------------------------------------------------------

dw <- dwtest(mod)

print(dw)

#-----------------------------------------------------------
# 16. Full reconstruction with confidence intervals
#-----------------------------------------------------------

pred_ci <- predict(mod, newdata = iso_chron, interval = "confidence")

recon <- data.frame(
  Year = iso_chron$Year,
  Reconstructed = pred_ci[, "fit"],
  CI_lower = pred_ci[, "lwr"],
  CI_upper = pred_ci[, "upr"]
)

#-----------------------------------------------------------
# 17. Plot full reconstruction
#-----------------------------------------------------------

p_recon <- ggplot(recon, aes(Year, Reconstructed)) +
  geom_line(color = "darkblue", linewidth = 1) +
  geom_ribbon(
    aes(ymin = CI_lower, ymax = CI_upper),
    fill = "steelblue",
    alpha = 0.3
  ) +
  labs(
    title = "δ18O-based climate reconstruction – Site A",
    x = "Year",
    y = "Reconstructed climate variable"
  ) +
  theme_minimal()

print(p_recon)

ggsave(
  filename = file.path(out_dir, "d18O_based_climate_reconstruction_siteA.png"),
  plot = p_recon,
  width = 10,
  height = 6,
  dpi = 500
)

#-----------------------------------------------------------
# 18. Save reconstruction outputs
#-----------------------------------------------------------

target_name <- "Precip_annual_sum"

write_csv(
  recon,
  file.path(out_dir, paste0("Reconstruction_SiteA_d18O_", target_name, ".csv"))
)

validation_summary <- data.frame(
  Method = c(
    "50/50 Early→Late",
    "50/50 Late→Early",
    paste0(round(split_ratio * 100), "/", round((1 - split_ratio) * 100)),
    "LOOCV",
    paste0("k-fold, k=", k)
  ),
  RE = c(RE_late, RE_early, RE_split, RE_loocv, NA),
  CE = c(CE_late, CE_early, CE_split, NA, NA),
  Correlation = c(NA, NA, NA, NA, k_cor),
  RMSE = c(NA, NA, NA, NA, k_rmse)
)

write_csv(
  validation_summary,
  file.path(out_dir, paste0("Validation_Summary_SiteA_d18O_", target_name, ".csv"))
)

cat("All analyses completed successfully.\n")

