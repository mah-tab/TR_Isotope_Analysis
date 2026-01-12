#Author Parisa.Z.Foroozan, 16.12.2025
############################################################
# Oxygen isotope analyses – as an example: Study Site A
# - Inter-series correlations
# - Mean δ18O chronology
# - dplR statistics (rwl format)
# - Linear trend analysis
############################################################

library(dplR)
library(utils)
library(dplyr)
library(tidyr)
library(broom)
library(ggplot2)

out_dir <- "E:/FAU master/Master Thesis/R outputs"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
A.Oxy <- read.csv("E:/FAU master/Master Thesis/Data/d18o_siteA_wide.csv")


# please import your data sheet, includes year and Oxygen series
#-----------------------------------------------------------
# 1. Select isotope series for Site A
#-----------------------------------------------------------

columns_to_correlate_A <- c("HNC_24a", "HNC_25a", "HNC_28a", "HNC_53a", "HNC_58b")

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
write.csv(cor_matrix_A, file.path(out_dir, "correlation_matrix_A.csv"))

# Mean inter-series correlation (off-diagonal)
mean_interseries_corr <- function(cor_matrix) {
  upper_vals <- cor_matrix[upper.tri(cor_matrix)]
  mean(upper_vals, na.rm = TRUE)
}

mean_corr_A <- mean_interseries_corr(cor_matrix_A)
cat("Mean inter-series correlation (Site A):",
    round(mean_corr_A, 3), "\n")

#-----------------------------------------------------------
# 3. Mean δ18O chronology
#-----------------------------------------------------------

A.Oxy$mean_d18O <- rowMeans(
  A.Oxy[, columns_to_correlate_A],
  na.rm = TRUE
)

A_mean <- A.Oxy[, c("Year", "mean_d18O")]

# Plot mean chronology
ggplot(A_mean, aes(x = Year, y = mean_d18O)) +
  geom_line() +
  geom_point() +
  labs(
    title = expression(paste("Mean δ"^18, "O chronology – Site A")),
    y = expression(delta^18 * "O (‰)")
  ) +
  theme_minimal()

#-----------------------------------------------------------
# 4. Convert to rwl format and dplR statistics
#-----------------------------------------------------------

# Individual series in rwl format
A.Oxy_raw <- A.Oxy[, c("Year", columns_to_correlate_A)]

A.Oxy_rwl <- data.frame(
  row.names = A.Oxy_raw$Year,
  A.Oxy_raw[, -1]
)

rwl_path <- file.path(out_dir, "A.Oxy_rwl.rwl")
write.rwl(A.Oxy_rwl, fname = rwl_path)

A.Oxy_rwl <- read.rwl(rwl_path)

# dplR summaries
rwl.report(A.Oxy_rwl)
rwl.stats(A.Oxy_rwl)

stats_A <- rwi.stats(A.Oxy_rwl)
print(stats_A)

# Mean chronology also in rwl-style (single series)
A_mean_rwl <- data.frame(
  row.names = A_mean$Year,
  A = A_mean$mean_d18O
)

rwl.stats(A_mean_rwl)
#-----------------------------------------------------------
# 5. Overall mean δ18O value
#-----------------------------------------------------------

mean_A <- mean(A.Oxy$mean_d18O, na.rm = TRUE)
cat("Overall mean δ18O (Site A):",
    round(mean_A, 2), "‰\n")

#-----------------------------------------------------------
# 6. Linear trend analysis (what you said you want to show)
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
write.csv(trend_A, file.path(out_dir, "d18O_trend_results_siteA.csv"))

# Visualisation of trends
ggplot(A_long, aes(x = Year, y = d18O)) +
  geom_point(alpha = 0.4) +
  geom_smooth(method = "lm", se = FALSE) +
  facet_wrap(~ Series, scales = "free_y") +
  labs(
    title = expression(paste("Linear trends in δ"^18, "O – Site A")),
    y = expression(delta^18 * "O (‰)")
  ) +
  theme_minimal()
#-----------------------------------------------------------
# 7. Save mean chronology for reconstruction
#-----------------------------------------------------------

write.csv(
  A_mean,
  file.path(out_dir, "iso_mean_chronology_siteA.csv"),
  row.names = FALSE
)

############################################################
# please perform Climate processing and climate–δ18O correlations
# for your Study Site 
############################################################
### After identifying the climate signal using monthly correlation analysis, we select one physically meaningful target variable:
## chronology can be mean_d18O or TRW
#I assume: 
#iso_mean_chronology_siteA.csv
#→ columns: Year, mean_d18O

#target series already prepared (after correlation analysis):
# → example: VPD Mar_Sep

############################################################
# δ18O-based climate reconstruction – Study Site A
# - Linear transfer function
# - 50/50 split validation
# - Flexible split validation (e.g. 70/30)
# - LOOCV
# - k-fold CV
# - Full reconstruction with confidence intervals
############################################################

library(readr)
library(dplyr)
library(ggplot2)
library(lmtest)
#-----------------------------------------------------------
# 1. Read data
#-----------------------------------------------------------

out_dir <- "E:/FAU master/Master Thesis/R outputs"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# δ18O mean chronology (prepared in Data folder)
iso_chron <- read_csv("E:/FAU master/Master Thesis/Data/iso_mean_chronology_siteA.csv")

# Climate target file (example: annual precipitation sum) ###### adjust for other variables
climate <- read_csv("E:/FAU master/Master Thesis/Data/climate_targets_R/climate_Precip_annual_sum.csv")
# climate must contain: Year, Target

# Merge by common years
dat <- inner_join(iso_chron, climate, by = "Year")

#-----------------------------------------------------------
# 2. Define calibration period
#-----------------------------------------------------------

cal_years <- 1974:2023   # adapt if needed

cal_dat <- dat %>%
  filter(Year %in% cal_years) %>%
  drop_na()

#-----------------------------------------------------------
# 3. Linear transfer function
#-----------------------------------------------------------

mod <- lm(Target ~ mean_d18O, data = cal_dat)
summary(mod)

# Observed vs fitted
cal_dat <- cal_dat %>%
  mutate(Fitted = predict(mod))

ggplot(cal_dat, aes(Target, Fitted)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(title = "Calibration: observed vs fitted") +
  theme_minimal()

#-----------------------------------------------------------
# Helper functions
#-----------------------------------------------------------

calc_re_ce <- function(obs, pred, ref_mean) {
  1 - sum((obs - pred)^2) / sum((obs - ref_mean)^2)
}

#-----------------------------------------------------------
# 4. 50 / 50 split validation
#-----------------------------------------------------------

n <- nrow(cal_dat)
split_50 <- floor(n / 2)

early <- cal_dat[1:split_50, ]
late  <- cal_dat[(split_50 + 1):n, ]

# Early → Late
m_early <- lm(Target ~ mean_d18O, data = early)
pred_late <- predict(m_early, newdata = late)

RE_late <- calc_re_ce(late$Target, pred_late, mean(early$Target))
CE_late <- calc_re_ce(late$Target, pred_late, mean(late$Target))

# Late → Early
m_late <- lm(Target ~ mean_d18O, data = late)
pred_early <- predict(m_late, newdata = early)

RE_early <- calc_re_ce(early$Target, pred_early, mean(late$Target))
CE_early <- calc_re_ce(early$Target, pred_early, mean(early$Target))

cat("50/50 split:\n")
cat("Early → Late: RE =", round(RE_late, 3), " CE =", round(CE_late, 3), "\n")
cat("Late → Early: RE =", round(RE_early, 3), " CE =", round(CE_early, 3), "\n\n")

#-----------------------------------------------------------
# 5. Flexible split (default 70 / 30)
#-----------------------------------------------------------

split_ratio <- 0.7   # students can change this

split_n <- floor(n * split_ratio)

cal_part <- cal_dat[1:split_n, ]
val_part <- cal_dat[(split_n + 1):n, ]

m_split <- lm(Target ~ mean_d18O, data = cal_part)
pred_val <- predict(m_split, newdata = val_part)

RE_split <- calc_re_ce(val_part$Target, pred_val, mean(cal_part$Target))
CE_split <- calc_re_ce(val_part$Target, pred_val, mean(val_part$Target))

cat(paste0(round(split_ratio*100), "/", round((1-split_ratio)*100), " split:\n"))
cat("RE =", round(RE_split, 3), " CE =", round(CE_split, 3), "\n\n")

#-----------------------------------------------------------
# 6. Leave-one-out cross-validation (LOOCV)
#-----------------------------------------------------------

loocv_pred <- numeric(n)

for (i in seq_len(n)) {
  m <- lm(Target ~ mean_d18O, data = cal_dat[-i, ])
  loocv_pred[i] <- predict(m, newdata = cal_dat[i, ])
}

RE_loocv <- calc_re_ce(cal_dat$Target, loocv_pred, mean(cal_dat$Target))

cat("LOOCV:\n")
cat("RE =", round(RE_loocv, 3), "\n\n")

#-----------------------------------------------------------
# 7. k-fold cross-validation
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

k_cor  <- cor(cal_dat$Target, pred_kfold)
k_rmse <- sqrt(mean((cal_dat$Target - pred_kfold)^2))

cat("k-fold CV (k =", k, "):\n")
cat("Correlation =", round(k_cor, 3),
    " RMSE =", round(k_rmse, 3), "\n\n")

#-----------------------------------------------------------
# Durbin–Watson test for residual autocorrelation
#-----------------------------------------------------------
# The Durbin–Watson test checks whether the residuals of your regression are autocorrelated, 
# which is a key assumption in linear regression
# It tests whether regression residuals are independent in time (no first-order autocorrelation).
#The test returns a DW statistic between 0 and 4:
# DW value	Interpretation
# ≈ 2	No autocorrelation (ideal) 
# < 2	Positive autocorrelation
# > 2	Negative autocorrelation

# And a p-value:

# p > 0.05 → no significant autocorrelation

# p ≤ 0.05 → residuals are autocorrelated (assumption violated)
#--------------------------------------------------------------
dw <- dwtest(mod)

print(dw)

#-----------------------------------------------------------
# 8. Full reconstruction with confidence intervals
#-----------------------------------------------------------

pred_ci <- predict(mod, newdata = iso_chron, interval = "confidence")

recon <- data.frame(
  Year = iso_chron$Year,
  Reconstructed = pred_ci[, "fit"],
  CI_lower = pred_ci[, "lwr"],
  CI_upper = pred_ci[, "upr"]
)

#-----------------------------------------------------------
# 9. Plot full reconstruction
#-----------------------------------------------------------

ggplot(recon, aes(Year, Reconstructed)) +
  geom_line(color = "darkblue", linewidth = 1) +
  geom_ribbon(aes(ymin = CI_lower, ymax = CI_upper),
              fill = "steelblue", alpha = 0.3) +
  labs(
    title = "δ18O-based climate reconstruction – Site A",
    y = "Reconstructed climate variable"
  ) +
  theme_minimal()

#-----------------------------------------------------------
# 10. Save outputs
#-----------------------------------------------------------

target_name <- "Precip_annual_sum"  # change this when you switch climate files

write_csv(recon, file.path(out_dir, paste0("Reconstruction_SiteA_d18O_", target_name, ".csv")))

validation_summary <- data.frame(
  Method = c("50/50 Early→Late",
             "50/50 Late→Early",
             paste0(round(split_ratio*100), "/", round((1-split_ratio)*100)),
             "LOOCV",
             paste0("k-fold (k=", k, ")")),
  
  RE = c(RE_late, RE_early, RE_split, RE_loocv, NA),
  CE = c(CE_late, CE_early, CE_split, NA, NA),
  Correlation = c(NA, NA, NA, NA, k_cor),
  RMSE = c(NA, NA, NA, NA, k_rmse)
)

write_csv(validation_summary, file.path(out_dir, paste0("Validation_Summary_SiteA_d18O_", target_name, ".csv")))

