############################################################################################################
# mean d18O vs monthly & seasonal climate correlations
# Spearman only
# SAVE plots as PNG with white background
############################################################################################################

# install.packages("dendroTools")
# install.packages("readxl")
# install.packages("ggplot2")
# install.packages("scales")

library(dendroTools)
library(ggplot2)
library(readxl)
library(scales)

# -----------------------------
# Paths
# -----------------------------
d18o_path    <- "E:/FAU master/Master Thesis/Data/d18o Data/new/Henza_mean_chron_final.xlsx"
climate_path <- "E:/FAU master/Master Thesis/Data/d18o Data/new/Baft-clim_with_SPEIS.xlsx"

out_dir <- "E:/FAU master/Master Thesis/Results/d18o Baft correlation/New_raw_final/spearmann"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# -----------------------------
# Analysis settings
# -----------------------------
START_YEAR <- 1989
END_YEAR   <- 2023

# Fixed color and y-axis scale for correlation plots
COR_SCALE_MIN <- -0.75
COR_SCALE_MAX <-  0.75
COR_SCALE_BREAKS <- c(-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75)

# Font sizes for all plots
TITLE_FONTSIZE <- 20
SUBTITLE_FONTSIZE <- 16
AXIS_TITLE_FONTSIZE <- 16
AXIS_TICK_FONTSIZE <- 14
LEGEND_TITLE_FONTSIZE <- 14
LEGEND_TEXT_FONTSIZE <- 13

# Spearman only for now
cor_methods <- c("spearman")

# Later, use one of these instead:
# cor_methods <- c("pearson")
# cor_methods <- c("kendall")

# Climate variables to use
# SPEI11 onwards are intentionally ignored
climate_vars <- c(
  "T_Mean",
  "T_Max",
  "T_Min",
  "Precip",
  "RH",
  "VP",
  "VPD",
  "PET",
  "BAL",
  "SPEI1",
  "SPEI2",
  "SPEI3",
  "SPEI4",
  "SPEI5",
  "SPEI6",
  "SPEI7",
  "SPEI8",
  "SPEI9",
  "SPEI10"
)

month_labels <- c(
  "Jan*", "Feb*", "Mar*", "Apr*", "May*", "Jun*", "Jul*", "Aug*", "Sep*", "Oct*", "Nov*", "Dec*",
  "Jan",  "Feb",  "Mar",  "Apr",  "May",  "Jun",  "Jul",  "Aug",  "Sep",  "Oct",  "Nov",  "Dec"
)

# -----------------------------
# Read mean d18O data
# -----------------------------
data_crn1 <- read_excel(d18o_path)
colnames(data_crn1)[1:2] <- c("Year", "mean_d18O")

data_crn1 <- as.data.frame(data_crn1)

data_crn1$Year <- as.numeric(data_crn1$Year)
data_crn1$mean_d18O <- as.numeric(data_crn1$mean_d18O)

# Keep only common analysis period
data_crn1 <- data_crn1[
  data_crn1$Year >= START_YEAR &
    data_crn1$Year <= END_YEAR,
]

# Remove missing response years because dendroTools does not allow NA in response
removed_response_years <- data_crn1$Year[is.na(data_crn1$mean_d18O)]

if (length(removed_response_years) > 0) {
  cat("\nRemoved years with missing mean_d18O from response:\n")
  print(removed_response_years)
  
  write.csv(
    data.frame(removed_years_missing_mean_d18O = removed_response_years),
    file = file.path(out_dir, "removed_years_missing_mean_d18O.csv"),
    row.names = FALSE
  )
}

data_crn1 <- data_crn1[!is.na(data_crn1$mean_d18O), ]

rownames(data_crn1) <- data_crn1$Year
data_crn1$Year <- NULL

response <- data_crn1[, 1, drop = FALSE]

cat("\nResponse years used:\n")
print(rownames(response))
cat("\nNumber of response years used:", nrow(response), "\n")

write.csv(
  data.frame(
    Year = as.numeric(rownames(response)),
    mean_d18O = response[, 1]
  ),
  file = file.path(out_dir, "mean_d18O_response_used.csv"),
  row.names = FALSE
)

# -----------------------------
# Read climate data
# -----------------------------
climate_data <- read_excel(climate_path)
climate_data <- as.data.frame(climate_data)

colnames(climate_data) <- trimws(colnames(climate_data))

climate_data$Year <- as.numeric(climate_data$Year)
climate_data$Month <- as.numeric(climate_data$Month)

climate_data <- climate_data[
  climate_data$Year >= START_YEAR &
    climate_data$Year <= END_YEAR,
]

missing_climate_cols <- setdiff(c("Year", "Month", climate_vars), colnames(climate_data))

if (length(missing_climate_cols) > 0) {
  stop(
    paste(
      "These required climate columns are missing:",
      paste(missing_climate_cols, collapse = ", ")
    )
  )
}

climate_data <- climate_data[, c("Year", "Month", climate_vars)]

for (v in climate_vars) {
  climate_data[[v]] <- as.numeric(climate_data[[v]])
}

climate_data <- climate_data[order(climate_data$Year, climate_data$Month), ]

cat("\nClimate period used:\n")
cat(min(climate_data$Year, na.rm = TRUE), "to", max(climate_data$Year, na.rm = TRUE), "\n")

cat("\nClimate variables used:\n")
print(climate_vars)

write.csv(
  climate_data,
  file = file.path(out_dir, "Baft_climate_used_until_SPEI10.csv"),
  row.names = FALSE
)

# -----------------------------
# Helper: safe filenames
# -----------------------------
safe_filename <- function(x) {
  gsub("[^A-Za-z0-9_]+", "_", x)
}

# -----------------------------
# Helper: convert dendroTools calculation matrix to long table
# -----------------------------
calc_to_long <- function(calc_matrix, approach, variable, cor_method) {
  
  calc_df <- as.data.frame(calc_matrix)
  calc_df[] <- lapply(calc_df, function(x) as.numeric(as.character(x)))
  
  n_rows <- nrow(calc_df)
  n_cols <- ncol(calc_df)
  
  out <- data.frame()
  
  for (i in seq_len(n_rows)) {
    for (j in seq_len(n_cols)) {
      
      r_value <- calc_df[i, j]
      
      start_label <- if (j <= length(month_labels)) {
        month_labels[j]
      } else {
        as.character(j)
      }
      
      out <- rbind(
        out,
        data.frame(
          Approach = approach,
          Variable = variable,
          Method = cor_method,
          Window_length = i,
          Start_index = j,
          Start_month = start_label,
          Correlation = r_value
        )
      )
    }
  }
  
  out <- out[!is.na(out$Correlation), ]
  
  return(out)
}

# -----------------------------
# Helper: build climate window series
# -----------------------------
get_window_climate_series <- function(
    climate_data,
    response_years,
    variable,
    start_index,
    window_length,
    aggregate_function
) {
  
  start_month <- ((start_index - 1) %% 12) + 1
  year_offset <- ifelse(start_index <= 12, -1, 0)
  
  values <- rep(NA_real_, length(response_years))
  
  climate_tmp <- climate_data
  climate_tmp$global_month <- climate_tmp$Year * 12 + climate_tmp$Month
  
  for (k in seq_along(response_years)) {
    
    y <- response_years[k]
    
    start_global <- (y + year_offset) * 12 + start_month
    needed_global <- start_global:(start_global + window_length - 1)
    
    selected <- climate_tmp[climate_tmp$global_month %in% needed_global, variable]
    
    if (length(selected) == window_length && all(!is.na(selected))) {
      
      if (aggregate_function == "sum") {
        values[k] <- sum(selected, na.rm = TRUE)
      } else {
        values[k] <- mean(selected, na.rm = TRUE)
      }
    }
  }
  
  return(values)
}

# -----------------------------
# Helper: calculate p-values for every dendroTools window
# -----------------------------
calculate_pvalues_for_windows <- function(
    res_obj,
    climate_data,
    response,
    variable,
    cor_method,
    aggregate_function,
    approach
) {
  
  calc_df <- as.data.frame(res_obj$calculations)
  calc_df[] <- lapply(calc_df, function(x) as.numeric(as.character(x)))
  
  response_years <- as.numeric(rownames(response))
  response_values <- as.numeric(response[, 1])
  
  n_rows <- nrow(calc_df)
  n_cols <- ncol(calc_df)
  
  out <- data.frame()
  
  for (i in seq_len(n_rows)) {
    for (j in seq_len(n_cols)) {
      
      r_value <- calc_df[i, j]
      
      if (is.na(r_value)) {
        next
      }
      
      climate_window <- get_window_climate_series(
        climate_data = climate_data,
        response_years = response_years,
        variable = variable,
        start_index = j,
        window_length = i,
        aggregate_function = aggregate_function
      )
      
      valid <- complete.cases(response_values, climate_window)
      n_valid <- sum(valid)
      
      if (n_valid >= 4) {
        
        test <- suppressWarnings(
          cor.test(
            response_values[valid],
            climate_window[valid],
            method = cor_method,
            exact = FALSE
          )
        )
        
        p_value <- test$p.value
        
      } else {
        p_value <- NA_real_
      }
      
      start_label <- if (j <= length(month_labels)) {
        month_labels[j]
      } else {
        as.character(j)
      }
      
      out <- rbind(
        out,
        data.frame(
          Approach = approach,
          Variable = variable,
          Method = cor_method,
          Window_length = i,
          Start_index = j,
          Start_month = start_label,
          Correlation = r_value,
          p_value = p_value,
          n_used = n_valid,
          Significant_p05 = !is.na(p_value) & p_value < 0.05,
          Significant_p01 = !is.na(p_value) & p_value < 0.01
        )
      )
    }
  }
  
  return(out)
}

# -----------------------------
# Helper: approximate critical correlation coefficient
# for two-sided p-value threshold
# -----------------------------
critical_r <- function(n, alpha) {
  
  if (is.na(n) || n <= 3) {
    return(NA_real_)
  }
  
  tcrit <- qt(1 - alpha / 2, df = n - 2)
  rcrit <- sqrt(tcrit^2 / (tcrit^2 + n - 2))
  
  return(rcrit)
}

# -----------------------------
# Helper: custom heatmap
# negative = blue, zero = white, positive = red
# fixed legend from -0.75 to 0.75
# -----------------------------
save_custom_heatmap <- function(res_obj, prefix, out_dir, approach, variable, cor_method) {
  
  plot_df <- calc_to_long(
    calc_matrix = res_obj$calculations,
    approach = approach,
    variable = variable,
    cor_method = cor_method
  )
  
  if (nrow(plot_df) == 0) {
    cat("\nNo correlation values available for heatmap:", prefix, "\n")
    return(NULL)
  }
  
  plot_df$Start_month <- factor(plot_df$Start_month, levels = month_labels)
  plot_df$Window_length <- factor(
    plot_df$Window_length,
    levels = sort(unique(plot_df$Window_length))
  )
  
  p <- ggplot(
    plot_df,
    aes(
      x = Start_month,
      y = Window_length,
      fill = Correlation
    )
  ) +
    geom_tile(color = "white", linewidth = 0.4) +
    scale_fill_gradient2(
      low = "#2166AC",
      mid = "white",
      high = "#B2182B",
      midpoint = 0,
      limits = c(COR_SCALE_MIN, COR_SCALE_MAX),
      breaks = COR_SCALE_BREAKS,
      oob = scales::squish,
      name = "Correlation\ncoefficient"
    ) +
    labs(
      title = paste0(approach, ": mean δ18O vs ", variable, " (", cor_method, ")"),
      subtitle = paste0("Analysed period: ", START_YEAR, "-", END_YEAR),
      x = "Starting month of calculation, including previous year",
      y = "Number of consecutive months"
    ) +
    theme_minimal(base_size = 14) +
    theme(
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      legend.background = element_rect(fill = "white", color = NA),
      panel.grid = element_blank(),
      axis.text.x = element_text(
        angle = 90,
        vjust = 0.5,
        hjust = 1,
        size = AXIS_TICK_FONTSIZE
      ),
      axis.text.y = element_text(
        size = AXIS_TICK_FONTSIZE
      ),
      axis.title.x = element_text(
        size = AXIS_TITLE_FONTSIZE,
        face = "bold"
      ),
      axis.title.y = element_text(
        size = AXIS_TITLE_FONTSIZE,
        face = "bold"
      ),
      plot.title = element_text(
        size = TITLE_FONTSIZE,
        face = "bold"
      ),
      plot.subtitle = element_text(
        size = SUBTITLE_FONTSIZE
      ),
      legend.title = element_text(
        size = LEGEND_TITLE_FONTSIZE,
        face = "bold"
      ),
      legend.text = element_text(
        size = LEGEND_TEXT_FONTSIZE
      )
    )
  
  ggsave(
    filename = file.path(out_dir, paste0(prefix, "_type1.png")),
    plot = p,
    width = 12,
    height = 7,
    dpi = 500,
    bg = "white"
  )
  
  return(plot_df)
}

# -----------------------------
# Helper: custom type-2 bar plot
# y-axis = correlation coefficient
# dashed blue/red lines = correlation thresholds for p < 0.05 / p < 0.01
# y-axis fixed from -0.75 to 0.75
# -----------------------------
save_custom_type2_barplot <- function(pvalue_df, prefix, out_dir, approach, variable, cor_method) {
  
  if (nrow(pvalue_df) == 0) {
    cat("\nNo data available for bar plot:", prefix, "\n")
    return(NULL)
  }
  
  # For each starting month, choose the strongest absolute correlation.
  # For monthly, this is just the single-month value.
  bar_df <- do.call(
    rbind,
    lapply(
      split(pvalue_df, pvalue_df$Start_month),
      function(x) {
        x[which.max(abs(x$Correlation)), ]
      }
    )
  )
  
  bar_df$Start_month <- factor(bar_df$Start_month, levels = month_labels)
  bar_df <- bar_df[order(bar_df$Start_month), ]
  
  # Use median n_used for threshold lines.
  # In most cases this is constant across windows.
  n_for_threshold <- median(bar_df$n_used, na.rm = TRUE)
  
  rcrit_005 <- critical_r(n_for_threshold, 0.05)
  rcrit_001 <- critical_r(n_for_threshold, 0.01)
  
  p <- ggplot(
    bar_df,
    aes(
      x = Start_month,
      y = Correlation,
      fill = Correlation
    )
  ) +
    geom_col(width = 0.8) +
    scale_fill_gradient2(
      low = "#2166AC",
      mid = "white",
      high = "#B2182B",
      midpoint = 0,
      limits = c(COR_SCALE_MIN, COR_SCALE_MAX),
      breaks = COR_SCALE_BREAKS,
      oob = scales::squish,
      name = "Correlation\ncoefficient"
    ) +
    geom_hline(
      yintercept = 0,
      color = "black",
      linewidth = 0.4
    ) +
    geom_hline(
      yintercept = c(rcrit_005, -rcrit_005),
      color = "blue",
      linetype = "dashed",
      linewidth = 1
    ) +
    geom_hline(
      yintercept = c(rcrit_001, -rcrit_001),
      color = "red",
      linetype = "dashed",
      linewidth = 1
    ) +
    labs(
      title = paste0(approach, ": mean δ18O vs ", variable, " (", cor_method, ")"),
      subtitle = paste0(
        "Bars = correlation coefficient; blue dashed = p < 0.05, red dashed = p < 0.01; n ≈ ",
        n_for_threshold
      ),
      x = "Starting month of calculation, including previous year",
      y = "Correlation coefficient"
    ) +
    scale_y_continuous(
      limits = c(COR_SCALE_MIN, COR_SCALE_MAX),
      breaks = COR_SCALE_BREAKS
    ) +
    theme_minimal(base_size = 14) +
    theme(
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      legend.background = element_rect(fill = "white", color = NA),
      panel.grid.major.x = element_blank(),
      axis.text.x = element_text(
        angle = 90,
        vjust = 0.5,
        hjust = 1,
        size = AXIS_TICK_FONTSIZE
      ),
      axis.text.y = element_text(
        size = AXIS_TICK_FONTSIZE
      ),
      axis.title.x = element_text(
        size = AXIS_TITLE_FONTSIZE,
        face = "bold"
      ),
      axis.title.y = element_text(
        size = AXIS_TITLE_FONTSIZE,
        face = "bold"
      ),
      plot.title = element_text(
        size = TITLE_FONTSIZE,
        face = "bold"
      ),
      plot.subtitle = element_text(
        size = SUBTITLE_FONTSIZE
      ),
      legend.title = element_text(
        size = LEGEND_TITLE_FONTSIZE,
        face = "bold"
      ),
      legend.text = element_text(
        size = LEGEND_TEXT_FONTSIZE
      )
    )
  
  ggsave(
    filename = file.path(out_dir, paste0(prefix, "_type2.png")),
    plot = p,
    width = 12,
    height = 7,
    dpi = 500,
    bg = "white"
  )
  
  return(bar_df)
}

# -----------------------------
# Store summary tables across all variables
# -----------------------------
all_window_results <- data.frame()
all_significant_results <- data.frame()
best_window_summary <- data.frame()

# -----------------------------
# Loop over climate variables
# -----------------------------
for (v in climate_vars) {
  
  # Precip -> sum, everything else -> mean
  agg_fun <- ifelse(
    tolower(v) %in% c("precip", "precipitation", "ppt"),
    "sum",
    "mean"
  )
  
  for (cm in cor_methods) {
    
    safe_v <- safe_filename(v)
    
    # -----------------------------
    # MONTHLY (single months)
    # -----------------------------
    res_monthly <- monthly_response(
      response = response,
      lower_limit = 1,
      upper_limit = 1,
      env_data = climate_data[, c("Year", "Month", v)],
      fixed_width = 0,
      method = "cor",
      cor_method = cm,
      row_names_subset = TRUE,
      remove_insignificant = FALSE,
      previous_year = TRUE,
      reference_window = "start",
      alpha = 0.05,
      boot = TRUE,
      seed = 123,
      tidy_env_data = TRUE,
      boot_n = 100,
      month_interval = c(-1, 12)
    )
    
    cat("\n==============================\n")
    cat("MONTHLY:", v, "|", cm, "\n")
    cat("==============================\n")
    print(summary(res_monthly))
    
    prefix_m <- paste0("MONTHLY_mean_d18O_vs_", safe_v, "_", cm)
    
    # Save corrected heatmap as type1
    save_custom_heatmap(
      res_obj = res_monthly,
      prefix = prefix_m,
      out_dir = out_dir,
      approach = "MONTHLY",
      variable = v,
      cor_method = cm
    )
    
    # Save raw dendroTools calculation matrix
    write.csv(
      res_monthly$calculations,
      file = file.path(out_dir, paste0(prefix_m, "_results.csv")),
      row.names = FALSE
    )
    
    # Calculate p-values for thresholds and significance table
    pvals_monthly <- calculate_pvalues_for_windows(
      res_obj = res_monthly,
      climate_data = climate_data,
      response = response,
      variable = v,
      cor_method = cm,
      aggregate_function = agg_fun,
      approach = "MONTHLY"
    )
    
    write.csv(
      pvals_monthly,
      file = file.path(out_dir, paste0(prefix_m, "_pvalues_all_windows.csv")),
      row.names = FALSE
    )
    
    # Save correlation bar plot with p<0.05 and p<0.01 threshold lines
    save_custom_type2_barplot(
      pvalue_df = pvals_monthly,
      prefix = prefix_m,
      out_dir = out_dir,
      approach = "MONTHLY",
      variable = v,
      cor_method = cm
    )
    
    all_window_results <- rbind(all_window_results, pvals_monthly)
    all_significant_results <- rbind(
      all_significant_results,
      pvals_monthly[pvals_monthly$Significant_p05, ]
    )
    
    best_monthly <- pvals_monthly[which.min(pvals_monthly$p_value), ]
    
    if (nrow(best_monthly) > 0) {
      best_window_summary <- rbind(best_window_summary, best_monthly)
    }
    
    # -----------------------------
    # SEASONAL (1–8 months)
    # -----------------------------
    res_season <- monthly_response(
      response = response,
      lower_limit = 1,
      upper_limit = 8,
      env_data = climate_data[, c("Year", "Month", v)],
      fixed_width = 0,
      method = "cor",
      cor_method = cm,
      row_names_subset = TRUE,
      remove_insignificant = FALSE,
      previous_year = TRUE,
      reference_window = "start",
      alpha = 0.05,
      aggregate_function = agg_fun,
      boot = TRUE,
      seed = 123,
      tidy_env_data = TRUE,
      boot_n = 100,
      month_interval = c(-1, 12)
    )
    
    cat("\n==============================\n")
    cat("SEASONAL (1–8):", v, "|", cm, "\n")
    cat("==============================\n")
    print(summary(res_season))
    
    prefix_s <- paste0("SEASONAL1to8_mean_d18O_vs_", safe_v, "_", cm)
    
    # Save corrected heatmap as type1
    save_custom_heatmap(
      res_obj = res_season,
      prefix = prefix_s,
      out_dir = out_dir,
      approach = "SEASONAL 1-8",
      variable = v,
      cor_method = cm
    )
    
    # Save raw dendroTools calculation matrix
    write.csv(
      res_season$calculations,
      file = file.path(out_dir, paste0(prefix_s, "_results.csv")),
      row.names = FALSE
    )
    
    # Calculate p-values for thresholds and significance table
    pvals_season <- calculate_pvalues_for_windows(
      res_obj = res_season,
      climate_data = climate_data,
      response = response,
      variable = v,
      cor_method = cm,
      aggregate_function = agg_fun,
      approach = "SEASONAL_1to8"
    )
    
    write.csv(
      pvals_season,
      file = file.path(out_dir, paste0(prefix_s, "_pvalues_all_windows.csv")),
      row.names = FALSE
    )
    
    # Save correlation bar plot with p<0.05 and p<0.01 threshold lines
    save_custom_type2_barplot(
      pvalue_df = pvals_season,
      prefix = prefix_s,
      out_dir = out_dir,
      approach = "SEASONAL 1-8",
      variable = v,
      cor_method = cm
    )
    
    all_window_results <- rbind(all_window_results, pvals_season)
    all_significant_results <- rbind(
      all_significant_results,
      pvals_season[pvals_season$Significant_p05, ]
    )
    
    best_season <- pvals_season[which.min(pvals_season$p_value), ]
    
    if (nrow(best_season) > 0) {
      best_window_summary <- rbind(best_window_summary, best_season)
    }
  }
}

# -----------------------------
# Save combined outputs
# -----------------------------
write.csv(
  all_window_results,
  file = file.path(out_dir, "ALL_correlation_windows_with_pvalues.csv"),
  row.names = FALSE
)

write.csv(
  all_significant_results,
  file = file.path(out_dir, "ALL_significant_results_p_lt_0.05.csv"),
  row.names = FALSE
)

write.csv(
  all_significant_results[all_significant_results$Significant_p01, ],
  file = file.path(out_dir, "ALL_significant_results_p_lt_0.01.csv"),
  row.names = FALSE
)

write.csv(
  best_window_summary,
  file = file.path(out_dir, "ALL_best_window_per_variable_and_approach.csv"),
  row.names = FALSE
)

cat("\nFinished.\nOutputs saved in:\n", out_dir, "\n")

cat("\nMain summary files:\n")
cat("1) ALL_correlation_windows_with_pvalues.csv\n")
cat("2) ALL_significant_results_p_lt_0.05.csv\n")
cat("3) ALL_significant_results_p_lt_0.01.csv\n")
cat("4) ALL_best_window_per_variable_and_approach.csv\n")
