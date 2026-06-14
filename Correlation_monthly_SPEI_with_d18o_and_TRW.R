############################################################################################################
# SPEI1-SPEI10 monthly correlation heatmaps
# mean d18O vs SPEI1-SPEI10
# mean TRW  vs SPEI1-SPEI10
# Spearman only
# SAVE plots and CSVs with white background
############################################################################################################

# install.packages("readxl")
# install.packages("ggplot2")
# install.packages("scales")

library(readxl)
library(ggplot2)
library(scales)

# -----------------------------
# Paths
# -----------------------------
d18o_path <- "E:/FAU master/Master Thesis/Data/d18o Data/new/Henza_mean_chron_final.xlsx"
trw_path  <- "E:/FAU master/Master Thesis/Data/Tree Ring Width Chronology/TRW_chronology_mean_alone.xlsx"

climate_path <- "E:/FAU master/Master Thesis/Data/d18o Data/new/Baft-clim_with_SPEIS.xlsx"

out_dir_d18o <- "E:/FAU master/Master Thesis/Results/d18o Baft correlation/SPEI_monthly_heatmap"
out_dir_trw  <- "E:/FAU master/Master Thesis/Results/TRW Baft correlation/SPEI_monthly_heatmap"

dir.create(out_dir_d18o, showWarnings = FALSE, recursive = TRUE)
dir.create(out_dir_trw, showWarnings = FALSE, recursive = TRUE)

# -----------------------------
# Analysis settings
# -----------------------------
START_YEAR <- 1989
END_YEAR   <- 2023

cor_method <- "spearman"

SPEI_vars <- paste0("SPEI", 1:10)

month_labels <- c(
  "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
  "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"
)

# Fixed color scale
COR_SCALE_MIN <- -0.75
COR_SCALE_MAX <-  0.75
COR_SCALE_BREAKS <- c(-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75)

# Font sizes
TITLE_FONTSIZE <- 20
SUBTITLE_FONTSIZE <- 16
AXIS_TITLE_FONTSIZE <- 16
AXIS_TICK_FONTSIZE <- 14
LEGEND_TITLE_FONTSIZE <- 14
LEGEND_TEXT_FONTSIZE <- 13

# -----------------------------
# Read climate/SPEI data
# -----------------------------
climate_data <- read_excel(climate_path)
climate_data <- as.data.frame(climate_data)

colnames(climate_data) <- trimws(colnames(climate_data))

climate_data$Year <- as.numeric(climate_data$Year)
climate_data$Month <- as.numeric(climate_data$Month)

missing_cols <- setdiff(c("Year", "Month", SPEI_vars), colnames(climate_data))

if (length(missing_cols) > 0) {
  stop(
    paste(
      "These required columns are missing from climate file:",
      paste(missing_cols, collapse = ", ")
    )
  )
}

climate_data <- climate_data[
  climate_data$Year >= START_YEAR &
    climate_data$Year <= END_YEAR,
  c("Year", "Month", SPEI_vars)
]

for (v in SPEI_vars) {
  climate_data[[v]] <- as.numeric(climate_data[[v]])
}

climate_data <- climate_data[order(climate_data$Year, climate_data$Month), ]

# -----------------------------
# Helper: read response chronology
# -----------------------------
read_response <- function(response_path, response_name) {
  
  response_data <- read_excel(response_path)
  response_data <- as.data.frame(response_data)
  
  colnames(response_data)[1:2] <- c("Year", response_name)
  
  response_data$Year <- as.numeric(response_data$Year)
  response_data[[response_name]] <- as.numeric(response_data[[response_name]])
  
  response_data <- response_data[
    response_data$Year >= START_YEAR &
      response_data$Year <= END_YEAR,
  ]
  
  removed_years <- response_data$Year[is.na(response_data[[response_name]])]
  
  response_data <- response_data[!is.na(response_data[[response_name]]), ]
  
  return(
    list(
      data = response_data,
      removed_years = removed_years
    )
  )
}

# -----------------------------
# Helper: calculate SPEI month x timescale correlations
# -----------------------------
calculate_spei_correlations <- function(response_data, response_name) {
  
  out <- data.frame()
  
  for (spei in SPEI_vars) {
    
    for (m in 1:12) {
      
      climate_sub <- climate_data[
        climate_data$Month == m,
        c("Year", spei)
      ]
      
      colnames(climate_sub) <- c("Year", "SPEI_value")
      
      merged <- merge(
        response_data[, c("Year", response_name)],
        climate_sub,
        by = "Year"
      )
      
      merged <- merged[complete.cases(merged), ]
      
      n_used <- nrow(merged)
      
      if (n_used >= 4) {
        
        test <- suppressWarnings(
          cor.test(
            merged[[response_name]],
            merged$SPEI_value,
            method = cor_method,
            exact = FALSE
          )
        )
        
        r_value <- as.numeric(test$estimate)
        p_value <- test$p.value
        
      } else {
        
        r_value <- NA_real_
        p_value <- NA_real_
      }
      
      out <- rbind(
        out,
        data.frame(
          Response = response_name,
          SPEI = spei,
          SPEI_number = as.numeric(gsub("SPEI", "", spei)),
          Month = m,
          Month_label = month_labels[m],
          Correlation = r_value,
          p_value = p_value,
          n_used = n_used,
          Significant_p05 = !is.na(p_value) & p_value < 0.05,
          Significant_p01 = !is.na(p_value) & p_value < 0.01
        )
      )
    }
  }
  
  return(out)
}

# -----------------------------
# Helper: plot heatmap
# -----------------------------
save_spei_heatmap <- function(cor_df, response_label, out_dir, file_prefix) {
  
  plot_df <- cor_df
  
  plot_df$Month_label <- factor(plot_df$Month_label, levels = month_labels)
  plot_df$SPEI <- factor(plot_df$SPEI, levels = rev(SPEI_vars))
  
  p <- ggplot(
    plot_df,
    aes(
      x = Month_label,
      y = SPEI,
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
      title = paste0(response_label, " vs SPEI1-SPEI10"),
      subtitle = paste0("Spearman correlation, ", START_YEAR, "-", END_YEAR),
      x = "Month",
      y = "Standardized Precipitation-Evapotranspiration Index (SPEI)"
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
        size = AXIS_TICK_FONTSIZE,
        face = "bold"
      ),
      axis.text.y = element_text(
        size = AXIS_TICK_FONTSIZE,
        face = "bold"
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
    filename = file.path(out_dir, paste0(file_prefix, "_SPEI1to10_monthly_heatmap.png")),
    plot = p,
    width = 10,
    height = 7,
    dpi = 500,
    bg = "white"
  )
}

# -----------------------------
# Helper: plot absolute correlation heatmap
# useful for finding strongest correlation regardless of sign
# -----------------------------
save_spei_abs_heatmap <- function(cor_df, response_label, out_dir, file_prefix) {
  
  plot_df <- cor_df
  plot_df$Abs_correlation <- abs(plot_df$Correlation)
  
  plot_df$Month_label <- factor(plot_df$Month_label, levels = month_labels)
  plot_df$SPEI <- factor(plot_df$SPEI, levels = rev(SPEI_vars))
  
  p <- ggplot(
    plot_df,
    aes(
      x = Month_label,
      y = SPEI,
      fill = Abs_correlation
    )
  ) +
    geom_tile(color = "white", linewidth = 0.4) +
    scale_fill_gradient(
      low = "white",
      high = "#B2182B",
      limits = c(0, COR_SCALE_MAX),
      breaks = c(0, 0.25, 0.5, 0.75),
      oob = scales::squish,
      name = "|Correlation|"
    ) +
    labs(
      title = paste0(response_label, " vs SPEI1-SPEI10"),
      subtitle = paste0("Absolute Spearman correlation, ", START_YEAR, "-", END_YEAR),
      x = "Month",
      y = "Standardized Precipitation-Evapotranspiration Index (SPEI)"
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
        size = AXIS_TICK_FONTSIZE,
        face = "bold"
      ),
      axis.text.y = element_text(
        size = AXIS_TICK_FONTSIZE,
        face = "bold"
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
    filename = file.path(out_dir, paste0(file_prefix, "_SPEI1to10_absolute_correlation_heatmap.png")),
    plot = p,
    width = 10,
    height = 7,
    dpi = 500,
    bg = "white"
  )
}

# -----------------------------
# Helper: run full SPEI heatmap analysis
# -----------------------------
run_spei_heatmap_analysis <- function(response_path, response_name, response_label, out_dir, file_prefix) {
  
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  
  response_list <- read_response(
    response_path = response_path,
    response_name = response_name
  )
  
  response_data <- response_list$data
  
  if (length(response_list$removed_years) > 0) {
    write.csv(
      data.frame(Removed_years_missing_response = response_list$removed_years),
      file = file.path(out_dir, paste0(file_prefix, "_removed_years_missing_response.csv")),
      row.names = FALSE
    )
  }
  
  write.csv(
    response_data,
    file = file.path(out_dir, paste0(file_prefix, "_response_used.csv")),
    row.names = FALSE
  )
  
  cor_df <- calculate_spei_correlations(
    response_data = response_data,
    response_name = response_name
  )
  
  write.csv(
    cor_df,
    file = file.path(out_dir, paste0(file_prefix, "_SPEI1to10_monthly_correlations.csv")),
    row.names = FALSE
  )
  
  significant_p05 <- cor_df[cor_df$Significant_p05, ]
  significant_p01 <- cor_df[cor_df$Significant_p01, ]
  
  write.csv(
    significant_p05,
    file = file.path(out_dir, paste0(file_prefix, "_SPEI1to10_significant_p_lt_0.05.csv")),
    row.names = FALSE
  )
  
  write.csv(
    significant_p01,
    file = file.path(out_dir, paste0(file_prefix, "_SPEI1to10_significant_p_lt_0.01.csv")),
    row.names = FALSE
  )
  
  best_positive <- cor_df[which.max(cor_df$Correlation), ]
  best_negative <- cor_df[which.min(cor_df$Correlation), ]
  best_absolute <- cor_df[which.max(abs(cor_df$Correlation)), ]
  
  best_summary <- rbind(
    data.frame(Type = "Strongest_positive", best_positive),
    data.frame(Type = "Strongest_negative", best_negative),
    data.frame(Type = "Strongest_absolute", best_absolute)
  )
  
  write.csv(
    best_summary,
    file = file.path(out_dir, paste0(file_prefix, "_SPEI1to10_best_correlations.csv")),
    row.names = FALSE
  )
  
  save_spei_heatmap(
    cor_df = cor_df,
    response_label = response_label,
    out_dir = out_dir,
    file_prefix = file_prefix
  )
  
  save_spei_abs_heatmap(
    cor_df = cor_df,
    response_label = response_label,
    out_dir = out_dir,
    file_prefix = file_prefix
  )
  
  cat("\nFinished SPEI monthly heatmap analysis for:", response_label, "\n")
  cat("Outputs saved in:\n", out_dir, "\n")
}

# -----------------------------
# Run d18O vs SPEI1-SPEI10
# -----------------------------
run_spei_heatmap_analysis(
  response_path = d18o_path,
  response_name = "mean_d18O",
  response_label = "mean δ18O",
  out_dir = out_dir_d18o,
  file_prefix = "mean_d18O"
)

# -----------------------------
# Run TRW vs SPEI1-SPEI10
# -----------------------------
run_spei_heatmap_analysis(
  response_path = trw_path,
  response_name = "mean_TRW",
  response_label = "mean TRW",
  out_dir = out_dir_trw,
  file_prefix = "mean_TRW"
)

cat("\nAll SPEI1-SPEI10 monthly heatmap analyses finished.\n")

