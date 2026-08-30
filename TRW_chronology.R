################################################################################
# Plot standardized tree-ring width (TRW) chronology
#
# Input Excel structure:
# Year | std | samp.depth
#
# The script:
# 1. Reads the already standardized TRW chronology from Excel
# 2. Uses the "std" column as the standardized tree-ring width index
# 3. Plots the annual TRW chronology
# 4. Adds a 10-year smoothing spline to show lower-frequency variation
# 5. Saves the figure as a high-resolution PNG (600 dpi)
################################################################################


# -----------------------------
# Install required packages
# Run these lines only once if the packages are not already installed
# -----------------------------
# install.packages("readxl")
# install.packages("dplR")


# -----------------------------
# Load libraries
# -----------------------------
library(readxl)
library(dplR)


# -----------------------------
# Input and output paths
# -----------------------------
input_file <- "E:/FAU master/Master Thesis/Data/Tree Ring Width Chronology/50years-ads.xlsx"

output_dir <- "E:/FAU master/Master Thesis/Plots/TRW"

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

output_file <- file.path(
  output_dir,
  "TRW_chronology.png"
)


# -----------------------------
# Read standardized TRW chronology
# Expected columns:
# Year | std | samp.depth
# -----------------------------
data.crn <- read_excel(input_file)


# -----------------------------
# Check column names and data
# -----------------------------
print(colnames(data.crn))
print(head(data.crn))
print(tail(data.crn))


# -----------------------------
# Convert columns to numeric
# -----------------------------
data.crn$Year <- as.numeric(data.crn$Year)
data.crn$std <- as.numeric(data.crn$std)
data.crn$samp.depth <- as.numeric(data.crn$samp.depth)


# -----------------------------
# Remove rows with missing Year or std values
# -----------------------------
data.crn <- data.crn[
  !is.na(data.crn$Year) &
    !is.na(data.crn$std),
]


# -----------------------------
# Extract years and standardized TRW chronology
# -----------------------------
yrs <- data.crn$Year
TRWstd <- data.crn$std


# -----------------------------
# Calculate 10-year smoothing spline
# -----------------------------
spl.10 <- ffcsaps(
  TRWstd,
  nyrs = 10
)


# -----------------------------
# Create high-resolution PNG
# -----------------------------
png(
  filename = output_file,
  width = 10,
  height = 6,
  units = "in",
  res = 600
)


# -----------------------------
# Plot settings
# -----------------------------
par(
  mar = c(5.5, 6, 2, 1.5),
  mgp = c(3.4, 1.0, 0),
  tcl = -0.35,
  xaxs = "i",
  yaxs = "r"
)


# -----------------------------
# Draw standardized TRW chronology
# -----------------------------
plot(
  yrs,
  TRWstd,
  type = "n",
  xlab = "Year",
  ylab = "Tree-Ring Width Index",
  axes = FALSE,
  cex.lab = 1.6,
  xlim = range(yrs),
  ylim = range(TRWstd) + c(-0.05, 0.05)
)


# -----------------------------
# Add grid
# -----------------------------
abline(
  h = pretty(range(TRWstd)),
  col = "grey88",
  lty = 1,
  lwd = 0.8
)


# -----------------------------
# Add reference line at TRWI = 1
# -----------------------------
abline(
  h = 1,
  col = "grey65",
  lty = 2,
  lwd = 1
)


# -----------------------------
# Plot annual standardized chronology
# -----------------------------
lines(
  yrs,
  TRWstd,
  col = "grey25",
  lwd = 1.3
)


# -----------------------------
# Add 10-year smoothing spline
# -----------------------------
lines(
  yrs,
  spl.10,
  col = "#d62728",
  lwd = 3
)


# -----------------------------
# X-axis
# Show years every 5 years,
# including the first and last year
# -----------------------------
year_min <- min(yrs)
year_max <- max(yrs)

x_ticks <- seq(
  from = ceiling(year_min / 5) * 5,
  to = floor(year_max / 5) * 5,
  by = 5
)

x_ticks <- sort(
  unique(
    c(year_min, x_ticks, year_max)
  )
)

axis(
  side = 1,
  at = x_ticks,
  labels = x_ticks,
  cex.axis = 1.35,
  las = 1
)


# -----------------------------
# Y-axis
# -----------------------------
axis(
  side = 2,
  cex.axis = 1.35,
  las = 1
)


# -----------------------------
# Add plot border
# No duplicated axes or tick labels
# -----------------------------
box(
  bty = "l",
  lwd = 1.2
)


# -----------------------------
# Legend
# Bottom-left with transparent background
# -----------------------------
legend(
  "bottomleft",
  legend = c(
    "Annual TRW chronology",
    "10-year spline"
  ),
  col = c(
    "grey25",
    "#d62728"
  ),
  lwd = c(
    1.3,
    3
  ),
  bty = "n",
  cex = 1.3,
  inset = c(0.015, 0.02)
)


# -----------------------------
# Close and save figure
# -----------------------------
dev.off()

print(
  paste(
    "Saved:",
    output_file
  )
)

################################################################################

