################################################################################
# Plot standardized tree-ring width (TRW) chronology
#
# Input Excel structure:
# Year | std | samp.depth
#
# The script:
# 1. Reads the already standardized TRW chronology from Excel
# 2. Uses the "std" column as the standardized tree-ring width index
# 3. Plots the annual chronology
# 4. Adds 16-year, 64-year, and default smoothing splines
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
# Input file
# -----------------------------
input_file <- "E:/FAU master/Master Thesis/Data/Tree Ring Width Chronology/50years-ads.xlsx"

# -----------------------------
# Read standardized TRW chronology
# -----------------------------
data.crn <- read_excel(input_file)


# -----------------------------
# Check column names and data
# Expected columns:
# Year | std | samp.depth
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
# Plot settings
# Based on the original dplR plotting code
# -----------------------------
par(
  tcl = 0.5,
  mgp = c(1.25, 0.25, 0),
  xaxs = "i",
  mar = rep(3, 4)
)


# -----------------------------
# Draw standardized TRW chronology
# -----------------------------
plot(
  yrs,
  TRWstd,
  type = "n",
  ylab = "Tree-Ring Width Index",
  xlab = "Year",
  axes = FALSE
)

grid(
  lty = 1,
  col = "grey"
)

lines(
  yrs,
  TRWstd,
  col = "grey10"
)


# -----------------------------
# Add 16-year smoothing spline
# -----------------------------
spl.16 <- ffcsaps(
  TRWstd,
  nyrs = 16
)

lines(
  yrs,
  spl.16,
  col = "red",
  lwd = 2
)


# -----------------------------
# Add 64-year smoothing spline
# -----------------------------
spl.64 <- ffcsaps(
  TRWstd,
  nyrs = 64
)

lines(
  yrs,
  spl.64,
  col = "green",
  lwd = 2
)


# -----------------------------
# Add default smoothing spline
# Default nyrs = approximately half the chronology length
# -----------------------------
spl.def <- ffcsaps(TRWstd)

lines(
  yrs,
  spl.def,
  col = "blue",
  lwd = 2
)


# -----------------------------
# Add legend
# -----------------------------
legend(
  "bottomright",
  c(
    "Chronology",
    "nyrs = 16",
    "nyrs = 64",
    paste(
      "Default nyrs (",
      floor(length(TRWstd) / 2),
      ")",
      sep = ""
    )
  ),
  fill = c(
    "grey10",
    "red",
    "green",
    "blue"
  ),
  bg = "grey90"
)


# -----------------------------
# Add axes and plot border
# -----------------------------
axis(1)
axis(2)
axis(3)
axis(4)

box()

################################################################################