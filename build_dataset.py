"""
Build ML-Ready Dataset for AQI Prediction
==========================================
Uses geocode_cities.py and merge_csvs.py to:
1. Find all CPCB daily CSV files
2. Geocode each city (lat/lon)
3. Merge into one DataFrame
4. Calculate AQI from raw pollutant concentrations
5. Select weather + location features, clean, and save

Features (X): AT, RH, WS, BP, SR, latitude, longitude
Target  (y): Calculated_AQI

Run: python build_dataset.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler

OUTPUT_FILE = "ml_ready_dataset.csv"


# ═══════════════════════════════════════════════════════════════════════════
# AQI CALCULATION
# ═══════════════════════════════════════════════════════════════════════════
#
# AQI is calculated using the CPCB formula:
#
#   Sub-Index = ((I_hi - I_lo) / (BP_hi - BP_lo)) × (C - BP_lo) + I_lo
#
# Where:
#   C     = pollutant concentration (sensor reading)
#   BP    = breakpoint concentration for this AQI range
#   I     = AQI score for this range
#
# Overall AQI = maximum of all individual sub-indices.
# ═══════════════════════════════════════════════════════════════════════════

# Breakpoint tables: (concentration_low, concentration_high, aqi_low, aqi_high)
# Ranges must be continuous (no gaps) to handle decimal sensor values
BREAKPOINTS = {
    "PM2.5 (µg/m³)": [
        (0, 30, 0, 50), (30, 60, 51, 100), (60, 90, 101, 200),
        (90, 120, 201, 300), (120, 250, 301, 400), (250, 500, 401, 500),
    ],
    "PM10 (µg/m³)": [
        (0, 50, 0, 50), (50, 100, 51, 100), (100, 250, 101, 200),
        (250, 350, 201, 300), (350, 430, 301, 400), (430, 600, 401, 500),
    ],
    "NO2 (µg/m³)": [
        (0, 40, 0, 50), (40, 80, 51, 100), (80, 180, 101, 200),
        (180, 280, 201, 300), (280, 400, 301, 400), (400, 800, 401, 500),
    ],
    "SO2 (µg/m³)": [
        (0, 40, 0, 50), (40, 80, 51, 100), (80, 380, 101, 200),
        (380, 800, 201, 300), (800, 1600, 301, 400), (1600, 3200, 401, 500),
    ],
    "CO (mg/m³)": [
        (0, 1.0, 0, 50), (1.0, 2.0, 51, 100), (2.0, 10.0, 101, 200),
        (10.0, 17.0, 201, 300), (17.0, 34.0, 301, 400), (34.0, 68.0, 401, 500),
    ],
    "Ozone (µg/m³)": [
        (0, 50, 0, 50), (50, 100, 51, 100), (100, 168, 101, 200),
        (168, 208, 201, 300), (208, 748, 301, 400), (748, 1500, 401, 500),
    ],
    "NH3 (µg/m³)": [
        (0, 200, 0, 50), (200, 400, 51, 100), (400, 800, 101, 200),
        (800, 1200, 201, 300), (1200, 1800, 301, 400), (1800, 3600, 401, 500),
    ],
}


def get_sub_index(concentration, breakpoints):
    """
    Calculate AQI sub-index for a single pollutant.

    Example: PM2.5 = 75 µg/m³
        Falls in range (61, 90) → AQI range (101, 200)
        Sub-index = ((200-101)/(90-61)) × (75-61) + 101 = 148.8
    """
    if pd.isna(concentration) or concentration < 0:
        return np.nan

    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= concentration <= c_hi:
            return ((i_hi - i_lo) / (c_hi - c_lo)) * (concentration - c_lo) + i_lo

    if concentration > breakpoints[-1][1]:
        return 500
    return np.nan


def calculate_aqi_for_row(row):
    """Calculate overall AQI = max sub-index across all 7 pollutants."""
    sub_indices = []
    for column_name, breakpoints in BREAKPOINTS.items():
        if column_name in row.index:
            si = get_sub_index(row[column_name], breakpoints)
            if not pd.isna(si):
                sub_indices.append(si)

    return max(sub_indices) if sub_indices else np.nan


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("  Building ML-Ready Dataset")
print("=" * 60)

# Load the merged data (produced by merge_csvs.py)
MERGED_FILE = "merged_data.csv"
merged = pd.read_csv(MERGED_FILE, low_memory=False)
print(f"  {len(merged):,} rows loaded")

print("\nColumns:")
print(merged.columns)
print("\nDescribe:")
print(merged.describe())
print("\nHead:")
print(merged.head())

# Step 1: Calculate AQI
print("\n[Step 1] Calculating AQI")
merged["Calculated_AQI"] = merged.apply(calculate_aqi_for_row, axis=1)
valid = merged["Calculated_AQI"].notna().sum()
print(f"  Valid AQI: {valid:,} / {len(merged):,} ({valid/len(merged)*100:.1f}%)")

# Step 2: Select and rename columns
print("\n[Step 2] Selecting features...")
final = merged.rename(columns={
    "AT (°C)":    "AT (Temperature)",
    "RH (%)":     "RH (Humidity)",
    "WS (m/s)":   "WS (Wind Speed)",
    "BP (mmHg)":  "BP (Barometric Pressure)",
    "SR (W/mt2)": "SR (Solar Radiation)",
})[["AT (Temperature)", "RH (Humidity)", "WS (Wind Speed)", "BP (Barometric Pressure)", "SR (Solar Radiation)", "latitude", "longitude", "Calculated_AQI"]]

# Step 3: Clean
print("\n[Step 3] Cleaning...")

print("\nMissing values:")
print(final.isna().sum())

print("\nHead:")
print(final.head())

print("\nColumns:")
print(final.columns)

print("\nInfo:")
print(final.info())

print("\nDescribe:")
print(final.describe())

for i in final.columns:
    sns.histplot(final[i], kde=True)
    plt.show()

sns.heatmap(final.corr())
plt.show()

print(final.corr())

for col in final.columns:
    plt.figure()
    sns.boxplot(y=final[col])
    plt.title(f"Boxplot of {col}")
    plt.xticks(rotation=90)
    plt.show()

before = len(final)
final = final.dropna(subset=["Calculated_AQI"])
print(f"  Removed {before - len(final):,} rows with no AQI")

before = len(final)
final = final.dropna(subset=["BP (Barometric Pressure)"])
print(f"  Removed {before - len(final):,} rows with no BP")

# Fill remaining missing weather values with median
for col in ["AT (Temperature)", "RH (Humidity)", "WS (Wind Speed)", "SR (Solar Radiation)"]:
    missing = final[col].isna().sum()
    if missing > 0:
        median = final[col].median()
        final[col] = final[col].fillna(median)
        print(f"  Filled {missing:,} missing {col} values with median ({median:.2f})")

# Outlier Capping (sensor columns only — skip AT and lat/lon)
print("\n[Step 3b] IQR Capping — preview (AT skipped):")
iqr_cols = ["RH (Humidity)", "WS (Wind Speed)", "BP (Barometric Pressure)", "SR (Solar Radiation)"]
for col in iqr_cols:
    Q1 = final[col].quantile(0.25)
    Q3 = final[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    below = (final[col] < lower).sum()
    above = (final[col] > upper).sum()
    total = below + above
    print(f"  {col}: {total:,} values to cap  ({below:,} below {lower:.2f}, {above:,} above {upper:.2f})")

for col in iqr_cols:
    Q1 = final[col].quantile(0.25)
    Q3 = final[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    final[col] = final[col].clip(lower, upper)
print("  Capping applied!")

print("\nDescribe after capping:")
print(final.describe())

# Step 3c: RobustScaler (sensor columns only, not lat/lon or AQI)
print("\n[Step 3c] Applying RobustScaler to sensor columns...")
scale_cols = ["AT (Temperature)", "RH (Humidity)", "WS (Wind Speed)", "BP (Barometric Pressure)", "SR (Solar Radiation)"]
scaler = RobustScaler()
final[scale_cols] = scaler.fit_transform(final[scale_cols])
print("  Scaled columns:", scale_cols)
print("  (latitude, longitude, Calculated_AQI left unscaled)")

print("\nDescribe after scaling:")
print(final.describe())

for i in final.columns:
    sns.histplot(final[i], kde=True)
    plt.show()

for col in final.columns:
    plt.figure()
    sns.boxplot(y=final[col])
    plt.title(f"Boxplot of {col}")
    plt.xticks(rotation=90)
    plt.show() 

# Step 4: Save
print(f"\n[Step 4] Saving...")
final.to_csv(OUTPUT_FILE, index=False)

# Summary
print("\n" + "=" * 60)
print(f"  DONE! {len(final):,} rows")
print(f"  Features: AT (Temperature), RH (Humidity), WS (Wind Speed), BP (Barometric Pressure), SR (Solar Radiation), latitude, longitude")
print(f"  Target:   Calculated_AQI (range {final['Calculated_AQI'].min():.0f}–{final['Calculated_AQI'].max():.0f}, mean {final['Calculated_AQI'].mean():.1f})")
print("=" * 60)