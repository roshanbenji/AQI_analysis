"""
Merge CPCB CSV Files
====================
Walks the cpcb_data/ folder, loads every CSV, and combines them into
one big DataFrame with state, city, latitude, and longitude columns added.

Usage:
    from merge_csvs import load_all_csvs
    df = load_all_csvs(coordinates_dict)
"""

import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpcb_data")


def find_all_csvs():
    """
    Scan cpcb_data/ and return a list of dicts with path, state, city
    for every CSV file found.
    """
    csv_files = []
    for state_folder in sorted(os.listdir(DATA_DIR)):
        state_path = os.path.join(DATA_DIR, state_folder)
        if not os.path.isdir(state_path):
            continue
        for city_folder in sorted(os.listdir(state_path)):
            city_path = os.path.join(state_path, city_folder)
            if not os.path.isdir(city_path):
                continue
            for filename in sorted(os.listdir(city_path)):
                if filename.endswith(".csv"):
                    csv_files.append({
                        "path": os.path.join(city_path, filename),
                        "state": state_folder,
                        "city": city_folder,
                    })
    return csv_files


def get_unique_cities(csv_files):
    """Get a list of unique (city, state) pairs from the CSV file list."""
    return list(set((f["city"], f["state"]) for f in csv_files))


def load_all_csvs(csv_files, coordinates):
    """
    Load all CSVs into one DataFrame, adding state, city, lat, lon columns.

    Args:
        csv_files: list from find_all_csvs()
        coordinates: dict from get_city_coordinates(), maps "city, state" → {lat, lon}

    Returns:
        One merged pandas DataFrame
    """
    all_data = []
    skipped = 0

    for i, file_info in enumerate(csv_files):
        # Look up coordinates for this city
        key = f"{file_info['city']}, {file_info['state']}"
        coords = coordinates.get(key)
        if not coords:
            skipped += 1
            continue

        # Read the CSV and add metadata columns
        df = pd.read_csv(file_info["path"], low_memory=False)
        df["state"] = file_info["state"]
        df["city"] = file_info["city"]
        df["latitude"] = coords["lat"]
        df["longitude"] = coords["lon"]
        all_data.append(df)

        if (i + 1) % 200 == 0:
            print(f"  Loaded {i+1}/{len(csv_files)} files...")

    merged = pd.concat(all_data, ignore_index=True)
    print(f"  Total: {len(merged):,} rows from {len(all_data)} files (skipped {skipped})")
    return merged


# ═══════════════════════════════════════════════════════════════════════════
# Run this script to produce merged_data.csv
# ═══════════════════════════════════════════════════════════════════════════

from geocode_cities import get_city_coordinates

MERGED_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "merged_data.csv")

print("Finding CSV files...")
csv_files = find_all_csvs()
print(f"  Found {len(csv_files)} files")

print("Geocoding cities...")
unique_cities = get_unique_cities(csv_files)
coordinates = get_city_coordinates(unique_cities)

print("Merging all CSVs...")
merged = load_all_csvs(csv_files, coordinates)

print(f"Saving to merged_data.csv...")
merged.to_csv(MERGED_OUTPUT, index=False)
size_mb = os.path.getsize(MERGED_OUTPUT) / (1024 * 1024)
print(f"  Done! {len(merged):,} rows, {size_mb:.1f} MB")

