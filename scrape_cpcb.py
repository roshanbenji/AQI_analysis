"""
CPCB Historical AQI Data Scraper
=================================
Fetches hourly raw AQI data files from the CPCB Data Repository
(airquality.cpcb.gov.in) for one station per city, years 2020–2025.

How it works:
1. Fetches the complete station list from CPCB's all_india_stationlist API
2. Picks one station per city (prefers CPCB-operated central stations)
3. For each station, queries available file paths (POST with base64 body)
4. Downloads each CSV for years 2020 onward
5. Saves files to: ./cpcb_data/{state}/{city}/{station_name}_{year}.csv

Usage:
    python scrape_cpcb.py
"""

import requests
import base64
import json
import os
import time

# ── Config ──────────────────────────────────────────────────────────────────
BASE_URL = "https://airquality.cpcb.gov.in"
STATION_LIST_ENDPOINT = f"{BASE_URL}/dataRepository/all_india_stationlist"
FILE_PATH_ENDPOINT = f"{BASE_URL}/dataRepository/file_Path"
DOWNLOAD_ENDPOINT = f"{BASE_URL}/dataRepository/download_file"

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpcb_data")
TARGET_YEARS = [str(y) for y in range(2020, 2027)]  # 2020 through 2026
FREQUENCY = "1day"  # No space - must match exactly what the CPCB site sends
DATA_TYPE = "raw"

# Throttle to be respectful to the server
REQUEST_DELAY = 1.0  # seconds between requests

HEADERS = {
    "Accept": "q=0.8;application/json;q=0.9",
    "Origin": "https://airquality.cpcb.gov.in",
    "Referer": "https://airquality.cpcb.gov.in/ccr/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


# ── Helper Functions ────────────────────────────────────────────────────────

def b64_encode(data: dict) -> str:
    """Base64 encode a dict as JSON string."""
    return base64.b64encode(json.dumps(data).encode("utf-8")).decode("utf-8")


def b64_decode(encoded: str) -> dict:
    """Base64 decode a string back to dict."""
    try:
        return json.loads(base64.b64decode(encoded.encode("utf-8")).decode("utf-8"))
    except Exception:
        return {}


def safe_filename(name: str) -> str:
    """Remove/replace characters that are invalid in file paths."""
    for ch in '/\\:*?"<>|,':
        name = name.replace(ch, '_')
    return name.strip()


def pick_best_station(stations: list) -> dict:
    """
    Pick the best (most central) station from a list for a city.
    Priority: CPCB-operated > state board > others.
    Avoids industrial areas, airports, and university stations.
    """
    if not stations:
        return None

    avoid_keywords = ['airport', 'industrial', 'riico', 'midc', 'gidc', 'sidco']
    prefer_keywords = ['cpcb', 'central', 'civil line', 'collectorate', 'sector']

    # Filter out undesirable stations
    filtered = [s for s in stations
                if not any(kw in s['label'].lower() for kw in avoid_keywords)]
    if not filtered:
        filtered = stations  # Fall back to full list if all filtered out

    # Prefer CPCB-operated or central stations
    for kw in prefer_keywords:
        preferred = [s for s in filtered if kw in s['label'].lower()]
        if preferred:
            return preferred[0]

    # Default: first available station
    return filtered[0]


# ── Core Functions ──────────────────────────────────────────────────────────

def fetch_station_list():
    """
    Fetch the complete station list from CPCB's all_india_stationlist API.
    Returns list of dicts: [{station_id, station_name, state, city}, ...]
    One station per city (the most central one).
    """
    print("Fetching complete station list from CPCB...")

    try:
        resp = requests.post(
            STATION_LIST_ENDPOINT,
            data="e30=",  # Base64 for {}
            headers={
                **HEADERS,
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"
            },
            timeout=30
        )

        if resp.status_code != 200:
            print(f"  Error: HTTP {resp.status_code}")
            return []

        decoded = b64_decode(resp.text.strip())
        if decoded.get("status") != "success" or "dropdown" not in decoded:
            print(f"  Error: unexpected response format")
            return []

        dropdown = decoded["dropdown"]
        cities_by_state = dropdown.get("cities", {})
        stations_by_city = dropdown.get("stations", {})

        result = []
        for state, city_list in cities_by_state.items():
            for city_obj in city_list:
                city_name = city_obj["value"]
                city_stations = stations_by_city.get(city_name, [])

                if not city_stations:
                    continue

                best = pick_best_station(city_stations)
                if best:
                    result.append({
                        "station_id": best["value"],
                        "station_name": best["label"],
                        "state": state,
                        "city": city_name
                    })

        print(f"  Found {len(result)} cities across {len(cities_by_state)} states")
        return result

    except Exception as e:
        print(f"  Error fetching station list: {e}")
        return []


def get_file_paths(station_id: str, station_name: str):
    """
    POST to CPCB file_Path endpoint to get available CSV file paths.
    Returns list of {filepath, year} dicts.
    """
    payload = {
        "station_id": station_id,
        "station_name": station_name,
        "state": "",
        "city": "",
        "frequency": FREQUENCY,
        "dataType": DATA_TYPE
    }

    encoded_body = b64_encode(payload)

    try:
        resp = requests.post(
            FILE_PATH_ENDPOINT,
            data=encoded_body,
            headers={
                **HEADERS,
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"
            },
            timeout=30
        )

        if resp.status_code == 200:
            decoded = b64_decode(resp.text.strip())
            if decoded.get("status") == "success" and "data" in decoded:
                return decoded["data"]

        return []
    except Exception as e:
        print(f"    Error fetching file paths: {e}")
        return []


def download_file(filepath: str, save_path: str):
    """Download a CSV file from the CPCB repository."""
    download_url = f"{DOWNLOAD_ENDPOINT}?file_name={filepath}"

    try:
        resp = requests.get(download_url, headers=HEADERS, timeout=120, stream=True)

        if resp.status_code == 200:
            content = resp.content
            if len(content) > 100:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'wb') as f:
                    f.write(content)
                return True
            else:
                print(f"    Skipped (empty response): {filepath}")
                return False
        else:
            print(f"    HTTP {resp.status_code}: {filepath}")
            return False
    except Exception as e:
        print(f"    Download error: {e}")
        return False


# ── Main Scraper ────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  CPCB Historical AQI Data Scraper")
    print(f"  Target years: {', '.join(TARGET_YEARS)}")
    print(f"  Frequency: {FREQUENCY}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 70)

    # Step 1: Get station list (one per city)
    stations = fetch_station_list()
    if not stations:
        print("\nFailed to fetch station list. Exiting.")
        return

    print(f"\nProcessing {len(stations)} cities...\n")

    total_downloaded = 0
    total_skipped = 0
    total_failed = 0
    total_no_data = 0

    for i, station in enumerate(stations):
        station_id = station["station_id"]
        station_name = station["station_name"]
        state = station["state"]
        city = station["city"]

        print(f"[{i+1}/{len(stations)}] {city}, {state}")
        print(f"         Station: {station_name} ({station_id})")

        # Step 2: Get available file paths for this station
        file_data = get_file_paths(station_id, station_name)
        time.sleep(REQUEST_DELAY)

        if not file_data:
            print(f"         No files available")
            total_no_data += 1
            continue

        # Step 3: Download files for target years
        for entry in file_data:
            year = str(entry.get("year", ""))
            filepath = entry.get("filepath", "")

            if year not in TARGET_YEARS or not filepath:
                continue

            # Build save path
            save_path = os.path.join(
                OUTPUT_DIR,
                safe_filename(state),
                safe_filename(city),
                f"{safe_filename(station_name)}_{year}.csv"
            )

            # Skip if already downloaded
            if os.path.exists(save_path):
                print(f"         {year}: Already exists, skipping")
                total_skipped += 1
                continue

            # Download
            print(f"         {year}: Downloading...", end=" ")
            success = download_file(filepath, save_path)

            if success:
                size_kb = os.path.getsize(save_path) / 1024
                print(f"OK ({size_kb:.0f} KB)")
                total_downloaded += 1
            else:
                total_failed += 1

            time.sleep(REQUEST_DELAY)

    # Summary
    print("\n" + "=" * 70)
    print(f"  DONE!")
    print(f"  Downloaded:  {total_downloaded} files")
    print(f"  Skipped:     {total_skipped} files (already exist)")
    print(f"  No data:     {total_no_data} stations")
    print(f"  Failed:      {total_failed} files")
    print(f"  Output:      {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
