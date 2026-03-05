"""
Geocode Indian Cities
=====================
Looks up latitude/longitude for city names using OpenStreetMap (via geopy).
Results are cached to city_coordinates.json so lookups only happen once.

Usage:
    from geocode_cities import get_city_coordinates
    coords = get_city_coordinates([("Delhi", "Delhi"), ("Mumbai", "Maharashtra")])
"""

import os
import json
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

GEOCODE_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "city_coordinates.json")


def get_city_coordinates(city_state_pairs):
    """
    Look up latitude/longitude for each (city, state) pair.

    Args:
        city_state_pairs: list of (city_name, state_name) tuples

    Returns:
        dict like {"Delhi, Delhi": {"lat": 28.6, "lon": 77.2}, ...}
        Returns None for cities that couldn't be found.
    """
    # Load cache if it exists
    cache = {}
    if os.path.exists(GEOCODE_CACHE):
        with open(GEOCODE_CACHE, "r") as f:
            cache = json.load(f)

    # Set up geocoder (OpenStreetMap, free, no API key needed)
    geolocator = Nominatim(user_agent="aqi_analysis_project")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.1)

    new_lookups = 0
    for city, state in city_state_pairs:
        key = f"{city}, {state}"

        # Skip if already cached
        if key in cache:
            continue

        # Look up coordinates
        try:
            location = geocode(f"{city}, {state}, India")
            if not location:
                location = geocode(f"{city}, India")

            if location:
                cache[key] = {"lat": location.latitude, "lon": location.longitude}
                print(f"  Found: {key} → ({location.latitude:.4f}, {location.longitude:.4f})")
            else:
                cache[key] = None
                print(f"  Not found: {key}")
            new_lookups += 1
        except Exception as e:
            cache[key] = None
            print(f"  Error: {key}: {e}")

    # Save cache
    with open(GEOCODE_CACHE, "w") as f:
        json.dump(cache, f, indent=2)

    if new_lookups:
        print(f"  Geocoded {new_lookups} new cities (total cached: {len(cache)})")
    else:
        print(f"  All {len(cache)} cities already cached")

    return cache
