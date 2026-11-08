"""Flight search using the Ignav API (https://ignav.com/docs).

This file ONLY talks to Ignav and returns its raw itineraries.
Turning them into our own format happens in agent.py (normalize_flights).
"""
import os

import httpx

from schemas import TravelError

IGNAV_URL = "https://ignav.com/api/fares/one-way"


def search_flights_api(origin_airport, destination_airport, travel_date, flight_class):
    """Call Ignav's one-way fare search and return the list of raw itineraries."""
    api_key = os.getenv("IGNAV_API_KEY")
    if not api_key:
        raise TravelError("Flight search is not set up: IGNAV_API_KEY is missing in backend/.env")

    body = {
        "origin": origin_airport,            # e.g. "AMD"
        "destination": destination_airport,  # e.g. "BLR"
        "departure_date": travel_date,       # "YYYY-MM-DD"
        "cabin_class": flight_class,         # economy / premium_economy / business / first
        "market": "IN",                      # asks Ignav for prices in Indian market (INR)
    }

    try:
        response = httpx.post(IGNAV_URL, headers={"X-Api-Key": api_key}, json=body, timeout=30)
    except httpx.HTTPError:
        raise TravelError("Could not reach the flight service. Please try again.")

    if response.status_code != 200:
        raise TravelError(f"Flight service returned an error (code {response.status_code}).")

    try:
        data = response.json()
    except ValueError:
        raise TravelError("Flight service sent an invalid response.")

    return data.get("itineraries", [])
