"""Train search using the IRCTC API on RapidAPI (host: irctc1.p.rapidapi.com).

Two calls are used:
  1. trainBetweenStations -> list of trains (times, duration, classes offered)
  2. getFare              -> fare per class, for ONE train at a time

NOTE: the free RapidAPI plan has a very small monthly quota, so we only ask
for fares of the first few trains that offer the requested class.
Change MAX_FARE_LOOKUPS if you have a bigger plan.
"""
import os

import httpx

from schemas import TravelError

HOST = "irctc1.p.rapidapi.com"
MAX_FARE_LOOKUPS = 5


def _get(path, params):
    """Make one GET request to the RapidAPI IRCTC service and return the JSON."""
    api_key = os.getenv("RAPIDAPI_KEY")
    if not api_key:
        raise TravelError("Train search is not set up: RAPIDAPI_KEY is missing in backend/.env")

    headers = {"x-rapidapi-key": api_key, "x-rapidapi-host": HOST}
    try:
        response = httpx.get(f"https://{HOST}{path}", headers=headers, params=params, timeout=30)
    except httpx.HTTPError:
        raise TravelError("Could not reach the train service. Please try again.")

    if response.status_code == 429:
        raise TravelError("Train API limit reached (too many requests on the free plan).")
    if response.status_code != 200:
        raise TravelError(f"Train service returned an error (code {response.status_code}).")

    try:
        return response.json()
    except ValueError:
        raise TravelError("Train service sent an invalid response.")


def _get_fare(train_number, from_code, to_code, train_class):
    """Return the fare (number) for one train and class, or None if not found."""
    try:
        data = _get("/api/v2/getFare", {
            "trainNo": train_number,
            "fromStationCode": from_code,
            "toStationCode": to_code,
        })
    except TravelError:
        return None  # one failed fare should not kill the whole search

    fares = (data.get("data") or {}).get("general") or []
    for item in fares:
        if str(item.get("classType", "")).upper() == train_class:
            try:
                return float(item["fare"])
            except (KeyError, TypeError, ValueError):
                return None
    return None


def search_trains_api(from_station, to_station, travel_date, train_class):
    """Return raw train dicts that offer train_class. Each gets a 'fare' key added."""
    data = _get("/api/v3/trainBetweenStations", {
        "fromStationCode": from_station,
        "toStationCode": to_station,
        "dateOfJourney": travel_date,
    })

    trains = data.get("data") or []
    if isinstance(trains, dict):          # some responses wrap the list once more
        trains = trains.get("data") or []

    # keep only trains that offer the class the user asked for
    matching = [t for t in trains if train_class in (t.get("class_type") or [])]

    matching = matching[:MAX_FARE_LOOKUPS]
    for train in matching:
        train["fare"] = _get_fare(train.get("train_number"), from_station, to_station, train_class)

    return matching
