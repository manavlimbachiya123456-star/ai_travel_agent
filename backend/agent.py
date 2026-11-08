"""The whole LangGraph agent in ONE file.

Flow:
  START -> parse_prompt -> validate_trip --(invalid)--> ask_user -> END
                                         --(valid)--> location_lookup
  location_lookup -> search_flights -> normalize_flights -> filter_flights -> score_flights -> rank_flights --+
  location_lookup -> search_trains  -> normalize_trains  -> filter_trains  -> score_trains  -> rank_trains  --+
  (both branches run side by side, then join)
     -> compare_options -> pick_top_3 -> ai_recommendation -> return_results -> END

The user types ONE text request. parse_prompt (Gemini) turns it into JSON: the trip fields
(cities, date, passengers, classes) and the conditions (times, prices, stops...).
Missing details get defaults. The filter_* nodes then apply the conditions to the REAL
API results, and pick_top_3 chooses the 3 best options overall.

Every node is a small function: it receives the shared "state" (a dict)
and returns ONLY the keys it wants to change.
"""
import json
import os
import time
from datetime import date, datetime, timedelta
from typing import TypedDict

import httpx
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from flight_api import search_flights_api
from schemas import TravelError
from train_api import search_trains_api

load_dotenv()

# ---------------------------------------------------------------------------
# Simple lookup data (add more cities here whenever you like)
# city -> (airport code, railway station code)
# ---------------------------------------------------------------------------
CITIES = {
    "ahmedabad": ("AMD", "ADI"),
    "bangalore": ("BLR", "SBC"),
    "mumbai": ("BOM", "CSMT"),
    "delhi": ("DEL", "NDLS"),
    "chennai": ("MAA", "MAS"),
    "kolkata": ("CCU", "HWH"),
    "hyderabad": ("HYD", "SC"),
    "pune": ("PNQ", "PUNE"),
    "jaipur": ("JAI", "JP"),
    "lucknow": ("LKO", "LKO"),
    "goa": ("GOI", "MAO"),
    "kochi": ("COK", "ERS"),
    "surat": ("STV", "ST"),
    "vadodara": ("BDQ", "BRC"),
    "rajkot": ("RAJ", "RJT"),
    "nagpur": ("NAG", "NGP"),
    "indore": ("IDR", "INDB"),
    "bhopal": ("BHO", "BPL"),
    "patna": ("PAT", "PNBE"),
    "chandigarh": ("IXC", "CDG"),
    "thiruvananthapuram": ("TRV", "TVC"),
    "varanasi": ("VNS", "BSB"),
    "amritsar": ("ATQ", "ASR"),
    "coimbatore": ("CJB", "CBE"),
    "visakhapatnam": ("VTZ", "VSKP"),
}
# other spellings people use -> the name used in CITIES
ALIASES = {
    "bengaluru": "bangalore", "bombay": "mumbai", "new delhi": "delhi",
    "madras": "chennai", "calcutta": "kolkata", "baroda": "vadodara",
    "cochin": "kochi", "trivandrum": "thiruvananthapuram", "vizag": "visakhapatnam",
}

FLIGHT_CLASSES = ["economy", "premium_economy", "business", "first"]
TRAIN_CLASSES = ["2S", "SL", "3A", "2A", "1A", "CC", "EC"]   # 2S = "General" (second sitting)
MAX_STOPS = 2          # flights with more stops than this are dropped
MAX_PASSENGERS = 6

# The constraints Gemini is allowed to return, grouped by type.
# (Anything else Gemini returns is thrown away.)
TIME_KEYS = ["arrival_before", "arrival_after", "departure_after", "departure_before"]   # "HH:MM"
NUMBER_KEYS = ["airport_buffer_minutes", "station_buffer_minutes", "max_flight_price",
               "max_train_price", "max_total_price", "max_flight_stops",
               "max_train_duration_minutes", "min_time_saved_minutes"]
BOOL_KEYS = ["prefer_flight", "prefer_train"]


class TripState(TypedDict, total=False):
    """Everything the agent remembers while it runs."""
    # input
    prompt: str                # what the user typed in the one text box
    # filled in by parse_prompt (from Gemini's JSON, plus defaults)
    origin: str
    destination: str
    travel_date: str
    passengers: int
    flight_class: str
    train_class: str
    conditions: str            # the user's original text (used in the recommendation)
    parsed_conditions: dict    # structured constraints extracted by Gemini ({} if none)
    assumptions: list          # defaults we used for things the user did not say
    # validation / lookup
    error: str
    airports: dict
    stations: dict
    # flights
    raw_flights: list
    flights: list
    flight_error: str
    flights_removed_by_conditions: int
    best_flight: dict
    cheapest_flight: dict
    fastest_flight: dict
    # trains
    raw_trains: list
    trains: list
    train_error: str
    trains_removed_by_conditions: int
    best_train: dict
    cheapest_train: dict
    fastest_train: dict
    # final
    comparison: dict
    top_options: list          # the 3 best options overall (flights and/or trains)
    recommendation: str
    result: dict


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def clean_city(name):
    """Lowercase a city name and map spelling variants (Bengaluru -> bangalore)."""
    name = (name or "").strip().lower()
    return ALIASES.get(name, name)


def format_duration(minutes):
    """160 -> '2h 40m'"""
    return f"{minutes // 60}h {minutes % 60}m"


def format_price(amount):
    """4850 -> '₹4,850'"""
    return f"₹{amount:,.0f}"


# ---------------------------------------------------------------------------
# NODE 1: parse_prompt - Gemini turns the user's ONE text box into structured JSON.
# Anything the user did not mention gets a sensible default (and is listed in "assumptions").
# ---------------------------------------------------------------------------
PROMPT_PARSER = """You convert a traveller's request into JSON.
Return ONLY valid JSON. No explanation, no markdown.
Today's date is {today} ({weekday}).

Use exactly this structure. Use null for anything the user did not mention. Do NOT invent anything.

{{
  "origin": "city name" or null,
  "destination": "city name" or null,
  "travel_date": "YYYY-MM-DD" or null,
  "passengers": number or null,
  "flight_class": "economy" or "premium_economy" or "business" or "first" or null,
  "train_class": "General" or "SL" or "3A" or "2A" or "1A" or "CC" or "EC" or null,
  "conditions": {{
    "arrival_before": "HH:MM" or null,
    "arrival_after": "HH:MM" or null,
    "departure_after": "HH:MM" or null,
    "departure_before": "HH:MM" or null,
    "airport_buffer_minutes": number or null,
    "station_buffer_minutes": number or null,
    "max_flight_price": number or null,
    "max_train_price": number or null,
    "max_total_price": number or null,
    "max_flight_stops": number or null,
    "max_train_duration_minutes": number or null,
    "prefer_flight": true/false or null,
    "prefer_train": true/false or null,
    "min_time_saved_minutes": number or null
  }}
}}

Rules:
- origin and destination: the Indian city names, in English (Bengaluru -> "Bangalore").
- travel_date: "YYYY-MM-DD". Work out words like "tomorrow" or "next Friday" from today's date.
  If the year is missing, use the next upcoming date.
- Times must be 24-hour "HH:MM" (7 PM -> "19:00", 9 AM -> "09:00").
- Durations must be in minutes (1 hour 30 minutes -> 90).
- Prices must be plain numbers (₹8000 -> 8000).
- "flights before 3 PM", "leave/depart/take off before 3 PM" -> departure_before "15:00".
- "reach/arrive/land before 3 PM" -> arrival_before "15:00".
- "not before 9 AM", "cannot leave before 9 AM" -> departure_after "09:00".
- "non-stop" -> max_flight_stops 0.
- Keep the user's meaning exactly. Do NOT make any travel recommendation.

User request:
{prompt}
"""


def clean_json_text(text):
    """Remove ```json fences if Gemini added them anyway."""
    text = text.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return text


def clean_conditions(raw):
    """Keep only the known condition keys with the right type, and drop every empty (null) one.

    Result example: {"arrival_before": "13:00", "max_flight_price": 9000}
    An empty dict {} means "the user gave no conditions".
    """
    cleaned = {}
    for key in TIME_KEYS:
        try:
            # accepts "9:00" or "09:00" and always stores "09:00"
            cleaned[key] = datetime.strptime(raw.get(key), "%H:%M").strftime("%H:%M")
        except (TypeError, ValueError):
            cleaned[key] = None
    for key in NUMBER_KEYS:
        value = raw.get(key)
        is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
        cleaned[key] = value if is_number and value >= 0 else None
    for key in BOOL_KEYS:
        value = raw.get(key)
        cleaned[key] = value if isinstance(value, bool) else None
    return {key: value for key, value in cleaned.items() if value is not None}


def parse_prompt(state: TripState):
    """Ask Gemini to convert the user's text into trip fields + conditions, then fill in defaults."""
    text = (state.get("prompt") or "").strip()
    if not text:
        return {"error": "Please type your trip, for example: Ahmedabad to Bangalore on 19 Nov, reach before 1 PM."}

    today = date.today()
    answer = ask_gemini(PROMPT_PARSER.format(today=today.isoformat(), weekday=today.strftime("%A"),
                                             prompt=text), json_mode=True)
    raw = None
    if answer:
        try:
            raw = json.loads(clean_json_text(answer))
        except ValueError:
            print(f"[parse_prompt] Gemini answer was not valid JSON: {answer[:200]}")
    if not isinstance(raw, dict):
        # without Gemini we cannot read the text, so stop with a clear message
        return {"error": "Could not understand your request. Please try again "
                         "(also check that GEMINI_API_KEY is set in backend/.env)."}

    assumptions = []   # every default we had to use, shown to the user

    # --- date: default = tomorrow ---
    try:
        travel_date = datetime.strptime(str(raw.get("travel_date")), "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        travel_date = (today + timedelta(days=1)).isoformat()
        assumptions.append(f"No travel date found, so I used tomorrow ({travel_date}).")

    # --- passengers: default = 1 ---
    passengers = raw.get("passengers")
    if isinstance(passengers, (int, float)) and not isinstance(passengers, bool):
        passengers = int(passengers)
    else:
        passengers = 1
        assumptions.append("Number of passengers not mentioned, so I used 1.")

    # --- flight class: default = economy ---
    flight_class = str(raw.get("flight_class") or "").strip().lower()
    if flight_class not in FLIGHT_CLASSES:
        flight_class = "economy"
        assumptions.append("Flight class not mentioned, so I used economy.")

    # --- train class: default = 3A ---
    train_class = str(raw.get("train_class") or "").strip().upper()
    if train_class == "GENERAL":
        train_class = "2S"                       # IRCTC calls General "2S"
    if train_class not in TRAIN_CLASSES:
        train_class = "3A"
        assumptions.append("Train class not mentioned, so I used 3A.")

    conditions = raw.get("conditions") if isinstance(raw.get("conditions"), dict) else {}

    return {
        # origin / destination have NO default: validate_trip asks the user if they are missing
        "origin": str(raw.get("origin") or "").strip(),
        "destination": str(raw.get("destination") or "").strip(),
        "travel_date": travel_date,
        "passengers": passengers,
        "flight_class": flight_class,
        "train_class": train_class,
        "conditions": text,                                  # the original text (used in the recommendation)
        "parsed_conditions": clean_conditions(conditions),   # only the conditions the user really gave
        "assumptions": assumptions,
    }


# ---------------------------------------------------------------------------
# NODE 3: validate_trip - check every field, store a friendly message in "error" if wrong
# ---------------------------------------------------------------------------
def validate_trip(state: TripState):
    if state.get("error"):      # parse_prompt already failed: keep its message
        return {}
    if not state["origin"] or not state["destination"]:
        return {"error": "Please mention both the starting city and the destination, "
                         "for example: Ahmedabad to Bangalore."}
    if clean_city(state["origin"]) not in CITIES:
        return {"error": f"I don't know the city '{state['origin']}'. Try cities like Ahmedabad, Mumbai, Delhi, Bangalore."}
    if clean_city(state["destination"]) not in CITIES:
        return {"error": f"I don't know the city '{state['destination']}'. Try cities like Ahmedabad, Mumbai, Delhi, Bangalore."}
    if clean_city(state["origin"]) == clean_city(state["destination"]):
        return {"error": "From and To cities are the same."}

    if not state["travel_date"]:
        return {"error": "Please choose a travel date."}
    try:
        trip_day = datetime.strptime(state["travel_date"], "%Y-%m-%d").date()
    except ValueError:
        return {"error": "The date is not valid. Use the format YYYY-MM-DD."}
    if trip_day < date.today():
        return {"error": "The travel date is in the past."}

    if not isinstance(state["passengers"], int) or not 1 <= state["passengers"] <= MAX_PASSENGERS:
        return {"error": f"Passengers must be between 1 and {MAX_PASSENGERS}."}
    if state["flight_class"] not in FLIGHT_CLASSES:
        return {"error": f"Flight class must be one of: {', '.join(FLIGHT_CLASSES)}."}
    if state["train_class"] not in TRAIN_CLASSES:
        return {"error": "Train class must be one of: General, SL, 3A, 2A, 1A, CC, EC."}

    return {"error": ""}   # empty error = everything is fine


def route_after_validation(state: TripState):
    """Conditional edge: decide which node runs next."""
    return "invalid" if state.get("error") else "valid"


# ---------------------------------------------------------------------------
# NODE 4 (invalid path): ask_user - stop and return the error so the user can fix the form
# ---------------------------------------------------------------------------
def ask_user(state: TripState):
    return {"result": {"error": state["error"]}}


# ---------------------------------------------------------------------------
# NODE 5: location_lookup - city name -> airport code AND railway station code (kept separate)
# ---------------------------------------------------------------------------
def location_lookup(state: TripState):
    from_airport, from_station = CITIES[clean_city(state["origin"])]
    to_airport, to_station = CITIES[clean_city(state["destination"])]
    return {
        "airports": {"from": from_airport, "to": to_airport},
        "stations": {"from": from_station, "to": to_station},
    }


# ---------------------------------------------------------------------------
# Applying the user's conditions (plain Python - Gemini is NOT involved here)
# ---------------------------------------------------------------------------
def at_time(travel_date, hhmm):
    """('2026-11-19', '19:00') -> datetime for 19 Nov 2026, 19:00"""
    return datetime.strptime(f"{travel_date} {hhmm}", "%Y-%m-%d %H:%M")


def passes_time_rules(item, conditions, travel_date, buffer_minutes):
    """Check departure/arrival time limits. All times are compared on the TRAVEL DATE.

    - departure_after + buffer: the user can't leave before e.g. 09:00 AND needs
      `buffer_minutes` to reach the airport/station, so the earliest valid departure is
      09:00 + buffer (a 09:20 flight with a 60-minute buffer is NOT valid; 10:00 is).
    - arrival_before: an overnight train that arrives the next day fails "reach before 7 PM"
      because it arrives after 19:00 on the travel date.
    """
    try:
        departure = datetime.fromisoformat(item["departure"])
        arrival = datetime.fromisoformat(item["arrival"])
    except ValueError:
        return True   # can't read the time, so don't throw the option away

    if conditions.get("departure_after"):
        earliest = at_time(travel_date, conditions["departure_after"]) + timedelta(minutes=buffer_minutes)
        if departure < earliest:
            return False
    if conditions.get("departure_before"):
        if departure > at_time(travel_date, conditions["departure_before"]):
            return False
    if conditions.get("arrival_before"):
        if arrival > at_time(travel_date, conditions["arrival_before"]):
            return False
    if conditions.get("arrival_after"):
        if arrival < at_time(travel_date, conditions["arrival_after"]):
            return False
    return True


def apply_flight_conditions(flights, conditions, travel_date):
    """Remove flights that break the user's conditions. (Prices are totals for all passengers.)"""
    buffer_minutes = conditions.get("airport_buffer_minutes") or 0
    kept = []
    for flight in flights:
        # "is not None" (not just "if value") so that max_flight_stops = 0 (non-stop only) works
        if conditions.get("max_flight_price") is not None and flight["price"] > conditions["max_flight_price"]:
            continue
        if conditions.get("max_total_price") is not None and flight["price"] > conditions["max_total_price"]:
            continue
        if conditions.get("max_flight_stops") is not None and flight["stops"] > conditions["max_flight_stops"]:
            continue
        if not passes_time_rules(flight, conditions, travel_date, buffer_minutes):
            continue
        kept.append(flight)
    return kept


def apply_train_conditions(trains, conditions, travel_date):
    """Remove trains that break the user's conditions."""
    buffer_minutes = conditions.get("station_buffer_minutes") or 0
    kept = []
    for train in trains:
        if conditions.get("max_train_price") is not None and train["price"] > conditions["max_train_price"]:
            continue
        if conditions.get("max_total_price") is not None and train["price"] > conditions["max_total_price"]:
            continue
        if conditions.get("max_train_duration_minutes") is not None \
                and train["duration_minutes"] > conditions["max_train_duration_minutes"]:
            continue
        if not passes_time_rules(train, conditions, travel_date, buffer_minutes):
            continue
        kept.append(train)
    return kept


# ===========================================================================
# FLIGHT BRANCH
# ===========================================================================
def search_flights(state: TripState):
    """Call Ignav. If it fails, remember the message and continue with no flights."""
    try:
        raw = search_flights_api(state["airports"]["from"], state["airports"]["to"],
                                 state["travel_date"], state["flight_class"])
        return {"raw_flights": raw}
    except TravelError as err:
        return {"raw_flights": [], "flight_error": str(err)}


def normalize_flights(state: TripState):
    """Turn Ignav itineraries into our simple flight format."""
    flights = []
    for item in state.get("raw_flights", []):
        try:
            segments = item["outbound"]["segments"]
            first, last = segments[0], segments[-1]
            flights.append({
                "type": "flight",
                "airline": item["outbound"].get("carrier", "Unknown"),
                "flight_number": f"{first.get('marketing_carrier_code', '')}{first.get('flight_number', '')}",
                "from": first.get("departure_airport", state["airports"]["from"]),
                "to": last.get("arrival_airport", state["airports"]["to"]),
                "departure": first["departure_time_local"],
                "arrival": last["arrival_time_local"],
                "duration_minutes": int(item["outbound"]["duration_minutes"]),
                "stops": len(segments) - 1,          # 1 segment = non-stop
                # Ignav gives the fare for one traveller, so multiply by passengers
                "price": round(float(item["price"]["amount"]) * state["passengers"]),
                "currency": item["price"].get("currency", "INR"),
                "class": item.get("cabin_class", state["flight_class"]),
            })
        except (KeyError, IndexError, TypeError, ValueError):
            continue   # skip any itinerary with a strange shape
    return {"flights": flights}


def filter_flights(state: TripState):
    """Normal filter first (class, price, stops), then the user's conditions (if any)."""
    normal = [f for f in state["flights"]
              if f["class"] == state["flight_class"]
              and f["price"] > 0
              and f["stops"] <= MAX_STOPS]

    conditions = state.get("parsed_conditions") or {}
    kept = apply_flight_conditions(normal, conditions, state["travel_date"]) if conditions else normal
    return {"flights": kept, "flights_removed_by_conditions": len(normal) - len(kept)}


def lower_is_better(value, lowest, highest):
    """Turn a value into 0-100 where the LOWEST value gets 100 and the HIGHEST gets 0."""
    if highest == lowest:
        return 100.0
    return (highest - value) / (highest - lowest) * 100


def convenience_score(departure_iso):
    """100 if the journey starts between 06:00 and 22:59, otherwise 40 (very early/late)."""
    try:
        hour = datetime.fromisoformat(departure_iso).hour
    except ValueError:
        return 50
    return 100 if 6 <= hour <= 22 else 40


def score_flights(state: TripState):
    """Flight score = 40% price + 35% duration + 15% stops + 10% convenience (each part is 0-100)."""
    flights = state["flights"]
    if not flights:
        return {"flights": []}
    prices = [f["price"] for f in flights]
    times = [f["duration_minutes"] for f in flights]
    for f in flights:
        price_part = lower_is_better(f["price"], min(prices), max(prices))
        time_part = lower_is_better(f["duration_minutes"], min(times), max(times))
        stops_part = {0: 100, 1: 50}.get(f["stops"], 0)     # fewer stops is better
        comfort_part = convenience_score(f["departure"])
        f["score"] = round(0.40 * price_part + 0.35 * time_part
                           + 0.15 * stops_part + 0.10 * comfort_part, 1)
    return {"flights": flights}


def rank_flights(state: TripState):
    """Pick best (highest score), cheapest (lowest price), fastest (shortest duration)."""
    flights = state["flights"]
    if not flights:
        return {"best_flight": None, "cheapest_flight": None, "fastest_flight": None}
    return {
        "best_flight": max(flights, key=lambda f: f["score"]),
        "cheapest_flight": min(flights, key=lambda f: f["price"]),
        "fastest_flight": min(flights, key=lambda f: f["duration_minutes"]),
    }


# ===========================================================================
# TRAIN BRANCH
# ===========================================================================
def search_trains(state: TripState):
    """Call the IRCTC API. If it fails, remember the message and continue with no trains."""
    try:
        raw = search_trains_api(state["stations"]["from"], state["stations"]["to"],
                                state["travel_date"], state["train_class"])
        return {"raw_trains": raw}
    except TravelError as err:
        return {"raw_trains": [], "train_error": str(err)}


def hhmm_to_minutes(text):
    """'26:30' -> 1590 (returns 0 if the text is not in that format)."""
    try:
        hours, minutes = str(text).split(":")[:2]
        return int(hours) * 60 + int(minutes)
    except ValueError:
        return 0


def normalize_trains(state: TripState):
    """Turn IRCTC train dicts into our simple train format."""
    trains = []
    for item in state.get("raw_trains", []):
        try:
            duration = hhmm_to_minutes(item.get("duration"))
            departure = datetime.strptime(f"{state['travel_date']} {item['from_std']}", "%Y-%m-%d %H:%M")
            arrival = departure + timedelta(minutes=duration)   # arrival = departure + duration
            fare = item.get("fare")
            trains.append({
                "type": "train",
                "train_number": str(item.get("train_number", "")),
                "train_name": item.get("train_name", "Unknown"),
                "from": f"{state['origin'].title()} ({state['stations']['from']})",
                "to": f"{state['destination'].title()} ({state['stations']['to']})",
                "departure": departure.isoformat(timespec="minutes"),
                "arrival": arrival.isoformat(timespec="minutes"),
                "duration_minutes": duration,
                "price": round(fare * state["passengers"]) if fare else 0,
                "currency": "INR",
                "class": state["train_class"],
            })
        except (KeyError, TypeError, ValueError):
            continue
    return {"trains": trains}


def filter_trains(state: TripState):
    """Normal filter first (class, fare, duration), then the user's conditions (if any)."""
    normal = [t for t in state["trains"]
              if t["class"] == state["train_class"]
              and t["price"] > 0
              and t["duration_minutes"] > 0]

    conditions = state.get("parsed_conditions") or {}
    kept = apply_train_conditions(normal, conditions, state["travel_date"]) if conditions else normal
    return {"trains": kept, "trains_removed_by_conditions": len(normal) - len(kept)}


def train_type_score(train_name):
    """Faster premium trains score higher: 100 premium, 60 superfast/express, 30 others."""
    name = train_name.lower()
    if any(word in name for word in ["vande", "rajdhani", "shatabdi", "duronto", "tejas"]):
        return 100
    if any(word in name for word in ["sf", "superfast", "express", "exp"]):
        return 60
    return 30


def score_trains(state: TripState):
    """Train score = 45% price + 30% duration + 15% train type + 10% convenience (each part is 0-100)."""
    trains = state["trains"]
    if not trains:
        return {"trains": []}
    prices = [t["price"] for t in trains]
    times = [t["duration_minutes"] for t in trains]
    for t in trains:
        price_part = lower_is_better(t["price"], min(prices), max(prices))
        time_part = lower_is_better(t["duration_minutes"], min(times), max(times))
        type_part = train_type_score(t["train_name"])
        comfort_part = convenience_score(t["departure"])
        t["score"] = round(0.45 * price_part + 0.30 * time_part
                           + 0.15 * type_part + 0.10 * comfort_part, 1)
    return {"trains": trains}


def rank_trains(state: TripState):
    """Pick best, cheapest and fastest train."""
    trains = state["trains"]
    if not trains:
        return {"best_train": None, "cheapest_train": None, "fastest_train": None}
    return {
        "best_train": max(trains, key=lambda t: t["score"]),
        "cheapest_train": min(trains, key=lambda t: t["price"]),
        "fastest_train": min(trains, key=lambda t: t["duration_minutes"]),
    }


# ===========================================================================
# AFTER BOTH BRANCHES FINISH
# ===========================================================================
def compare_options(state: TripState):
    """Compare the best flight with the best train (only if we have both)."""
    flight, train = state.get("best_flight"), state.get("best_train")
    if not flight or not train:
        return {"comparison": None}
    extra_cost = flight["price"] - train["price"]                         # >0 means flight costs more
    time_saved = train["duration_minutes"] - flight["duration_minutes"]   # >0 means flight is faster
    return {"comparison": {
        "flight_price": flight["price"],
        "train_price": train["price"],
        "flight_duration_minutes": flight["duration_minutes"],
        "train_duration_minutes": train["duration_minutes"],
        "extra_cost_of_flight": extra_cost,
        "time_saved_by_flight_minutes": time_saved,
        "summary": (f"Flight: {format_price(flight['price'])} — {format_duration(flight['duration_minutes'])} | "
                    f"Train: {format_price(train['price'])} — {format_duration(train['duration_minutes'])}"),
    }}


def pick_top_3(state: TripState):
    """Choose the 3 best options overall (flights and trains together).

    The flight and train scores earlier were only comparable INSIDE each group, so here
    every option gets one common score (each part 0-100, using ALL options together):
        overall = 50% price + 40% duration + 10% convenience
    Then we take the top 3. If both flights and trains exist, we make sure the best of
    each kind is included, so you can always compare the two.
    """
    options = [dict(o) for o in (state.get("flights") or []) + (state.get("trains") or [])]
    if not options:
        return {"top_options": []}

    prices = [o["price"] for o in options]
    times = [o["duration_minutes"] for o in options]
    for o in options:
        price_part = lower_is_better(o["price"], min(prices), max(prices))
        time_part = lower_is_better(o["duration_minutes"], min(times), max(times))
        o["overall_score"] = round(0.50 * price_part + 0.40 * time_part
                                   + 0.10 * convenience_score(o["departure"]), 1)

    ranked = sorted(options, key=lambda o: o["overall_score"], reverse=True)
    top = ranked[:3]
    for kind in ("flight", "train"):     # keep at least one of each kind when both exist
        if any(o["type"] == kind for o in ranked) and not any(o["type"] == kind for o in top):
            top = top[:2] + [next(o for o in ranked if o["type"] == kind)]
    top.sort(key=lambda o: o["overall_score"], reverse=True)

    for position, o in enumerate(top, start=1):
        number = o["flight_number"] if o["type"] == "flight" else o["train_number"]
        o["rank"] = position
        # a stable id: phase 2 (booking / payment) can use this to say WHICH option was chosen
        o["option_id"] = f"{o['type']}-{number}-{o['departure']}"
        o["tags"] = (["Best overall"] if position == 1 else []) \
            + (["Cheapest"] if o["price"] == min(prices) else []) \
            + (["Fastest"] if o["duration_minutes"] == min(times) else [])
    return {"top_options": top}


def ask_gemini(prompt, json_mode=False):
    """Ask Gemini and return its text. Returns None if there is no key or anything fails.

    json_mode=True asks Gemini to answer with JSON only (used by parse_prompt).
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    if json_mode:
        body["generationConfig"] = {"responseMimeType": "application/json", "temperature": 0}
    # try up to 2 times: Gemini's free plan sometimes says "busy" (503) or "slow down" (429)
    for attempt in range(2):
        try:
            response = httpx.post(url, headers={"x-goog-api-key": api_key}, json=body, timeout=30)
            if response.status_code in (429, 500, 503) and attempt == 0:
                time.sleep(2)
                continue
            response.raise_for_status()
            return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as err:
            # print the REAL reason in the backend terminal so you can see what went wrong
            detail = getattr(getattr(err, "response", None), "text", "")[:300]
            print(f"[Gemini error] {type(err).__name__}: {err} {detail}")
            return None   # the app still works without the AI text
    return None



def simple_recommendation(state: TripState):
    """Plain-Python recommendation, used when Gemini is not available."""
    c = state["comparison"]
    conditions = state.get("parsed_conditions") or {}

    # the user's stated preferences come first
    if conditions.get("prefer_train") and not conditions.get("prefer_flight"):
        return f"You said you prefer the train, so the train is recommended. {c['summary']}"
    if conditions.get("prefer_flight") and not conditions.get("prefer_train"):
        return f"You said you prefer flying, so the flight is recommended. {c['summary']}"
    min_saved = conditions.get("min_time_saved_minutes")
    if min_saved is not None and c["time_saved_by_flight_minutes"] < min_saved:
        return (f"The flight saves only {format_duration(max(c['time_saved_by_flight_minutes'], 0))}, "
                f"less than the {format_duration(int(min_saved))} you asked for, so the train is recommended. {c['summary']}")

    if c["time_saved_by_flight_minutes"] <= 0 and c["extra_cost_of_flight"] >= 0:
        return "The train is both cheaper and not slower here, so the train is the clear choice."
    if c["extra_cost_of_flight"] <= 0:
        return "The flight is cheaper and faster, so the flight is the clear choice."
    hours_saved = c["time_saved_by_flight_minutes"] / 60
    cost_per_hour = c["extra_cost_of_flight"] / hours_saved if hours_saved > 0 else float("inf")
    # Rule of thumb (change as you like): up to ₹600 per hour saved is worth paying
    pick = "flight" if cost_per_hour <= 600 else "train"
    return (f"The flight costs {format_price(c['extra_cost_of_flight'])} more but saves about "
            f"{format_duration(c['time_saved_by_flight_minutes'])}. "
            f"I'd pick the {pick}. If saving money matters most choose the train; "
            f"if saving time matters most choose the flight.")


def ai_recommendation(state: TripState):
    """Write the recommendation text. Gemini only EXPLAINS numbers we already calculated."""
    flight, train, c = state.get("best_flight"), state.get("best_train"), state.get("comparison")

    if not c:   # only one kind of transport found
        if flight:
            return {"recommendation": "Only flights were found for this trip, so flying is your option."}
        if train:
            return {"recommendation": "Only trains were found for this trip, so the train is your option."}
        return {"recommendation": ""}

    # If the user gave conditions, tell Gemini (the options below ALREADY respect them)
    conditions_line = ""
    if state.get("conditions") and state.get("parsed_conditions"):
        conditions_line = (f"The traveller's own conditions (already applied to the options): "
                           f"{state['conditions']}\nParsed: {json.dumps(state['parsed_conditions'])}\n")

    prompt = (
        "You are a travel assistant. Using ONLY the facts below, write 3-4 short sentences "
        "recommending flight or train. Do not invent any numbers, names or availability.\n\n"
        f"Best flight: {flight['airline']} {flight['flight_number']}, {format_price(flight['price'])}, "
        f"{format_duration(flight['duration_minutes'])}, {flight['stops']} stop(s)\n"
        f"Best train: {train['train_name']} ({train['train_number']}), class {train['class']}, "
        f"{format_price(train['price'])}, {format_duration(train['duration_minutes'])}\n"
        f"Extra cost of flight: {format_price(c['extra_cost_of_flight'])}\n"
        f"Time saved by flight: {format_duration(abs(c['time_saved_by_flight_minutes']))}"
        f"{' (train is actually faster)' if c['time_saved_by_flight_minutes'] < 0 else ''}\n"
        f"{conditions_line}"
        "End with one line for budget travellers and one line for people short on time."
    )
    text = ask_gemini(prompt) or simple_recommendation(state)
    return {"recommendation": text}


def missing_message(kind, state):
    """Explain why no flight/train is shown. kind is 'flight' or 'train'."""
    route = f"{state['origin'].title()} → {state['destination'].title()}"
    if state.get(f"{kind}_error"):                       # the API itself failed
        return state[f"{kind}_error"]
    removed = state.get(f"{kind}s_removed_by_conditions", 0)
    if removed:                                          # results existed, conditions removed them all
        return f"{removed} {kind}(s) were found, but none match your travel conditions."
    if kind == "flight":
        return f"No flights found for {route} on this date."
    return f"No trains found for {route} on this date in class {state['train_class']}."


def return_results(state: TripState):
    """Build the final dictionary that FastAPI sends to the frontend."""
    nothing_found = not state.get("best_flight") and not state.get("best_train")
    if nothing_found:
        return {"result": {"error": f"{missing_message('flight', state)} {missing_message('train', state)}"}}

    warnings = []
    if not state.get("best_flight"):
        warnings.append(missing_message("flight", state))
    if not state.get("best_train"):
        warnings.append(missing_message("train", state))

    return {"result": {
        # what we understood from the text (shown on the page, and the input for phase 2 booking)
        "trip": {
            "origin": state["origin"].title(),
            "destination": state["destination"].title(),
            "travel_date": state["travel_date"],
            "passengers": state["passengers"],
            "flight_class": state["flight_class"],
            "train_class": state["train_class"],
        },
        "assumptions": state.get("assumptions", []),          # defaults we used
        "parsed_conditions": state.get("parsed_conditions", {}),   # what Gemini understood
        "top_options": state.get("top_options", []),          # the 3 best options
        "comparison": state.get("comparison"),
        "recommendation": state.get("recommendation", ""),
        # every option that passed filtering, earliest departure first (collapsed table in the UI)
        "flights": sorted(state.get("flights", []), key=lambda f: f["departure"]),
        "trains": sorted(state.get("trains", []), key=lambda t: t["departure"]),
        "warnings": warnings,
    }}


# ---------------------------------------------------------------------------
# Build the graph: add the nodes, then connect them with edges
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(TripState)

    for name, function in [
        ("parse_prompt", parse_prompt),
        ("validate_trip", validate_trip), ("ask_user", ask_user),
        ("location_lookup", location_lookup),
        ("search_flights", search_flights), ("normalize_flights", normalize_flights),
        ("filter_flights", filter_flights), ("score_flights", score_flights), ("rank_flights", rank_flights),
        ("search_trains", search_trains), ("normalize_trains", normalize_trains),
        ("filter_trains", filter_trains), ("score_trains", score_trains), ("rank_trains", rank_trains),
        ("compare_options", compare_options), ("pick_top_3", pick_top_3),
        ("ai_recommendation", ai_recommendation),
        ("return_results", return_results),
    ]:
        graph.add_node(name, function)

    graph.add_edge(START, "parse_prompt")                 # one text box -> Gemini -> JSON + defaults
    graph.add_edge("parse_prompt", "validate_trip")

    # conditional edge: valid -> keep going, invalid -> ask_user
    graph.add_conditional_edges("validate_trip", route_after_validation,
                                {"valid": "location_lookup", "invalid": "ask_user"})
    graph.add_edge("ask_user", END)

    # two edges leaving location_lookup = both branches start at the same time
    graph.add_edge("location_lookup", "search_flights")
    graph.add_edge("location_lookup", "search_trains")

    # flight branch
    graph.add_edge("search_flights", "normalize_flights")
    graph.add_edge("normalize_flights", "filter_flights")
    graph.add_edge("filter_flights", "score_flights")
    graph.add_edge("score_flights", "rank_flights")

    # train branch
    graph.add_edge("search_trains", "normalize_trains")
    graph.add_edge("normalize_trains", "filter_trains")
    graph.add_edge("filter_trains", "score_trains")
    graph.add_edge("score_trains", "rank_trains")

    # a LIST as the start means: wait until BOTH rank nodes are finished
    graph.add_edge(["rank_flights", "rank_trains"], "compare_options")

    graph.add_edge("compare_options", "pick_top_3")
    graph.add_edge("pick_top_3", "ai_recommendation")
    graph.add_edge("ai_recommendation", "return_results")
    graph.add_edge("return_results", END)

    return graph.compile()


travel_graph = build_graph()


def run_search(request_data: dict):
    """Run the graph once and return the final 'result' dictionary."""
    final_state = travel_graph.invoke(request_data)
    return final_state["result"]
