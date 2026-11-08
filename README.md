# India Travel Search Agent

You type ONE request in plain English. Gemini turns it into JSON, the app searches flights (Ignav) and trains
(IRCTC on RapidAPI), applies your conditions, and suggests the **3 best options**.
Stack: LangGraph + FastAPI backend, React/Vite frontend. No database, auth, booking or payment yet.

## Install

```bash
# Backend (Python 3.10+)
cd backend
python -m venv venv
venv\Scripts\activate             # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env            # Mac/Linux: cp .env.example .env   (then paste your API keys)

# Frontend (Node 18+)
cd ../frontend
npm install
```

## Run

```bash
# Terminal 1 - backend (http://localhost:8000)
cd backend
uvicorn main:app --reload

# Terminal 2 - frontend (http://localhost:5173)
cd frontend
npm run dev
```

## API keys (backend/.env)

| Key | Where | Needed? |
|---|---|---|
| `GEMINI_API_KEY` | https://aistudio.google.com/apikey | **Yes** - it reads your text |
| `IGNAV_API_KEY` | https://ignav.com | Yes, for flights |
| `RAPIDAPI_KEY` | RapidAPI -> "IRCTC" (irctc1) | Yes, for trains (small free quota) |

## Example API request

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Ahmedabad to Bangalore on 19 Nov, 2 passengers. Reach before 1 PM, cannot leave before 9 AM, need 1 hour to reach the airport, flight under 18000 total."}'
```

The request has ONE field: `prompt`.

Response keys: `trip` (what was understood), `assumptions` (defaults used), `parsed_conditions`,
`top_options` (the 3 best, each with `rank`, `tags`, `option_id`, `overall_score`), `comparison`,
`recommendation`, `flights` / `trains` (every option that passed the filters), `warnings`.
Errors come back as `{"detail": "friendly message"}`.

## What you can write

Anything about the trip. Gemini fills this JSON, and **anything you don't mention gets a default**:

| Field | If missing |
|---|---|
| from / to city | **No default** - you get a message asking for them |
| travel date | tomorrow |
| passengers | 1 |
| flight class | economy |
| train class | 3A |

Conditions (all optional; they filter the real results):
`arrival_before`, `arrival_after`, `departure_after`, `departure_before` (HH:MM), `airport_buffer_minutes`,
`station_buffer_minutes`, `max_flight_price`, `max_train_price`, `max_total_price`, `max_flight_stops`
(0 = non-stop), `max_train_duration_minutes`, `prefer_flight`, `prefer_train`, `min_time_saved_minutes`
(the last three only change the recommendation text).

Rules: times are compared on the travel date; prices are totals for all passengers; the earliest valid
departure = `departure_after` + buffer.

## The 3 suggestions (`pick_top_3`)

All flights and trains that passed the filters get one common score:
`50% price + 40% duration + 10% convenience` (each part 0-100). The top 3 are returned, and if both flights
and trains exist, the best of each kind is always included so you can compare. Each option has a stable
`option_id` (e.g. `flight-QP1941-2026-11-19T07:50:00`) that phase 2 (booking / payment) can use.

## The graph, node by node (all in `backend/agent.py`)

| Node | What it does |
|---|---|
| `parse_prompt` | Gemini converts your text into trip fields + conditions JSON (checked and cleaned in Python). Fills defaults and records them in `assumptions`. If Gemini fails -> clear error. |
| `validate_trip` | Checks cities, date, passengers, classes. Puts a friendly message in `error` if wrong. |
| `ask_user` | Invalid path: returns the error, then END. |
| `location_lookup` | City -> airport code (flights) and station code (trains), kept separate. |
| `search_flights` / `search_trains` | Call Ignav / IRCTC. A failure is stored as a message so the other branch continues. |
| `normalize_*` | Convert each API's JSON into our simple dicts. |
| `filter_*` | Normal filter, then your conditions (plain Python, no AI). |
| `score_*` | 0-100 score inside each group (flights: 40/35/15/10, trains: 45/30/15/10). |
| `rank_*` | Best, cheapest, fastest of each group (used for the comparison and recommendation). |
| `compare_options` | Best flight vs best train: extra cost and time saved. |
| `pick_top_3` | Common score for all options, returns the 3 best. |
| `ai_recommendation` | Gemini explains the already-calculated numbers (falls back to a plain-Python text). |
| `return_results` | Builds the final JSON. |

**Edges:** START -> parse_prompt -> validate_trip. A conditional edge goes to `location_lookup` (valid) or
`ask_user` (invalid). Two edges leave `location_lookup`, so flights and trains run side by side.
`["rank_flights", "rank_trains"] -> compare_options` waits for both, then
compare_options -> pick_top_3 -> ai_recommendation -> return_results -> END.
