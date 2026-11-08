import { useState, useRef } from "react";
import { searchTrips } from "./api";
import "./App.css";

// the browser's speech recognizer (works in Chrome and Edge)
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

// 160 -> "2h 40m"
function formatDuration(minutes) {
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
}

// 4850 -> "₹4,850"
function formatPrice(amount) {
  return "₹" + Number(amount).toLocaleString("en-IN");
}

// "2026-11-19T08:30" -> "19 Nov, 08:30"
function formatTime(isoText) {
  const d = new Date(isoText);
  if (isNaN(d)) return isoText;
  return d.toLocaleString("en-IN", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", hour12: false });
}

// One row inside a card: label on the left, value on the right
function Row({ label, value }) {
  return (
    <div className="row">
      <span className="label">{label}</span>
      <span>{value}</span>
    </div>
  );
}

// One of the 3 suggested options (it can be a flight or a train)
function OptionCard({ option }) {
  const isFlight = option.type === "flight";
  return (
    <div className="card option">
      <div className="option-head">
        <h2>#{option.rank} {isFlight ? "✈️ Flight" : "🚆 Train"}</h2>
        <div>{option.tags.map((tag) => <span className="tag" key={tag}>{tag}</span>)}</div>
      </div>

      {isFlight ? (
        <>
          <Row label="Airline" value={option.airline} />
          <Row label="Flight Number" value={option.flight_number} />
        </>
      ) : (
        <>
          <Row label="Train Name" value={option.train_name} />
          <Row label="Train Number" value={option.train_number} />
        </>
      )}
      <Row label="Departure" value={`${option.from} · ${formatTime(option.departure)}`} />
      <Row label="Arrival" value={`${option.to} · ${formatTime(option.arrival)}`} />
      <Row label="Duration" value={formatDuration(option.duration_minutes)} />
      {isFlight && <Row label="Stops" value={option.stops === 0 ? "Non-stop" : option.stops} />}
      <Row label="Price" value={formatPrice(option.price)} />
      <Row label="Class" value={option.class} />
    </div>
  );
}

// Collapsed table with EVERY flight that passed the filters
function FlightTable({ flights }) {
  if (!flights || flights.length === 0) return null;
  return (
    <details className="table-box">
      <summary>✈️ All matching flights ({flights.length})</summary>
      <table>
        <thead>
          <tr><th>Flight</th><th>Departs</th><th>Arrives</th><th>Duration</th><th>Stops</th><th>Price</th></tr>
        </thead>
        <tbody>
          {flights.map((f, i) => (
            <tr key={i}>
              <td>{f.airline} {f.flight_number}</td>
              <td>{formatTime(f.departure)}</td>
              <td>{formatTime(f.arrival)}</td>
              <td>{formatDuration(f.duration_minutes)}</td>
              <td>{f.stops === 0 ? "Non-stop" : f.stops}</td>
              <td>{formatPrice(f.price)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </details>
  );
}

// Same idea for trains
function TrainTable({ trains }) {
  if (!trains || trains.length === 0) return null;
  return (
    <details className="table-box">
      <summary>🚆 All matching trains ({trains.length})</summary>
      <table>
        <thead>
          <tr><th>Train</th><th>Departs</th><th>Arrives</th><th>Duration</th><th>Price</th></tr>
        </thead>
        <tbody>
          {trains.map((t, i) => (
            <tr key={i}>
              <td>{t.train_name} ({t.train_number})</td>
              <td>{formatTime(t.departure)}</td>
              <td>{formatTime(t.arrival)}</td>
              <td>{formatDuration(t.duration_minutes)}</td>
              <td>{formatPrice(t.price)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </details>
  );
}

export default function App() {
  const [prompt, setPrompt] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [listening, setListening] = useState(false);   // true while the microphone is on
  const recognitionRef = useRef(null);                 // remembers the recognizer so we can stop it

  async function handleSubmit(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);
    try {
      setResult(await searchTrips({ prompt: prompt.trim() })); // the ONLY thing we send
    } catch (err) {
      setError(err.message);
    }
    setLoading(false);
  }

  // click once to start speaking, click again to stop
  function toggleMic() {
    if (!SpeechRecognition) {
      setError("Speech recognition is not supported in this browser. Please use Chrome or Edge.");
      return;
    }
    if (listening) {
      recognitionRef.current.stop();
      return;
    }
    const recognition = new SpeechRecognition();
    recognition.lang = "en-IN";            // Indian English
    recognition.interimResults = true;     // show words while you speak
    const oldText = prompt ? prompt + " " : "";   // keep what is already typed

    recognition.onresult = (event) => {
      let spoken = "";
      for (const result of event.results) spoken += result[0].transcript;
      setPrompt(oldText + spoken);
    };
    recognition.onend = () => setListening(false);
    recognition.onerror = (e) => {
      setListening(false);
      setError("Microphone problem: " + e.error);
    };

    setError("");
    recognition.start();
    recognitionRef.current = recognition;
    setListening(true);
  }

  const trip = result?.trip;

  return (
    <div className="page">
      <h1>India Travel Search</h1>
      <p className="subtitle">Describe your trip in your own words. We'll find the 3 best options.</p>

      <form className="form" onSubmit={handleSubmit}>
        <textarea
          rows="5"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder={"e.g. Ahmedabad to Bangalore on 19 November, 2 passengers, economy flight, 3A train.\nI must reach before 1 PM and can't leave before 9 AM. I need 1 hour to reach the airport. Flight under ₹9000."}
        />
        {/* type="button" so clicking the mic does NOT submit the form */}
        <button type="button" className={listening ? "mic on" : "mic"} onClick={toggleMic}>
          {listening ? "⏹ Stop" : "🎤 Speak"}
        </button>
        <button type="submit" disabled={loading || !prompt.trim()}>{loading ? "Searching..." : "Search"}</button>
      </form>

      {error && <div className="error">{error}</div>}

      {result && (
        <>
          {/* what we understood from the text */}
          <p className="understood">
            <b>{trip.origin} → {trip.destination}</b> · {trip.travel_date} · {trip.passengers} passenger(s) ·
            flight: {trip.flight_class} · train: {trip.train_class}
          </p>
          {result.assumptions?.length > 0 && (
            <div className="assumptions">
              <b>Defaults used (you didn't say):</b>
              <ul>{result.assumptions.map((a) => <li key={a}>{a}</li>)}</ul>
            </div>
          )}
          {result.warnings?.map((w) => <div className="warning" key={w}>{w}</div>)}

          <h2 className="section">Top {result.top_options.length} suggestions</h2>
          <div className="grid">
            {result.top_options.map((option) => <OptionCard option={option} key={option.option_id} />)}
          </div>

          {result.comparison && <p className="compare">{result.comparison.summary}</p>}

          {result.recommendation && (
            <div className="card recommendation">
              <h2>🤖 AI RECOMMENDATION</h2>
              <p>{result.recommendation}</p>
            </div>
          )}

          <FlightTable flights={result.flights} />
          <TrainTable trains={result.trains} />

          {/* what Gemini understood as conditions (only the ones you really gave) */}
          {Object.keys(result.parsed_conditions || {}).length > 0 && (
            <details className="debug">
              <summary>Parsed conditions (debug)</summary>
              <pre>{JSON.stringify(result.parsed_conditions, null, 2)}</pre>
            </details>
          )}
        </>
      )}
    </div>
  );
}