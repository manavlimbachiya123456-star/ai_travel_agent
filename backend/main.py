"""FastAPI app: two endpoints, /health and /api/search."""
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()  # reads backend/.env so the API keys are available

from agent import run_search   # noqa: E402  (imported after load_dotenv on purpose)
from schemas import SearchRequest   # noqa: E402

app = FastAPI(title="India Travel Search Agent")

# allow the Vite dev server to call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Quick check that the server is running."""
    return {"status": "ok"}


@app.post("/api/search")
def search(request: SearchRequest):
    """Run the LangGraph agent and return flights, trains and the recommendation."""
    try:
        result = run_search(request.model_dump())
    except Exception:
        # never show a Python stack trace to the user
        raise HTTPException(status_code=500, detail="Something went wrong while searching. Please try again.")

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result
