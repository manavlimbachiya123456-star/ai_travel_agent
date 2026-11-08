"""Small shared definitions: the request body and one custom error."""
from pydantic import BaseModel


class SearchRequest(BaseModel):
    """The JSON body that the React page sends to POST /api/search.

    There is only ONE field now: the user's request written in plain English.
    Example: "Ahmedabad to Bangalore on 19 Nov, 2 passengers, reach before 1 PM, flight under 9000"
    """
    prompt: str


class TravelError(Exception):
    """Raised by the API files with a friendly message (no stack traces for the user)."""
