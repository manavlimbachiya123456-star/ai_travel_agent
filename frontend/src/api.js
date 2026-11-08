import axios from "axios";

// In development this is empty, so "/api/search" goes through the Vite proxy to localhost:8000.
// On Render we set VITE_API_URL to the backend's address (e.g. https://my-backend.onrender.com).
const API_URL = import.meta.env.VITE_API_URL || "";

// Sends the text to the backend and returns the result.
// If the backend sends an error message, we throw it as a normal Error.
export async function searchTrips(formData) {
  try {
    const response = await axios.post(`${API_URL}/api/search`, formData);
    return response.data;
  } catch (error) {
    const message =
      error.response?.data?.detail || "Could not reach the server. Is the backend running?";
    throw new Error(message);
  }
}