import axios from "axios";

// Sends the form data to the backend and returns the result.
// If the backend sends an error message, we throw it as a normal Error.
export async function searchTrips(formData) {
  try {
    const response = await axios.post("/api/search", formData);
    return response.data;
  } catch (error) {
    const message =
      error.response?.data?.detail || "Could not reach the server. Is the backend running?";
    throw new Error(message);
  }
}
