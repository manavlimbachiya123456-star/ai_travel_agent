import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The proxy sends any request starting with /api to the FastAPI server,
// so the frontend can just call "/api/search" without CORS trouble.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
