import axios from "axios";

const api = axios.create({
  baseURL: "http://127.0.0.1:8000",
});

export const backend = {
  models: () => api.get("/models"),

  processRepo: (payload) => api.post("/process-repo", payload),

  askAI: (payload) => api.post("/ask-ai", payload),

  cleanup: () => api.post("/cleanup"),
};

export default backend;
