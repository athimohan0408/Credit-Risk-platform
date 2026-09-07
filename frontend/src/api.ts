import axios from 'axios';

// In Docker, VITE_API_URL is set to relative '/api' via build arg.
// In local dev, it defaults to the FastAPI dev server.
const baseURL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

const api = axios.create({ baseURL });

export default api;
