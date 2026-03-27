# Deployment Guide

This project is split into:
- `frontend`: React + Vite (deploy to Vercel)
- `backend`: FastAPI (deploy to Render)

## 1) Deploy backend on Render

1. Go to [Render Dashboard](https://dashboard.render.com/).
2. Click **New +** -> **Blueprint**.
3. Connect GitHub repo: `Thingjae98/calculatechart`.
4. Render detects `render.yaml` and creates service:
   - `calculatechart-api`
5. Deploy, then copy backend URL, e.g.:
   - `https://calculatechart-api.onrender.com`

Health check:
- `GET https://<your-backend-url>/`

## 2) Deploy frontend on Vercel

1. Go to [Vercel Dashboard](https://vercel.com/dashboard).
2. Import GitHub repo: `Thingjae98/calculatechart`.
3. Set **Root Directory** to `frontend`.
4. Build settings:
   - Framework: Vite
   - Build command: `npm run build`
   - Output directory: `dist`
5. Add env var in Vercel Project Settings:
   - `VITE_API_BASE_URL = https://<your-backend-url>`
6. Redeploy.

## 3) CORS

Current backend CORS is open (`*`) for development convenience.
For production, set explicit frontend domains in backend CORS settings.
