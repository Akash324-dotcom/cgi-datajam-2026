# CGI-DATAJAM-2026

Full-stack app with:
- `frontend`: React + Vite UI (`http://localhost:5173`)
- `backend`: FastAPI API (`http://localhost:8000`)
- n8n webhook integration: backend forwards prompt text to your webhook URL

## Current Flow

1. User enters symptom prompt in frontend chat.
2. Frontend sends `POST /api/triage` to backend with:
```json
{ "text": "..." }
```
3. Backend forwards `{ "text": "..." }` to `N8N_WEBHOOK_URL`.
4. Backend returns n8n JSON response to frontend.
5. Frontend displays `reply` from the response.

## Project Structure

```text
CGI-DATAJAM-2026/
├── README.md
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── .envSample
│   ├── data/
│   ├── graphs/
│   ├── scripts/
│   └── services/
└── frontend/
    ├── package.json
    ├── vite.config.js
    ├── index.html
    └── src/
        ├── App.jsx
        ├── main.jsx
        ├── index.css
        ├── components/
        └── services/
```

## Backend Setup

From repo root:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set webhook URL:

```bash
export N8N_WEBHOOK_URL="http://localhost:5678/webhook-test/triage"
```

Or source from your env file:

```bash
source .env
```

Run backend:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Frontend Setup

From repo root:

```bash
cd frontend
npm install
npm run dev
```

Frontend defaults to backend at:
- `http://localhost:8000`

To override, set:

```bash
VITE_API_BASE_URL=http://localhost:8000
```

## n8n Requirements

- Webhook should accept `POST` JSON body containing `text`.
- Webhook should return valid JSON.
- Returned JSON should include `reply` for current chat display behavior.

Example response:

```json
{
  "reply": "debug output from n8n"
}
```

## API Endpoints

- `GET /` -> health message
- `POST /api/triage` -> forwards prompt to n8n and returns n8n JSON response

## Troubleshooting

- `500 N8N_WEBHOOK_URL is not configured`: env var not loaded in the backend shell.
- `502 Bad Gateway`: backend could not reach n8n, n8n returned an HTTP error, or n8n returned non-JSON.
- If using `/webhook-test/...`, make sure n8n is in test listening mode.
