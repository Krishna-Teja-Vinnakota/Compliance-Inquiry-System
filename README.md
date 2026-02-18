# Compliance Chatbot

## Overview
Compliance Chatbot is an AI-powered web application for querying Compliance-related documents through a conversational interface. The app authenticates users, loads document data from MongoDB, builds a hybrid retrieval layer using Qdrant + BM25, and generates responses with Google Vertex AI Gemini.

This repository contains the Flask server, web UI templates, static assets, retrieval pipeline, and deployment config for Google App Engine.

## Technologies
- Backend / Web Framework: Flask
- AI / LLM: Google Vertex AI (`ChatVertexAI`, Gemini `gemini-2.0-flash`)
- Embeddings: Vertex AI `text-embedding-004`
- Vector Database: Qdrant Cloud (`qdrant-client`, LangChain Qdrant integration)
- Hybrid Retrieval: BM25 + vector similarity via LangChain `EnsembleRetriever`
- Data Source: MongoDB (`pymongo`) for document records
- PDF Processing: `PyPDF2`
- HTML Parsing / URL Content Extraction: `requests`, `beautifulsoup4`
- Deployment: Google App Engine (`app.yaml`, `gunicorn`)

## Prerequisites
- Python 3.11+
- `pip`
- MongoDB connection with a `pdf_database.pdf_documents` collection
- Qdrant instance and API key
- Google Cloud project with Vertex AI enabled
- Service account credentials for Vertex AI

## Project Structure
```text
Compliance Chatbot/
|-- app/
|   |-- static/
|   `-- templates/
|-- Data/
|-- main.py
|-- requirements.txt
`-- app.yaml
```

## Installation
1. Clone the repository and move into the project directory.
2. Create and activate a virtual environment.
3. Install dependencies.

```bash
git clone <your-repository-url>
cd "Compliance Chatbot"

# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux / macOS
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Environment Configuration
Create a `.env` file in the project root and add the required values:

```env
SECRET_KEY=your_flask_secret_key
MONGO_URI=your_mongodb_connection_string
QDRANT_URL=https://your-qdrant-instance-url
QDRANT_API_KEY=your_qdrant_api_key

# Option 1: local credential file path
GOOGLE_APPLICATION_CREDENTIALS=C:/path/to/service-account.json

# Option 2: raw service account JSON (stringified)
GOOGLE_APPLICATION_CREDENTIALS_JSON={...json...}
```

## Run Locally
```bash
python main.py
```

The app starts on:
- `http://localhost:8081`

## Default Login (Current Implementation)
The current app uses hardcoded login credentials:
- Email: `user@example.com`
- Password: `1234`

## How It Works
1. User logs in through the Flask UI.
2. App loads document records from MongoDB (`pdf_documents`).
3. Text is chunked and indexed in a Qdrant collection.
4. User questions are processed through hybrid retrieval (BM25 + vector search).
5. Gemini generates final answers from retrieved context.

## Main Routes
- `GET/POST /` - login page
- `GET/POST /prompt` - chat interface
- `POST /clear-session` - clear conversation but keep login
- `GET/POST /signup` - signup page (UI flow only)
- `GET/POST /forgot-password` - forgot password page (UI flow only)
- `GET /logout` - logout

## Deployment (Google App Engine)
`app.yaml` is included for App Engine deployment.

```bash
gcloud app deploy
```

The entrypoint is configured as:
- `gunicorn -b :$PORT main:app`

## Notes
- The app is session-based and does not persist chat history beyond session storage.
- Signup and forgot-password endpoints are placeholders for future production auth logic.
- Ensure Qdrant and MongoDB are reachable from your runtime environment.
