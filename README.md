# DocAnalyzer AI

Single-server PDF question answering app. FastAPI serves the UI and API from one process; Chroma keeps the currently uploaded PDF in memory for that process.

## Run locally

1. Use Python 3.11.0.
2. Install dependencies: `python -m pip install -r requirements.txt`
3. Start the server from this directory:

   ```powershell
   uvicorn main:app --host 0.0.0.0 --port 10000 --workers 1
   ```

The local `.env` file supplies `COHERE_API_KEY` and `GOOGLE_API_KEY`. It is excluded from Git. Process environment variables override values from `.env`.

## Deploy to Render

- Create a **Python 3 web service** with this directory as the repository root.
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port 10000 --workers 1`
- Add these environment variables in the Render service dashboard:
  - `PYTHON_VERSION` = `3.11.0`
  - `COHERE_API_KEY` = your Cohere API key
  - `GOOGLE_API_KEY` = your Google AI API key

Keep API keys in Render's environment settings or a local ignored `.env` file; do not commit them. The supplied DeepSeek, Hugging Face, and Groq credentials are not used by this app's requested architecture.

## Behavior

- Upload accepts selectable-text PDFs and reports when OCR is needed.
- Each successful upload replaces the current in-memory document.
- Search uses Cohere embeddings and answers use Gemini 2.5 Flash. The model receives retrieved PDF excerpts as reference text and is instructed not to treat excerpt text as instructions.
- The one-worker start command is required because the in-memory vector store is process-local.
