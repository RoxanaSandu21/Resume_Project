# Resume-Job Compatibility Scoring and Bias Evaluation

A local research prototype built with FastAPI, Streamlit, PostgreSQL, Ollama, and Sentence Transformers.

## 1. Prerequisites

- Python 3.12
- PostgreSQL running locally, with an empty database created for this project
- Ollama installed and running
- The local model downloaded:
  ```powershell
  ollama pull qwen2.5:7b-instruct
  ```

The first request can be slow while the local model and embedding model load. Resume processing is synchronous in this prototype.

## 2. Configure the backend

Open PowerShell in the project folder:

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `backend/.env` and set `DATABASE_URL` to your local PostgreSQL database. Do not commit `.env` or place real credentials in Git.

Start the backend from the `backend` directory, with its virtual environment activated:

```powershell
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Keep this terminal open. Check the API in a browser at:

- http://127.0.0.1:8000/health/
- http://127.0.0.1:8000/docs

The health endpoint confirms FastAPI is running. It does not prove PostgreSQL or Ollama are ready.

## 3. Start the Streamlit frontend

Open a second PowerShell terminal:

```powershell
cd frontend
..\backend\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Keep both the backend and frontend terminals open. Open the Streamlit URL printed in the second terminal (usually http://localhost:8501).

## 4. Troubleshooting

- **Connection refused at 127.0.0.1:8000:** the backend is not running or failed during startup. Start it in the first terminal and read its error output.
- **Database connection error:** check PostgreSQL, the database name, and `DATABASE_URL` in `backend/.env`.
- **Ollama connection/model error:** start Ollama and run `ollama list` to confirm `qwen2.5:7b-instruct` is installed.
- **Resume processing is slow:** first-time model downloads/loading and CPU inference can take several minutes. Watch the backend terminal; it logs the current processing stage.
- **Scanned PDF:** PyMuPDF cannot extract selectable text from image-only PDFs. OCR must be added for those documents.

## 5. Research limitations

The compatibility score is an experimental, weighted assessment of extracted job requirements against resume evidence. It is not a calibrated probability of hiring success. Evaluate it against human-labeled resume/job pairs. The current bias module only flags a small number of textual proxy indicators; it is not a complete fairness audit.
