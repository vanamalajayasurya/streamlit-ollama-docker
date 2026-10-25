# Recruitment & Resume Screening Assistant

Streamlit + Ollama + SQLite, packaged with Docker. Screens resume PDFs against a job description,
scores each candidate, extracts skills, and writes interview questions.

## Run locally (docker compose)
```bash
docker compose up --build
```
Open http://localhost:8501. The first start downloads the model (~1.3 GB).
Change model: `OLLAMA_MODEL=qwen2.5:3b docker compose up --build`

## Deploy on Render
1. Push this folder to GitHub.
2. Render > New > **Web Service** > pick the repo. Runtime: **Docker** (it uses the Dockerfile).
3. Instance type: **Standard (2 GB RAM) or higher**. The free and 512 MB plans cannot run the model.
4. Add a **Disk**: mount path `/data`, size 5 GB or more. It stores the SQLite database and the downloaded model.
5. Environment variables (optional): `OLLAMA_MODEL` (default `llama3.2:1b`).
6. Deploy. On first start the model downloads in the background; the app shows a notice until it is ready.

Ollama runs inside the same container; `start.sh` launches it, then Streamlit on Render's `$PORT`.

## Data
SQLite file at `DB_PATH` (default `/data/recruit.db`). Tables: `jobs`, `candidates`.
