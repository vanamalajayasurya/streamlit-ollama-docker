#!/bin/sh
# Starts Ollama inside the container (Render) unless OLLAMA_HOST points somewhere else (docker compose),
# then starts the Streamlit app on $PORT (Render sets it; defaults to 8501).
set -e

PORT="${PORT:-8501}"
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
MODEL="${OLLAMA_MODEL:-llama3.2:1b}"
export OLLAMA_HOST

case "$OLLAMA_HOST" in
  *localhost*|*127.0.0.1*)
    # keep downloaded models on the persistent disk so they survive restarts
    export OLLAMA_MODELS="${OLLAMA_MODELS:-/data/ollama}"
    mkdir -p "$OLLAMA_MODELS"
    ollama serve > /tmp/ollama.log 2>&1 &
    (
      for i in $(seq 1 60); do ollama list > /dev/null 2>&1 && break; sleep 1; done
      ollama list | grep -q "^$MODEL" || ollama pull "$MODEL"
    ) &
    ;;
esac

exec streamlit run app.py \
  --server.port="$PORT" \
  --server.address=0.0.0.0 \
  --server.headless=true
