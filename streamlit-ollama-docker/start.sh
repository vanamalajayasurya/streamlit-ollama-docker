#!/bin/sh
# Starts Ollama inside the container (Render) unless OLLAMA_HOST
# points somewhere else (docker compose), then starts Streamlit.

set -e

PORT="${PORT:-8501}"
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
MODEL="${OLLAMA_MODEL:-llama3.2:1b}"

export OLLAMA_HOST

case "$OLLAMA_HOST" in
    http://localhost:*|http://127.0.0.1:*)
        # Keep downloaded models on the persistent disk
        export OLLAMA_MODELS="${OLLAMA_MODELS:-/data/ollama}"
        mkdir -p "$OLLAMA_MODELS"

        echo "Starting Ollama..."
        ollama serve > /tmp/ollama.log 2>&1 &

        echo "Waiting for Ollama..."

        for i in $(seq 1 60); do
            if ollama list > /dev/null 2>&1; then
                echo "Ollama is ready."
                break
            fi

            sleep 1
        done

        echo "Checking model: $MODEL"

        if ! ollama list | grep -q "^$MODEL"; then
            echo "Downloading model: $MODEL"
            ollama pull "$MODEL"
        else
            echo "Model already exists: $MODEL"
        fi
        ;;
esac

echo "Starting Streamlit on port $PORT..."

exec streamlit run app.py \
    --server.port="$PORT" \
    --server.address=0.0.0.0 \
    --server.headless=true