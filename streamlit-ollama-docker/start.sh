#!/bin/sh
# Starts Ollama inside the container (Render) unless OLLAMA_HOST
# points somewhere else (docker compose), then starts Streamlit.

set -e

PORT="${PORT:-8501}"
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
MODEL="${OLLAMA_MODEL:-qwen2.5:0.5b}"

export OLLAMA_HOST

case "$OLLAMA_HOST" in
    http://localhost:*|http://127.0.0.1:*)
        # Keep downloaded models on the persistent disk
        export OLLAMA_MODELS="${OLLAMA_MODELS:-/data/ollama}"
        mkdir -p "$OLLAMA_MODELS"

        # Keep Ollama's memory use as small as possible so the Render
        # instance is not killed for running out of RAM (which shows up
        # in the browser as "Connection failed with status 502").
        export OLLAMA_NUM_PARALLEL=1
        export OLLAMA_MAX_LOADED_MODELS=1
        export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-2m}"
        export OLLAMA_FLASH_ATTENTION=1

        # Run Ollama under a tiny supervisor: if it ever crashes, restart it
        # instead of leaving the app talking to a dead server.
        (
            while true; do
                ollama serve >> /tmp/ollama.log 2>&1 || true
                echo "Ollama exited, restarting in 2s..." >> /tmp/ollama.log
                sleep 2
            done
        ) &

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
            # Don't let a failed download kill the whole container
            ollama pull "$MODEL" || echo "WARNING: model pull failed"
        else
            echo "Model already exists: $MODEL"
        fi

        # Load the model into memory now, so the first request doesn't
        # have to do it while Streamlit is also busy.
        ollama run "$MODEL" "hi" > /dev/null 2>&1 || true
        ;;
esac

echo "Starting Streamlit on port $PORT..."

exec streamlit run app.py \
    --server.port="$PORT" \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.maxUploadSize=20