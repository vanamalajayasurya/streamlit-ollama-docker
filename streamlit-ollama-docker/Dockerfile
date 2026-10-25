FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# Ollama inside the image, so a single container works on Render.
# (With docker compose, OLLAMA_HOST points to the ollama service and this copy stays unused.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates zstd \
    && curl -fsSL https://ollama.com/install.sh | sh \
    && apt-get purge -y curl && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py start.sh ./
RUN sed -i 's/\r$//' start.sh && chmod +x start.sh && mkdir -p /data

# SQLite database lives in /data (mount a persistent disk here on Render)
ENV DB_PATH=/data/recruit.db
EXPOSE 8501
HEALTHCHECK CMD python -c "import os,urllib.request; urllib.request.urlopen('http://localhost:%s/_stcore/health' % os.environ.get('PORT','8501'))" || exit 1

CMD ["./start.sh"]
