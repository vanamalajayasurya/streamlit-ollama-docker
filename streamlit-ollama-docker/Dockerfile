FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV DB_PATH=/data/recruit.db

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       curl \
       ca-certificates \
       zstd \
    && curl -fsSL https://ollama.com/install.sh | sh \
    && apt-get purge -y curl \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY start.sh .

RUN sed -i 's/\r$//' start.sh \
    && chmod +x start.sh \
    && mkdir -p /data

EXPOSE 10000

CMD ["./start.sh"]