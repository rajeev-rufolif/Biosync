FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .
COPY frontend/ /frontend
RUN useradd -m app && chown -R app /app /frontend
USER app

ENV PORT=5000 APP_ENV=production
EXPOSE 5000

# ONE worker: sessions live in this process's memory. Threads give concurrency.
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT} --workers 1 --threads 8 --timeout 300"]
