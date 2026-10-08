FROM python:3.11-slim

# Install tesseract-ocr as a system package — guaranteed to work in Docker,
# unlike Render's plain Python runtime where apt-get access isn't guaranteed.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=5000
EXPOSE 5000

CMD ["python", "app.py"]
