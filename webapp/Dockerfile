FROM python:3.12-slim

# Install git (needed for cloning repos)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    fastapi uvicorn[standard] httpx

# Copy application code
COPY agent/ ./agent/
COPY webapp/ ./webapp/
COPY evals/ ./evals/

# Create non-root user for security
RUN useradd -m -u 1000 sentinel && chown -R sentinel /app

USER sentinel

# Render / Railway / Fly.io all set PORT env var
CMD uvicorn webapp.app:app --host 0.0.0.0 --port ${PORT:-8000}
