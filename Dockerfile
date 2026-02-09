FROM python:3.11-slim

# környezeti változók
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# python deps
COPY requirements.txt .
RUN pip install yfinance

# app code
COPY backend ./backend
COPY frontend ./frontend

# port
EXPOSE 8000

# PROD indítás (NEM reload!)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
