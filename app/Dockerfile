FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1

# Install curl for healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements FIRST so it can be cached
COPY app/requirements-app.txt /app/requirements-app.txt

# Install dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements-app.txt

# Copy the ENTIRE app folder
COPY app/ /app/app

