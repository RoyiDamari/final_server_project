FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1

# Create working directory
WORKDIR /ui_root

# Copy requirements FIRST so it can be cached
COPY ui/requirements-ui.txt /ui_root/requirements-ui.txt

# Install dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements-ui.txt

# Copy the ENTIRE app folder
COPY ui/ /ui_root/ui


