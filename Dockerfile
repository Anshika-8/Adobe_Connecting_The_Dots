# Use a slim Python base image compatible with amd64 architecture
FROM --platform=linux/amd64 python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements (to leverage Docker layer caching)
COPY requirements.txt .

# Install system dependencies and Python packages
RUN apt-get update && \
    apt-get install -y build-essential libglib2.0-0 libgl1-mesa-glx poppler-utils && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy your application code
COPY . .

# Ensure input/output directories exist (Docker volumes will mount over them)
RUN mkdir -p /app/input /app/output

# Set the entrypoint to run the PDF processor on startup
CMD ["python", "randomforestclassifier.py"]
