# KumoRFM Docker Image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install KumoRFM
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/data /app/checkpoints /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/app/.cache/torch

# Expose port for potential API service
EXPOSE 8000

# Default command
CMD ["python", "examples/demo.py"]