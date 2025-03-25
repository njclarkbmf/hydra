FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p /data/lancedb /data/models /data/temp

# Create a non-root user and switch to it
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app /data
USER appuser

# Set environment variables
ENV PYTHONPATH=/app
ENV LANCEDB_PATH=/data/lancedb
ENV MODELS_DIR=/data/models
ENV TEMP_DIR=/data/temp
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Expose the application port
EXPOSE 8000

# Start the application
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
