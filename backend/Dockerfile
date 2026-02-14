# Backend Dockerfile
FROM python:3.12-slim

# Set working directory
WORKDIR /project

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY backend/requirements.txt /project/backend/

# Install Python dependencies
RUN pip install --no-cache-dir -r /project/backend/requirements.txt

# Copy application code
COPY . /project

# Create data and docs directories
RUN mkdir -p /project/backend/data/chroma /project/docs

# Create non-root user
RUN adduser --disabled-password --no-create-home appuser
USER appuser

# Expose port
EXPOSE 8000

# Run the application (no --reload in production)
CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
