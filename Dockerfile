FROM python:3.11-slim

# Keep Python output unbuffered (useful for logs)
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps for common libs (if needed by unstructured/pdf2image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install first for better caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . /app

# Ensure output dir exists
RUN mkdir -p /app/final_reports

# Default command runs the risk manager; override at runtime if desired
CMD ["python", "risk_manager.py"]
