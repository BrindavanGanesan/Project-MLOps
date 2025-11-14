# Use slim Python base
FROM python:3.10-slim

# Prevent python buffering
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system deps (optional; slim wheels usually ok)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir numpy==1.24.1 scikit-learn==1.2.1 joblib==1.3.2 boto3 fastapi uvicorn[standard]
RUN pip install prometheus-client
# Copy app
COPY app/ ./app/

# Uvicorn web server
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
