FROM python:3.10-slim

# Set workdir
WORKDIR /app

# Install system libs required by XGBoost + pandas
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies
COPY requirements.txt /app/requirements.txt

# Install API + Model dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
        pandas \
        scikit-learn \
        xgboost \
        joblib \
        prometheus_client

# Copy application code
COPY app /app

# Expose app + metrics ports
EXPOSE 8080
EXPOSE 9090

# Run FastAPI with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
