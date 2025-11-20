FROM python:3.10-slim

# ----------------------------
# System dependencies
# ----------------------------
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------
# Install Python dependencies
# ----------------------------

COPY requirements.txt /app/requirements.txt

# Core ML + API + drift dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
        numpy \
        pandas \
        scikit-learn==1.2.1 \
        xgboost \
        joblib \
        prometheus_client

# ----------------------------
# Copy training baseline CSV for PSI
# ----------------------------
# IMPORTANT: Make sure adult.csv exists in your repo under data/
COPY data/adult.csv /app/data/adult.csv

# ----------------------------
# Copy application code
# ----------------------------
COPY app /app/app

# ----------------------------
# Expose API + metrics ports
# ----------------------------
EXPOSE 8080
EXPOSE 9090

# ----------------------------
# Run FastAPI via Uvicorn
# ----------------------------
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
