FROM python:3.10-slim

# Work inside SageMaker expected directory
WORKDIR /opt/ml/code

# Install system-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy training requirements
COPY requirements.txt .

# Install Python deps REQUIRED FOR TRAINING ONLY
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
        numpy==1.24.1 \
        pandas==1.5.3 \
        scikit-learn==1.2.1 \
        xgboost==1.7.6 \
        mlflow==2.12.2 \
        boto3 \
        joblib

# Copy training script
COPY train.py .

# Ensure code is available for Python
ENV PYTHONPATH=/opt/ml/code

# SageMaker training entry point
ENTRYPOINT ["python3", "train.py"]
