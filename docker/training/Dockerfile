FROM python:3.10-slim

# Required SageMaker folder
RUN mkdir -p /opt/ml/code
WORKDIR /opt/ml/code

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install ML packages pin-compatible for SageMaker
RUN pip install --no-cache-dir \
    xgboost==1.7.6 \
    pandas==1.5.3 \
    scikit-learn==1.2.1 \
    numpy==1.24.1 \
    mlflow boto3 pyyaml joblib

# Copy training script (from training/train.py → container train.py)
COPY ../../training/train.py /opt/ml/code/train.py

# Create SageMaker launcher
RUN echo '#!/usr/bin/env bash' > /opt/ml/code/train && \
    echo 'python3 /opt/ml/code/train.py "$@"' >> /opt/ml/code/train && \
    chmod +x /opt/ml/code/train

# Do NOT set ENTRYPOINT or CMD — SageMaker invokes /opt/ml/code/train automatically
