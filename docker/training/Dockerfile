FROM python:3.10-slim

# SageMaker-required directory
RUN mkdir -p /opt/ml/code
WORKDIR /opt/ml/code

# OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements (from project root)
COPY docker/training/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# ML compatible versions
RUN pip install --no-cache-dir \
    xgboost==1.7.6 \
    pandas==1.5.3 \
    scikit-learn==1.2.1 \
    numpy==1.24.1 \
    mlflow boto3 pyyaml joblib

# --------------------------------------------------------
# 🔥 Copy training script from correct location
# --------------------------------------------------------
# This path is relative to build context (project root)
COPY docker/training/train.py /opt/ml/code/train.py

# --------------------------------------------------------
# 🔥 Create SageMaker launcher script
# --------------------------------------------------------
RUN echo '#!/usr/bin/env bash' > /opt/ml/code/train && \
    echo 'python3 /opt/ml/code/train.py "$@"' >> /opt/ml/code/train && \
    chmod +x /opt/ml/code/train && \
    chmod +x /opt/ml/code/train.py

# SageMaker automatically calls /opt/ml/code/train
# NO ENTRYPOINT OR CMD
