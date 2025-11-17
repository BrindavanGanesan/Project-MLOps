FROM python:3.10-slim

RUN mkdir -p /opt/ml/code
WORKDIR /opt/ml/code

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir joblib

# Install ML training dependencies
RUN pip install --no-cache-dir \
    xgboost==1.7.6 \
    pandas==1.5.3 \
    numpy==1.24.1 \
    scikit-learn==1.2.1 \
    mlflow \
    boto3 \
    pyyaml

# Copy training script
COPY docker/training/train.py /opt/ml/code/train.py

# Required by SageMaker: `/opt/ml/code/train` must exist
RUN ln -s /opt/ml/code/train.py /opt/ml/code/train
RUN chmod +x /opt/ml/code/train.py

# DO NOT USE CMD for BYOC – SageMaker ignores it
ENTRYPOINT ["python3", "/opt/ml/code/train.py"]
