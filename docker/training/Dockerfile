FROM python:3.10-slim

# Required folders for SageMaker
RUN mkdir -p /opt/ml/code
WORKDIR /opt/ml/code

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir xgboost==1.7.6 pandas==1.5.3 scikit-learn==1.2.1 numpy==1.24.1 mlflow boto3 pyyaml

# Copy training code
COPY train.py .

# SageMaker requirement:
# This tells SageMaker which script is the entry script
ENV SAGEMAKER_PROGRAM=train.py

# SageMaker runs train.py automatically
CMD ["train.py"]
