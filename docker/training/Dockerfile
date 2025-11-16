FROM python:3.10-slim

# SageMaker training code lives here
RUN mkdir -p /opt/ml/code
WORKDIR /opt/ml/code

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install ML dependencies
RUN pip install --no-cache-dir \
    xgboost==1.7.6 \
    pandas==1.5.3 \
    scikit-learn==1.2.1 \
    numpy==1.24.1 \
    mlflow \
    boto3 \
    pyyaml

# Copy train script
COPY train.py /opt/ml/code/train.py

# Make a copy without extension (SageMaker executes THIS!)
RUN cp /opt/ml/code/train.py /opt/ml/code/train && \
    chmod +x /opt/ml/code/train

# SageMaker will call /opt/ml/code/train automatically
ENTRYPOINT ["python3", "/opt/ml/code/train"]
