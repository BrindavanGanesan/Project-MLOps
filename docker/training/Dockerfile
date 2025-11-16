FROM python:3.10-slim

RUN mkdir -p /opt/ml/code
WORKDIR /opt/ml/code

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training script
COPY train.py /opt/ml/code/train.py

# Required by SageMaker: `/opt/ml/code/train` must exist
RUN ln -s /opt/ml/code/train.py /opt/ml/code/train
RUN chmod +x /opt/ml/code/train.py

# DO NOT USE CMD for BYOC – SageMaker ignores it
ENTRYPOINT ["python3", "/opt/ml/code/train"]
