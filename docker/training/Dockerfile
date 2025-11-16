FROM python:3.10-slim

RUN mkdir -p /opt/ml/code
WORKDIR /opt/ml/code

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy training script
COPY train.py /opt/ml/code/train.py

# SageMaker requirement — executable entry script without extension
RUN cp /opt/ml/code/train.py /opt/ml/code/train && chmod +x /opt/ml/code/train

ENTRYPOINT ["python3", "/opt/ml/code/train"]
