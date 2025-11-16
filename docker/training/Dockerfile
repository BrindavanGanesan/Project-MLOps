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

# Make train.py executable
RUN chmod +x /opt/ml/code/train.py

# SageMaker will call: python train.py <args>
CMD ["python3", "/opt/ml/code/train.py"]
