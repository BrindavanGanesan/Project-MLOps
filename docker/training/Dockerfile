FROM python:3.10-slim

WORKDIR /opt/ml/code

# Install system deps
RUN apt-get update && apt-get install -y build-essential

# Install training-specific deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training code
COPY train.py .

ENV PYTHONPATH=/opt/ml/code
ENTRYPOINT ["python3", "train.py"]