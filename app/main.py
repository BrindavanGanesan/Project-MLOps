# app/main.py
import os, io, tarfile, tempfile, json, joblib, boto3
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Iris Classifier", version="1.0")

MODEL_S3_URI = os.getenv("MODEL_S3_URI")         # s3://bucket/path/model.tar.gz
AWS_REGION = os.getenv("AWS_REGION", "eu-west-1")
PUSHGATEWAY_URL = "http://108.130.158.94:9091"

_model = None

def _parse_s3(uri: str):
    assert uri.startswith("s3://")
    b, k = uri[5:].split("/", 1)
    return b, k

def _load_model_from_s3():
    """Downloads model.tar.gz or model.joblib from S3."""
    s3 = boto3.client("s3", region_name=AWS_REGION)
    bucket, key = _parse_s3(MODEL_S3_URI)

    obj = s3.get_object(Bucket=bucket, Key=key)
    body = io.BytesIO(obj["Body"].read())

    if key.endswith(".tar.gz"):
        with tarfile.open(fileobj=body, mode="r:gz") as tar:
            with tempfile.TemporaryDirectory() as td:
                tar.extractall(td)
                mp = os.path.join(td, "model.joblib")

                if not os.path.exists(mp):
                    cand = [m for m in tar.getnames() if m.endswith(".joblib")]
                    if not cand:
                        raise FileNotFoundError("No .joblib found inside model.tar.gz")
                    body.seek(0)
                    with tarfile.open(fileobj=body, mode="r:gz") as tar2:
                        tar2.extract(cand[0], td)
                    mp = os.path.join(td, cand[0])

                return joblib.load(mp)

    # direct joblib in S3
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        f.write(body.getvalue())
        f.flush()
        return joblib.load(f.name)

class PredictRequest(BaseModel):
    data: List[List[float]]

@app.on_event("startup")
def startup():
    global _model
    assert MODEL_S3_URI, "MODEL_S3_URI env var must be set"
    _model = _load_model_from_s3()

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: PredictRequest):

    data = request.data
    pred = int(_model.predict(data)[0])

    # --- Prometheus export ---
    registry = CollectorRegistry()
    prediction_metric = Gauge(
        "iris_prediction_count_total",
        "Total predictions made by class",
        ["class_label"],
        registry=registry
    )

    prediction_metric.labels(class_label=str(pred)).inc()
    push_to_gateway(PUSHGATEWAY_URL, job="iris_api", registry=registry)

    return {"predictions": [pred]}
