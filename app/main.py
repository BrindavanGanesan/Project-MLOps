# app/main.py
import os
import io
import tarfile
import tempfile
import json

import boto3
import joblib
import pandas as pd
from fastapi import FastAPI, Response
from pydantic import BaseModel
from typing import List

# Prometheus
from prometheus_client import (
    CollectorRegistry,
    Gauge,
    Counter,
    generate_latest,
    CONTENT_TYPE_LATEST,
    push_to_gateway,
)

# ------------------ FastAPI App ------------------
app = FastAPI(title="Adult Income Classifier", version="2.0")

# ------------------ ENV VARS ---------------------
MODEL_S3_URI = os.getenv("MODEL_S3_URI")   # Required
AWS_REGION = os.getenv("AWS_REGION", "eu-west-1")

# Default PushGateway (your IP)
PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL", "http://108.130.158.94:9091")

# ------------------ Globals ----------------------
_model = None


# -------------------------------------------------
# S3 Utilities
# -------------------------------------------------
def _parse_s3(uri: str):
    assert uri.startswith("s3://"), f"Invalid S3 URI: {uri}"
    bucket, key = uri[5:].split("/", 1)
    return bucket, key


def _load_model_from_s3():
    """
    Loads model.tar.gz (SageMaker format) OR model.joblib directly.
    """
    print(f"📦 Loading model from S3: {MODEL_S3_URI}")
    s3 = boto3.client("s3", region_name=AWS_REGION)
    bucket, key = _parse_s3(MODEL_S3_URI)

    obj = s3.get_object(Bucket=bucket, Key=key)
    body = io.BytesIO(obj["Body"].read())

    # Case 1 — SageMaker model.tar.gz
    if key.endswith(".tar.gz"):
        with tarfile.open(fileobj=body, mode="r:gz") as tar:
            with tempfile.TemporaryDirectory() as td:
                tar.extractall(td)

                # Find model.joblib inside the tar
                joblib_path = os.path.join(td, "model.joblib")

                if not os.path.exists(joblib_path):
                    # fallback: search for any .joblib
                    for member in tar.getmembers():
                        if member.name.endswith(".joblib"):
                            tar.extract(member, td)
                            joblib_path = os.path.join(td, member.name)
                            break

                if not os.path.exists(joblib_path):
                    raise FileNotFoundError("❌ No *.joblib file found inside model.tar.gz")

                print("✅ Loaded model.joblib from tar archive")
                return joblib.load(joblib_path)

    # Case 2 — Direct S3 joblib
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        f.write(body.getvalue())
        f.flush()
        print("✅ Loaded direct joblib file from S3")
        return joblib.load(f.name)


# -------------------------------------------------
# Request Schemas
# -------------------------------------------------
class AdultRecord(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


class PredictRequest(BaseModel):
    records: List[AdultRecord]


# -------------------------------------------------
# Prometheus Metrics
# -------------------------------------------------

# Total predictions counter
PREDICTION_COUNTER = Counter(
    "adult_income_prediction_count",
    "Total number of predictions served"
)

# Data Drift score
DRIFT_SCORE = Gauge(
    "adult_income_drift_score",
    "Drift score between live data and training distribution"
)

# Label distribution
LABEL_COUNTER = Counter(
    "adult_income_prediction_by_label",
    "Prediction count by label",
    ["label"]
)


# -------------------------------------------------
# Dummy Drift Detector (replace later with KS test or PSI)
# -------------------------------------------------
def compute_drift(df: pd.DataFrame) -> float:
    """
    Placeholder drift metric: 
    We compute the average relative difference between numeric cols.
    Replace with real drift detection if needed.
    """
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) == 0:
        return 0.0

    drift_value = float(df[numeric_cols].mean().sum() % 1)  # synthetic
    return drift_value


# -------------------------------------------------
# FastAPI Startup
# -------------------------------------------------
@app.on_event("startup")
def startup():
    global _model

    if not MODEL_S3_URI:
        raise RuntimeError("❌ MODEL_S3_URI environment variable must be set!")

    _model = _load_model_from_s3()
    print("🚀 Model is ready!")


# -------------------------------------------------
# Health Check
# -------------------------------------------------
@app.get("/ping")
def ping():
    return {"status": "ok"}


# -------------------------------------------------
# Metrics Endpoint (Prometheus Pull)
# -------------------------------------------------
@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


# -------------------------------------------------
# Prediction Endpoint
# -------------------------------------------------
@app.post("/predict")
def predict(request: PredictRequest):

    df = pd.DataFrame([r.dict() for r in request.records])

    # Run prediction
    preds = _model.predict(df)
    preds = preds.tolist()

    # ---------------------------------------------
    # Update metrics
    # ---------------------------------------------
    drift_value = compute_drift(df)
    drift_value = round(drift_value, 4)
    DRIFT_SCORE.set(drift_value)

    for p in preds:
        LABEL_COUNTER.labels(label=str(p)).inc()

    PREDICTION_COUNTER.inc(len(preds))

    # ---------------------------------------------
    # Push to PushGateway
    # ---------------------------------------------
    try:
        registry = CollectorRegistry()
        DRIFT_SCORE.registry = registry
        PREDICTION_COUNTER.registry = registry
        LABEL_COUNTER.registry = registry

        push_to_gateway(PUSHGATEWAY_URL, job="adult_income_api", registry=registry)
        print(f"📡 Metrics pushed to {PUSHGATEWAY_URL}")
    except Exception as e:
        print(f"⚠️ Pushgateway push failed: {e}")

    # ---------------------------------------------
    # Return predictions
    # ---------------------------------------------
    return {
        "predictions": preds,
        "drift_score": drift_value
    }
