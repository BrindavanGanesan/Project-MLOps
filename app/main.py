# app/main.py
import os
import io
import tarfile
import tempfile
import json

import boto3
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway


app = FastAPI(title="Adult Income Classifier", version="1.0")

MODEL_S3_URI = os.getenv("MODEL_S3_URI")  # e.g. s3://thebrowntiger/<job>/output/model.tar.gz
AWS_REGION = os.getenv("AWS_REGION", "eu-west-1")
PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL", "http://108.130.158.94:9091")

_model = None


def _parse_s3(uri: str):
    assert uri.startswith("s3://")
    b, k = uri[5:].split("/", 1)
    return b, k


def _load_model_from_s3():
    """
    Downloads model.tar.gz (or model.joblib) from S3 and loads it.
    Compatible with SageMaker training output.
    """
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
                    # search for any *.joblib in the tar content
                    for member in tar.getmembers():
                        if member.name.endswith(".joblib"):
                            tar.extract(member, td)
                            mp = os.path.join(td, member.name)
                            break
                    if not os.path.exists(mp):
                        raise FileNotFoundError("No *.joblib found in model.tar.gz")
                model = joblib.load(mp)
                return model
    else:
        # direct joblib in S3
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            f.write(body.getvalue())
            f.flush()
            return joblib.load(f.name)


# ---- Request schema for Adult Census ----

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


@app.on_event("startup")
def startup():
    global _model
    assert MODEL_S3_URI, "MODEL_S3_URI env var must be set"
    _model = _load_model_from_s3()
    print("✅ Model loaded successfully from S3.")


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest):
    # Convert list of records to DataFrame with canonical column names
    df = pd.DataFrame([r.dict() for r in request.records])

    preds = _model.predict(df)
    preds = [str(p) for p in preds]  # e.g. '<=50K' or '>50K'

    # --- Prometheus metrics: count predictions by label ---
    try:
        registry = CollectorRegistry()
        prediction_metric = Gauge(
            "adult_income_prediction_total",
            "Total number of Adult Income predictions by label",
            ["predicted_label"],
            registry=registry,
        )

        # Increment once per prediction
        for label in preds:
            prediction_metric.labels(predicted_label=label).inc()

        push_to_gateway(PUSHGATEWAY_URL, job="adult_income_api", registry=registry)
    except Exception as e:
        # We don't want metrics issues to break inference
        print(f"⚠️ Failed to push metrics to Pushgateway: {e}")

    return {"predictions": preds}
