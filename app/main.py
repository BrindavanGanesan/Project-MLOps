# app/main.py

import os
import io
import tarfile
import tempfile
import math
import time
from typing import List, Dict, Any

import boto3
import joblib
import pandas as pd
from fastapi import FastAPI, Response
from pydantic import BaseModel, Field
import numpy as np

from prometheus_client import (
    Gauge,
    Counter,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    pushadd_to_gateway,
)

# ------------------ FastAPI App ------------------
app = FastAPI(title="Adult Income Classifier", version="4.0")

# ------------------ ENV VARS ---------------------
MODEL_S3_URI = os.getenv("MODEL_S3_URI")
AWS_REGION = os.getenv("AWS_REGION", "eu-west-1")
TRAINING_CSV_PATH = os.getenv("TRAINING_CSV_PATH", "/app/data/adult.csv")
PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL", "http://108.130.158.94:9091")

PSI_NUM_BINS = int(os.getenv("PSI_NUM_BINS", "10"))
PSI_EPS = 1e-6

# ------------------ Globals ----------------------
_model = None
_BASELINE_STATS: Dict[str, Dict[str, Any]] = {}

# -------------------------------------------------
# S3 Model Loader
# -------------------------------------------------
def _parse_s3(uri: str):
    assert uri.startswith("s3://"), f"Invalid S3 URI: {uri}"
    bucket, key = uri[5:].split("/", 1)
    return bucket, key


def _load_model_from_s3():
    print(f"📦 Loading model from S3: {MODEL_S3_URI}")
    s3 = boto3.client("s3", region_name=AWS_REGION)
    bucket, key = _parse_s3(MODEL_S3_URI)

    obj = s3.get_object(Bucket=bucket, Key=key)
    body = io.BytesIO(obj["Body"].read())

    if key.endswith(".tar.gz"):
        with tarfile.open(fileobj=body, mode="r:gz") as tar:
            with tempfile.TemporaryDirectory() as td:
                tar.extractall(td)
                joblib_path = os.path.join(td, "model.joblib")

                if not os.path.exists(joblib_path):
                    for member in tar.getmembers():
                        if member.name.endswith(".joblib"):
                            tar.extract(member, td)
                            joblib_path = os.path.join(td, member.name)
                            break

                print("✅ Loaded model.joblib from tar archive")
                return joblib.load(joblib_path)

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        f.write(body.getvalue())
        print("✅ Loaded model.joblib directly")
        return joblib.load(f.name)


# -------------------------------------------------
# Baseline PSI stats
# -------------------------------------------------
def _load_baseline_stats(csv_path: str):
    if not os.path.exists(csv_path):
        print("⚠️ Baseline CSV missing!")
        return {}

    df = pd.read_csv(csv_path)
    print(f"📊 Loaded baseline columns: {df.columns.tolist()}")

    stats = {}

    for col in df.columns:
        if col.lower() in {"income", "target"}:
            continue

        s = df[col].dropna()

        if pd.api.types.is_numeric_dtype(s):
            qs = [i / PSI_NUM_BINS for i in range(PSI_NUM_BINS + 1)]
            edges = sorted(set(float(x) for x in s.quantile(qs).values))
            if len(edges) < 2:
                continue

            ref_counts, _ = np.histogram(s, bins=edges)
            total = ref_counts.sum()
            ref_props = (ref_counts / total).tolist()

            stats[col] = {"type": "numeric", "bins": edges, "ref": ref_props}

        else:
            freq = s.value_counts(normalize=True)
            stats[col] = {"type": "categorical", "ref": freq.to_dict()}

    print(f"✅ PSI baseline built for {len(stats)} features")
    return stats


# -------------------------------------------------
# PSI Computation
# -------------------------------------------------
def _psi_numeric(ref_props, cur_props):
    psi = 0.0
    for pr, pc in zip(ref_props, cur_props):
        pr = max(float(pr), PSI_EPS)
        pc = max(float(pc), PSI_EPS)
        psi += (pr - pc) * math.log(pr / pc)
    return psi


def _psi_categorical(ref_dict, cur_dict):
    psi = 0.0
    cats = set(ref_dict.keys()) | set(cur_dict.keys())
    for c in cats:
        pr = max(ref_dict.get(c, 0.0), PSI_EPS)
        pc = max(cur_dict.get(c, 0.0), PSI_EPS)
        psi += (pr - pc) * math.log(pr / pc)
    return psi


def compute_drift(df: pd.DataFrame):
    if not _BASELINE_STATS:
        return 0.0, {}

    psi_by_feature = {}

    for col, meta in _BASELINE_STATS.items():
        if col not in df.columns:
            continue

        s = df[col].dropna()
        if s.empty:
            continue

        if meta["type"] == "numeric":
            edges = meta["bins"]
            ref_props = meta["ref"]

            cur_counts, _ = np.histogram(s, bins=edges)
            cur_props = (cur_counts / cur_counts.sum()).tolist()

            n = min(len(ref_props), len(cur_props))
            psi = _psi_numeric(ref_props[:n], cur_props[:n])

        else:
            freq = s.value_counts(normalize=True)
            psi = _psi_categorical(meta["ref"], freq.to_dict())

        psi_by_feature[col] = float(psi)

    global_psi = max(psi_by_feature.values()) if psi_by_feature else 0.0
    return global_psi, psi_by_feature


# -------------------------------------------------
# Request Schemas
# -------------------------------------------------
class AdultRecord(BaseModel):
    age: int = Field(alias="Age")
    workclass: str = Field(alias="Workclass")
    fnlwgt: int = Field(alias="fnlwgt")
    education: str = Field(alias="Education")
    education_num: int = Field(alias="Education_Num")
    marital_status: str = Field(alias="Martial_Status")
    occupation: str = Field(alias="Occupation")
    relationship: str = Field(alias="Relationship")
    race: str = Field(alias="Race")
    sex: str = Field(alias="Sex")
    capital_gain: int = Field(alias="Capital_Gain")
    capital_loss: int = Field(alias="Capital_Loss")
    hours_per_week: int = Field(alias="Hours_per_week")
    native_country: str = Field(alias="Country")

    class Config:
        populate_by_name = True


class PredictRequest(BaseModel):
    records: List[AdultRecord]


# -------------------------------------------------
# In-Process Prometheus Metrics
# -------------------------------------------------
PREDICTION_COUNTER = Counter("adult_income_prediction_count", "Total predictions served")
DRIFT_SCORE = Gauge("adult_income_drift_score", "Global PSI drift score")
LABEL_COUNTER = Counter("adult_income_prediction_by_label", "Predictions by label", ["label"])
FEATURE_PSI = Gauge("adult_income_feature_psi", "Feature-level PSI", ["feature"])


# -------------------------------------------------
# Startup
# -------------------------------------------------
@app.on_event("startup")
def startup():
    global _model, _BASELINE_STATS
    _model = _load_model_from_s3()
    print("🚀 Model loaded successfully")

    _BASELINE_STATS = _load_baseline_stats(TRAINING_CSV_PATH)


# -------------------------------------------------
# Health
# -------------------------------------------------
@app.get("/ping")
def ping():
    return {"status": "ok"}


# -------------------------------------------------
# Metrics endpoint
# -------------------------------------------------
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# -------------------------------------------------
# Prediction Endpoint
# -------------------------------------------------
@app.post("/predict")
def predict(request: PredictRequest):

    rows = [
        (x.model_dump(by_alias=True) if hasattr(x, "model_dump") else x.dict(by_alias=True))
        for x in request.records
    ]

    df = pd.DataFrame(rows)
    print("Incoming DF:", df.columns.tolist())

    preds = _model.predict(df).tolist()

    # Drift Computation
    global_psi, psi_by_feature = compute_drift(df)

    DRIFT_SCORE.set(global_psi)
    for feat, val in psi_by_feature.items():
        FEATURE_PSI.labels(feature=feat).set(val)

    # Increment counters
    for p in preds:
        LABEL_COUNTER.labels(label=str(p)).inc()

    PREDICTION_COUNTER.inc(len(preds))

    # -------------------------------------------------
    # FIXED PUSHGATEWAY — TIME SERIES ENABLED
    # -------------------------------------------------
    try:
        registry = CollectorRegistry()

        # We rebuild ONLY the values we want to push
        g_drift = Gauge("adult_income_drift_score", "Global drift score", registry=registry)
        g_preds = Gauge("adult_income_prediction_count", "Total preds", registry=registry)

        g_drift.set(global_psi)
        g_preds.set(PREDICTION_COUNTER._value.get())

        # Feature PSI
        g_feat = Gauge(
            "adult_income_feature_psi",
            "Feature PSI metric",
            ["feature"],
            registry=registry,
        )

        for feat, val in psi_by_feature.items():
            g_feat.labels(feature=feat).set(val)

        # Label count
        g_label = Gauge(
            "adult_income_prediction_by_label",
            "Pred count by label",
            ["label"],
            registry=registry,
        )

        for lbl, metric in LABEL_COUNTER._metrics.items():
            g_label.labels(label=lbl[0]).set(metric._value.get())

        # --------------------------
        # CRITICAL: UNIQUE TIMESTAMP
        # --------------------------
        grouping_key = {
            "instance": "ecs_api",
            "ts": str(int(time.time()))  # new timestamp ensures time-series!
        }

        pushadd_to_gateway(
            PUSHGATEWAY_URL,
            job="adult_income_api",
            registry=registry,
            grouping_key=grouping_key
        )

        print(f"📡 Metrics pushed to Pushgateway ({PUSHGATEWAY_URL})")

    except Exception as e:
        print("⚠️ Pushgateway error:", e)

    return {
        "predictions": preds,
        "global_psi": global_psi,
        "psi_by_feature": psi_by_feature,
    }
