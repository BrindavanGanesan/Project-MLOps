# app/main.py
import os
import io
import tarfile
import tempfile
import math
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
    push_to_gateway,
    CollectorRegistry,
)

# ------------------ FastAPI App ------------------
app = FastAPI(title="Adult Income Classifier", version="3.0")

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
# S3 Utilities
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

    # SageMaker-style model.tar.gz
    if key.endswith(".tar.gz"):
        with tarfile.open(fileobj=body, mode="r:gz") as tar:
            with tempfile.TemporaryDirectory() as td:
                tar.extractall(td)
                joblib_path = os.path.join(td, "model.joblib")

                if not os.path.exists(joblib_path):
                    for m in tar.getmembers():
                        if m.name.endswith(".joblib"):
                            tar.extract(m, td)
                            joblib_path = os.path.join(td, m.name)
                            break

                return joblib.load(joblib_path)

    # Plain .joblib
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        f.write(body.getvalue())
        return joblib.load(f.name)


# -------------------------------------------------
# Baseline PSI stats
# -------------------------------------------------
def _load_baseline_stats(csv_path: str):
    if not os.path.exists(csv_path):
        print(f"⚠️ Baseline CSV not found at {csv_path}")
        return {}

    df = pd.read_csv(csv_path)
    print(f"📊 Loaded training baseline from {csv_path} with cols: {df.columns.tolist()}")

    stats: Dict[str, Dict[str, Any]] = {}

    for col in df.columns:
        # Skip label column
        if col.lower() in {"income", "target"}:
            continue

        s = df[col].dropna()
        if s.empty:
            continue

        if pd.api.types.is_numeric_dtype(s):
            # quantile-based bin edges
            qs = [i / PSI_NUM_BINS for i in range(PSI_NUM_BINS + 1)]
            edges = sorted(set(float(x) for x in s.quantile(qs).values))
            if len(edges) < 2:
                continue

            ref_counts, _ = np.histogram(s, bins=edges)
            total = ref_counts.sum()
            if total == 0:
                continue
            ref_props = (ref_counts / total).tolist()

            stats[col] = {
                "type": "numeric",
                "bins": edges,
                "ref": ref_props,
            }
        else:
            freq = s.value_counts(normalize=True)
            stats[col] = {
                "type": "categorical",
                "ref": freq.to_dict(),
            }

    print(f"✅ Built baseline PSI stats for features: {list(stats.keys())}")
    return stats


def _psi_numeric(ref_props, cur_props):
    psi = 0.0
    for pr, pc in zip(ref_props, cur_props):
        pr = max(float(pr), PSI_EPS)
        pc = max(float(pc), PSI_EPS)
        psi += (pr - pc) * math.log(pr / pc)
    return float(psi)


def _psi_categorical(ref_dict: Dict[str, float], cur_dict: Dict[str, float]):
    psi = 0.0
    categories = set(ref_dict.keys()) | set(cur_dict.keys())
    for cat in categories:
        pr = max(float(ref_dict.get(cat, 0.0)), PSI_EPS)
        pc = max(float(cur_dict.get(cat, 0.0)), PSI_EPS)
        psi += (pr - pc) * math.log(pr / pc)
    return float(psi)


def compute_drift(df: pd.DataFrame):
    """
    Compute PSI per feature vs training baseline and a global score
    (max PSI across features).
    """
    if not _BASELINE_STATS:
        return 0.0, {}

    psi_by_feature: Dict[str, float] = {}

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
            total = cur_counts.sum()
            if total == 0:
                continue
            cur_props = (cur_counts / total).tolist()

            n = min(len(ref_props), len(cur_props))
            psi = _psi_numeric(ref_props[:n], cur_props[:n])
        else:
            cur_freq = s.value_counts(normalize=True)
            psi = _psi_categorical(meta["ref"], cur_freq.to_dict())

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
# Prometheus Metrics (in-process / scraped)
# -------------------------------------------------
PREDICTION_COUNTER = Counter(
    "adult_income_prediction_count",
    "Total number of predictions served",
)

DRIFT_SCORE = Gauge(
    "adult_income_drift_score",
    "Global drift score (max PSI across features)",
)

LABEL_COUNTER = Counter(
    "adult_income_prediction_by_label",
    "Prediction count by label",
    ["label"],
)

FEATURE_PSI = Gauge(
    "adult_income_feature_psi",
    "PSI per feature vs training baseline",
    ["feature"],
)


# -------------------------------------------------
# Startup
# -------------------------------------------------
@app.on_event("startup")
def startup():
    global _model, _BASELINE_STATS
    _model = _load_model_from_s3()
    print("🚀 Model is ready!")

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

    records = [
        (r.model_dump(by_alias=True)
         if hasattr(r, "model_dump")
         else r.dict(by_alias=True))
        for r in request.records
    ]

    df = pd.DataFrame(records)
    print("Incoming DF:", df.columns.tolist())

    # Model prediction
    preds = _model.predict(df).tolist()

    # Drift / PSI
    global_psi, psi_by_feature = compute_drift(df)

    # --- Update in-process metrics (for /metrics scrape) ---
    DRIFT_SCORE.set(global_psi)

    for feat, val in psi_by_feature.items():
        FEATURE_PSI.labels(feature=feat).set(val)

    for p in preds:
        LABEL_COUNTER.labels(label=str(p)).inc()

    PREDICTION_COUNTER.inc(len(preds))

    # --- Push snapshot to Pushgateway (separate registry) ---
    try:
        registry = CollectorRegistry()

        g_drift = Gauge(
            "adult_income_drift_score",
            "Global drift score (max PSI across features)",
            registry=registry,
        )
        g_preds = Gauge(
            "adult_income_prediction_count_total",
            "Total number of predictions served",
            registry=registry,
        )
        g_feat = Gauge(
            "adult_income_feature_psi",
            "PSI per feature vs training baseline",
            ["feature"],
            registry=registry,
        )
        g_label = Gauge(
            "adult_income_prediction_by_label_total",
            "Prediction count by label",
            ["label"],
            registry=registry,
        )

        # Use absolute values from in-process metrics so Prometheus
        # sees a monotonically increasing / stable series.
        g_drift.set(global_psi)
        g_preds.set(PREDICTION_COUNTER._value.get())

        for feat, val in psi_by_feature.items():
            g_feat.labels(feature=feat).set(val)

        for label_tuple, metric in LABEL_COUNTER._metrics.items():
            (label,) = label_tuple
            g_label.labels(label=label).set(metric._value.get())

        push_to_gateway(
            PUSHGATEWAY_URL,
            job="adult_income_api",
            registry=registry,
        )
        print(f"📡 Metrics pushed to {PUSHGATEWAY_URL}")

    except Exception as e:
        print(f"⚠️ Pushgateway error: {e}")

    return {
        "predictions": preds,
        "global_psi": global_psi,
        "psi_by_feature": psi_by_feature,
    }
