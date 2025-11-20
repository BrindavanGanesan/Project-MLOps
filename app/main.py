# app/main.py
import os
import io
import tarfile
import tempfile
import math
from typing import List, Dict, Any, Optional

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
MODEL_S3_URI = os.getenv("MODEL_S3_URI")  # required
AWS_REGION = os.getenv("AWS_REGION", "eu-west-1")

# Where the *training* adult.csv lives inside the container
TRAINING_CSV_PATH = os.getenv("TRAINING_CSV_PATH", "/app/data/adult.csv")

# Default PushGateway
PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL", "http://108.130.158.94:9091")

# PSI config
PSI_NUM_BINS = int(os.getenv("PSI_NUM_BINS", "10"))
PSI_EPS = 1e-6

# ------------------ Globals ----------------------
_model = None
_BASELINE_STATS: Dict[str, Dict[str, Any]] = {}  # per-feature ref distributions


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
# Baseline PSI stats (from training CSV)
# -------------------------------------------------
def _load_baseline_stats(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Precompute reference distributions for PSI from the training dataset.
    Returns a dict:
        {
          "Age": {
              "type": "numeric",
              "bins": [...],
              "ref": [...]
          },
          "Workclass": {
              "type": "categorical",
              "ref": {" Private": 0.7, ...}
          },
          ...
        }
    """
    if not os.path.exists(csv_path):
        print(f"⚠️ Training CSV not found at {csv_path}; PSI will be disabled.")
        return {}

    df = pd.read_csv(csv_path)
    print(f"📊 Loaded training baseline from {csv_path} with cols: {df.columns.tolist()}")

    stats: Dict[str, Dict[str, Any]] = {}

    for col in df.columns:
        if col.lower() in {"income", "target"}:
            continue

        s = df[col].dropna()

        # numeric?
        if pd.api.types.is_numeric_dtype(s):
            # quantile-based bins
            qs = [i / PSI_NUM_BINS for i in range(PSI_NUM_BINS + 1)]
            edges = s.quantile(qs).values
            # ensure strictly increasing bin edges
            edges = sorted(set(float(x) for x in edges))
            if len(edges) < 2:
                # degenerate distribution; skip
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


def _psi_numeric(ref_props, cur_props) -> float:
    psi = 0.0
    for pr, pc in zip(ref_props, cur_props):
        pr = float(pr) if pr is not None else 0.0
        pc = float(pc) if pc is not None else 0.0
        if pr < PSI_EPS and pc < PSI_EPS:
            continue
        pr = max(pr, PSI_EPS)
        pc = max(pc, PSI_EPS)
        psi += (pr - pc) * math.log(pr / pc)
    return float(psi)


def _psi_categorical(ref_dict: Dict[Any, float], cur_dict: Dict[Any, float]) -> float:
    psi = 0.0
    categories = set(ref_dict.keys()) | set(cur_dict.keys())
    for cat in categories:
        pr = float(ref_dict.get(cat, 0.0))
        pc = float(cur_dict.get(cat, 0.0))
        if pr < PSI_EPS and pc < PSI_EPS:
            continue
        pr = max(pr, PSI_EPS)
        pc = max(pc, PSI_EPS)
        psi += (pr - pc) * math.log(pr / pc)
    return float(psi)


def compute_drift(df: pd.DataFrame) -> tuple[float, Dict[str, float]]:
    """
    Compute PSI per feature vs baseline and a global drift score (max PSI).
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
            # use training bins
            edges = meta["bins"]
            ref_props = meta["ref"]

            cur_counts, _ = np.histogram(s, bins=edges)
            total = cur_counts.sum()
            if total == 0:
                continue
            cur_props = (cur_counts / total).tolist()

            # align lengths just in case
            m = min(len(ref_props), len(cur_props))
            psi = _psi_numeric(ref_props[:m], cur_props[:m])
        else:
            # categorical
            cur_freq = s.value_counts(normalize=True)
            psi = _psi_categorical(meta["ref"], cur_freq.to_dict())

        psi_by_feature[col] = psi

    if not psi_by_feature:
        return 0.0, {}

    global_psi = max(psi_by_feature.values())
    return float(global_psi), psi_by_feature


# -------------------------------------------------
# Request Schemas (with aliases matching training CSV)
# -------------------------------------------------
class AdultRecord(BaseModel):
    # Aliases = exact column names used in training data
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
        # allow using either the field names (age) or aliases (Age)
        populate_by_name = True


class PredictRequest(BaseModel):
    records: List[AdultRecord]


# -------------------------------------------------
# Prometheus Metrics
# -------------------------------------------------
PREDICTION_COUNTER = Counter(
    "adult_income_prediction_count",
    "Total number of predictions served",
)

DRIFT_SCORE = Gauge(
    "adult_income_drift_score",
    "Global drift score (max PSI across features) vs training baseline",
)

LABEL_COUNTER = Counter(
    "adult_income_prediction_by_label",
    "Prediction count by label",
    ["label"],
)

FEATURE_PSI = Gauge(
    "adult_income_feature_psi",
    "Population Stability Index per feature vs training baseline",
    ["feature"],
)


# -------------------------------------------------
# FastAPI Startup
# -------------------------------------------------
@app.on_event("startup")
def startup():
    global _model, _BASELINE_STATS

    if not MODEL_S3_URI:
        raise RuntimeError("❌ MODEL_S3_URI environment variable must be set!")

    _model = _load_model_from_s3()
    print("🚀 Model is ready!")

    _BASELINE_STATS = _load_baseline_stats(TRAINING_CSV_PATH)


# -------------------------------------------------
# Health Check
# -------------------------------------------------
@app.get("/ping")
def ping():
    return {"status": "ok"}


# -------------------------------------------------
# Metrics Endpoint (Prometheus pull)
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

    # Use aliases so we get the *training* column names
    records: List[dict] = []
    for r in request.records:
        if hasattr(r, "model_dump"):
            records.append(r.model_dump(by_alias=True))
        else:  # pydantic v1 fallback
            records.append(r.dict(by_alias=True))

    df = pd.DataFrame(records)

    # Log columns for debugging
    print("Incoming DF columns:", df.columns.tolist())

    # Run prediction
    preds = _model.predict(df)
    preds = preds.tolist()

    # ---------------------------------------------
    # Drift & PSI
    # ---------------------------------------------
    global_psi, psi_by_feature = compute_drift(df)
    DRIFT_SCORE.set(round(global_psi, 4))

    for feat, psi_val in psi_by_feature.items():
        FEATURE_PSI.labels(feature=feat).set(round(float(psi_val), 6))

    # Label metrics
    for p in preds:
        LABEL_COUNTER.labels(label=str(p)).inc()

    PREDICTION_COUNTER.inc(len(preds))

    # ---------------------------------------------
    # Push to PushGateway (uses default REGISTRY)
    # ---------------------------------------------
    try:
        registry = CollectorRegistry()

        # Recreate metrics inside this registry
        drift_gauge = Gauge("adult_income_drift_score", "Drift score", registry=registry)
        pred_counter = Counter("adult_income_prediction_count", "Total predictions", registry=registry)
        label_counter = Counter("adult_income_prediction_by_label", "Prediction count by label", ["label"], registry=registry)

        # Set values
        drift_gauge.set(global_psi)
        pred_counter.inc(len(preds))

        for p in preds:
            label_counter.labels(label=str(p)).inc()

    # Push metrics
        push_to_gateway(
            PUSHGATEWAY_URL,
            job="adult_income_api",
            registry=registry
        )

        print(f"📡 Metrics pushed to {PUSHGATEWAY_URL}")

    except Exception as e:
        print(f"⚠️ Pushgateway push failed: {e}")
