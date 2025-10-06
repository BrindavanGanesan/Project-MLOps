# app/main.py
import os, io, tarfile, tempfile, json, joblib, boto3
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Iris Classifier", version="1.0")

MODEL_S3_URI = os.getenv("MODEL_S3_URI")  # e.g. s3://thebrowntiger/<job>/output/model.tar.gz
AWS_REGION = os.getenv("AWS_REGION", "eu-west-1")

_model = None

def _parse_s3(uri: str):
    assert uri.startswith("s3://")
    b, k = uri[5:].split("/", 1)
    return b, k

def _load_model_from_s3():
    """Downloads model.tar.gz (or model.joblib) from S3 and loads it."""
    s3 = boto3.client("s3", region_name=AWS_REGION)
    bucket, key = _parse_s3(MODEL_S3_URI)

    # download to memory
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = io.BytesIO(obj["Body"].read())

    # support either .tar.gz or .joblib
    if key.endswith(".tar.gz"):
        with tarfile.open(fileobj=body, mode="r:gz") as tar:
            # typical SageMaker export: model.joblib at root
            with tempfile.TemporaryDirectory() as td:
                tar.extractall(td)
                mp = os.path.join(td, "model.joblib")
                if not os.path.exists(mp):
                    # search
                    cand = [m for m in tar.getnames() if m.endswith(".joblib")]
                    if not cand:
                        raise FileNotFoundError("No *.joblib found in model.tar.gz")
                    # extract again and pick first
                    body.seek(0)
                    with tarfile.open(fileobj=body, mode="r:gz") as tar2:
                        tar2.extract(cand[0], td)
                    mp = os.path.join(td, cand[0])
                model = joblib.load(mp)
                return model
    else:
        # direct joblib path in S3 (less common)
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            f.write(body.getvalue())
            f.flush()
            return joblib.load(f.name)

class PredictRequest(BaseModel):
    # 4 features for iris dataset
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
def predict(req: PredictRequest):
    preds = _model.predict(req.data).tolist()
    return {"predictions": preds}
