# training/train.py
import argparse
import os
import json
import joblib
import pathlib
import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

import mlflow
import mlflow.sklearn


def load_params(path="config/params.yaml"):
    defaults = {
        "training": {
            "test_size": 0.2,
            "random_state": 42,
            "xgb": {
                "n_estimators": 200,
                "max_depth": 5,
                "learning_rate": 0.1,
            },
        }
    }
    p = pathlib.Path(path)
    if not p.exists():
        return defaults

    try:
        import yaml
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return defaults

    data.setdefault("training", {})
    t = data["training"]
    t.setdefault("test_size", defaults["training"]["test_size"])
    t.setdefault("random_state", defaults["training"]["random_state"])
    t.setdefault("xgb", {})
    x = t["xgb"]
    x.setdefault("n_estimators", defaults["training"]["xgb"]["n_estimators"])
    x.setdefault("max_depth", defaults["training"]["xgb"]["max_depth"])
    x.setdefault("learning_rate", defaults["training"]["xgb"]["learning_rate"])
    return data


def resolve(cli_val, yaml_val, default):
    return cli_val if cli_val is not None else (yaml_val if yaml_val is not None else default)


def find_adult_csv():
    """
    Detect Adult CSV from SageMaker channels or fallback local paths.
    Works for local Docker and SageMaker BYOC.
    """

    # 1️⃣ SageMaker environment (real SM training job)
    sm_train = os.environ.get("SM_CHANNEL_TRAINING")
    if sm_train:
        candidate = os.path.join(sm_train, "adult.csv")
        if os.path.exists(candidate):
            return candidate

    # 2️⃣ Local Docker SageMaker-style path
    docker_candidate = "/opt/ml/input/data/training/adult.csv"
    if os.path.exists(docker_candidate):
        return docker_candidate

    # 3️⃣ Fallback local repo paths
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / "data" / "adult.csv",
        repo_root / "adult.csv",
        pathlib.Path("data") / "adult.csv",
    ]

    for c in candidates:
        if c.exists():
            return str(c)

    raise FileNotFoundError(
        "adult.csv not found. Checked SM_CHANNEL_TRAINING, "
        "/opt/ml/input/data/training/adult.csv, and ./data/adult.csv"
    )


def load_adult_data(csv_path: str):
    """
    Load Adult Census data, normalize column names, and convert target to 0/1.
    """
    df = pd.read_csv(csv_path)

    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    # Detect target column
    POSSIBLE_TARGETS = ["income", "target", "class", "salary"]

    actual_target = None
    for col in df.columns:
        if col.strip().lower() in POSSIBLE_TARGETS:
            actual_target = col
            break

    if actual_target is None:
        raise ValueError(
            f"Could not find target column. Expected one of {POSSIBLE_TARGETS}, found: {df.columns.tolist()}"
        )

    # Normalize name → income
    df = df.rename(columns={actual_target: "income"})

    # Clean target values
    df["income"] = df["income"].astype(str).str.strip()

    label_map = {
        "<=50K": 0,
        ">50K": 1,
        "<=50K.": 0,  # sometimes present
        ">50K.": 1,
    }

    df["income"] = df["income"].replace(label_map)

    # Validate mapping
    if df["income"].isnull().any():
        raise ValueError(
            f"Unrecognized class labels! Found: {df['income'].unique()}"
        )

    # Drop missing rows
    df = df.replace("?", pd.NA).dropna()

    y = df["income"].astype(int)
    X = df.drop(columns=["income"])

    return X, y

def build_pipeline(X_sample, random_state: int, xgb_params: dict):
    numeric_features = X_sample.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = [c for c in X_sample.columns if c not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    clf = XGBClassifier(
        n_estimators=int(xgb_params["n_estimators"]),
        max_depth=int(xgb_params["max_depth"]),
        learning_rate=float(xgb_params["learning_rate"]),
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        n_jobs=-1,
        random_state=int(random_state),
        tree_method="hist",
    )

    return Pipeline([
        ("preprocess", preprocessor),
        ("clf", clf),
    ])


def main():
    # Load parameters
    params = load_params()
    yaml_test_size = float(params["training"]["test_size"])
    yaml_random_state = int(params["training"]["random_state"])
    xgb_cfg = params["training"]["xgb"]

    # Parse arguments but ignore SageMaker's extra args
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-size", type=float, default=None)
    ap.add_argument("--random-state", type=int, default=None)
    args, _ = ap.parse_known_args()

    test_size = resolve(args.test_size, yaml_test_size, 0.2)
    random_state = resolve(args.random_state, yaml_random_state, 42)

    # SageMaker paths
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    csv_path = find_adult_csv()
    print(f"📂 Using Adult dataset from: {csv_path}")
    X, y = load_adult_data(csv_path)

    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=y,
    )

    model = build_pipeline(Xtr, random_state, xgb_cfg)
    model.fit(Xtr, ytr)

    pred = model.predict(Xte)
    metrics = {
        "accuracy": float(accuracy_score(yte, pred)),
        "f1_macro": float(f1_score(yte, pred, average="macro")),
    }

    resolved_params = {
        "test_size": float(test_size),
        "random_state": int(random_state),
        "n_estimators": int(xgb_cfg["n_estimators"]),
        "max_depth": int(xgb_cfg["max_depth"]),
        "learning_rate": float(xgb_cfg["learning_rate"]),
        "model_type": "XGBClassifier",
    }

    # Save artifacts for SageMaker
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({**metrics, "resolved_params": resolved_params}, f, indent=2)

    print(f"✅ Model saved to: {model_path}")
    print(f"📊 Metrics saved to: {metrics_path}")

    # MLflow logging
    mlflow_tracking_dir = "/opt/ml/output/data/mlruns"
    mlflow.set_tracking_uri(f"file:{mlflow_tracking_dir}")
    mlflow.set_experiment("adult-income-xgboost")
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://s3.eu-west-1.amazonaws.com"
    mlflow.set_registry_uri(f"file:{mlflow_tracking_dir}")

    run_name = f"adult-xgb-train-{datetime.datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(resolved_params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_artifact(metrics_path, artifact_path="metrics")
        print("✅ [MLflow] Run logged successfully.")


if __name__ == "__main__":
    main()
