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
    """
    Backwards-compatible param loader.
    If params.yaml is missing or partial, we fall back to defaults.
    """
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
    """

    # 1. SageMaker environment variable (works on real SM jobs)
    sm_train = os.environ.get("SM_CHANNEL_TRAINING")
    if sm_train:
        candidate = os.path.join(sm_train, "adult.csv")
        if os.path.exists(candidate):
            return candidate

    # 2. Local docker SageMaker-style path: /opt/ml/input/data/training/
    docker_sm_train = "/opt/ml/input/data/training/adult.csv"
    if os.path.exists(docker_sm_train):
        return docker_sm_train

    # 3. Repo local paths
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
        "adult.csv not found. Checked SM_CHANNEL_TRAINING brin, "
        "/opt/ml/input/data/training/adult.csv, and local ./data/adult.csv"
    )



def load_adult_data(csv_path: str):
    """
    Load Adult Census data from CSV and normalise column names.
    Assumes UCI-like schema with 'income' as target.
    """
    df = pd.read_csv(csv_path)

    # Strip whitespace and normalise column names to snake_case
    df.columns = [c.strip() for c in df.columns]

    # Map common UCI Adult column names to canonical ones
    col_map = {
        "age": "age",
        "workclass": "workclass",
        "fnlwgt": "fnlwgt",
        "education": "education",
        "education-num": "education_num",
        "marital-status": "marital_status",
        "occupation": "occupation",
        "relationship": "relationship",
        "race": "race",
        "sex": "sex",
        "capital-gain": "capital_gain",
        "capital-loss": "capital_loss",
        "hours-per-week": "hours_per_week",
        "native-country": "native_country",
        "income": "income",
    }

    # Only rename columns that actually exist
    rename_dict = {old: new for old, new in col_map.items() if old in df.columns}
    df = df.rename(columns=rename_dict)

    if "income" not in df.columns:
        # Try case-insensitive search
        lower_map = {c.lower(): c for c in df.columns}
        if "income" in lower_map:
            df = df.rename(columns={lower_map["income"]: "income"})
        else:
            raise ValueError("Could not find 'income' target column in adult.csv")

    # Basic cleaning: drop rows with missing target
    df = df.dropna(subset=["income"])

    # Replace '?' placeholders with NaN and drop them for simplicity
    df = df.replace("?", pd.NA).dropna()

    y = df["income"].astype(str)  # e.g. '<=50K', '>50K'
    X = df.drop(columns=["income"])

    return X, y


def build_pipeline(X_sample, random_state: int, xgb_params: dict):
    """
    Build a preprocessing + XGBoost pipeline based on the sample dataframe.
    """
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

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )

    return model


def main():
    # --- resolve hyperparameters ---
    params = load_params()
    yaml_test_size = float(params["training"]["test_size"])
    yaml_random_state = int(params["training"]["random_state"])
    xgb_cfg = params["training"]["xgb"]

    ap = argparse.ArgumentParser()
    ap.add_argument("--test-size", type=float, default=None)
    ap.add_argument("--random-state", type=int, default=None)
    args = ap.parse_args()

    test_size = resolve(args.test_size, yaml_test_size, 0.2)
    random_state = resolve(args.random_state, yaml_random_state, 42)

    # --- SageMaker dirs ---
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # --- Load data ---
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

    # --- Build and train model ---
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

    # --- Save outputs for SageMaker ---
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({**metrics, "resolved_params": resolved_params}, f, indent=2)

    print(f"✅ Model saved to {model_path}")
    print(f"✅ Metrics saved to {metrics_path}")
    print("ℹ️ Params used:", resolved_params)
    print("📊 Metrics:", metrics)

    # --- MLflow logging (file-based inside container) ---
    mlflow_tracking_dir = "/opt/ml/output/data/mlruns"
    mlflow.set_tracking_uri(f"file:{mlflow_tracking_dir}")
    mlflow.set_experiment("adult-income-xgboost")
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://s3.eu-west-1.amazonaws.com"
    mlflow.set_registry_uri(f"file:{mlflow_tracking_dir}")
    print(f"📊 [MLflow] tracking to {mlflow_tracking_dir}")

    run_name = f"adult-xgb-train-{datetime.datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(resolved_params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_artifact(metrics_path, artifact_path="metrics")
        print("✅ [MLflow] Run logged successfully.")


if __name__ == "__main__":
    main()
