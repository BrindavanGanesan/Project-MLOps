# training/train.py
import argparse, os, json, joblib, pathlib

def load_params(path="config/params.yaml"):
    defaults = {"training": {"test_size": 0.25, "random_state": 42, "clf": {"max_iter": 1000}}}
    p = pathlib.Path(path)
    if not p.exists():
        return defaults
    try:
        import yaml  # optional
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return defaults
    data.setdefault("training", {})
    data["training"].setdefault("test_size", defaults["training"]["test_size"])
    data["training"].setdefault("random_state", defaults["training"]["random_state"])
    data["training"].setdefault("clf", {})
    data["training"]["clf"].setdefault("max_iter", defaults["training"]["clf"]["max_iter"])
    return data

def resolve(cli_val, yaml_val, default):
    return cli_val if cli_val is not None else (yaml_val if yaml_val is not None else default)

def main():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    import mlflow
    import mlflow.sklearn

    # --- resolve hyperparameters ---
    params = load_params()
    yaml_test_size   = float(params["training"]["test_size"])
    yaml_random_state= int(params["training"]["random_state"])
    yaml_max_iter    = int(params["training"]["clf"]["max_iter"])

    ap = argparse.ArgumentParser()
    ap.add_argument("--test-size", type=float, default=None)
    ap.add_argument("--random-state", type=int, default=None)
    ap.add_argument("--max-iter", type=int, default=None)
    args = ap.parse_args()

    test_size    = resolve(args.test_size,   yaml_test_size,   0.25)
    random_state = resolve(args.random_state,yaml_random_state,42)
    max_iter     = resolve(args.max_iter,    yaml_max_iter,    1000)

    # --- SageMaker dirs ---
    model_dir  = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # --- Train ---
    X, y = load_iris(return_X_y=True, as_frame=False)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=float(test_size), random_state=int(random_state), stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=int(max_iter))),
    ])
    model.fit(Xtr, ytr)

    pred = model.predict(Xte)
    metrics = {
        "accuracy": float(accuracy_score(yte, pred)),
        "f1_macro": float(f1_score(yte, pred, average="macro")),
    }
    resolved_params = {
        "test_size": float(test_size),
        "random_state": int(random_state),
        "max_iter": int(max_iter),
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

    # --- MLflow logging (stored in output/data/mlruns) ---
    # --- MLflow logging to S3 instead of local ephemeral storage ---
    import datetime
    mlflow_tracking_dir = "/opt/ml/output/data/mlruns"
    mlflow.set_tracking_uri(f"file:{mlflow_tracking_dir}")
    mlflow.set_experiment("thesis-iris")
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://s3.eu-west-1.amazonaws.com"
    mlflow.set_registry_uri(f"file:{mlflow_tracking_dir}")  # local registry
    print(f"📊 [MLflow] tracking to {mlflow_tracking_dir}")

    with mlflow.start_run(run_name=f"sagemaker-train-{datetime.datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"):
        mlflow.log_params(resolved_params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")  # don't register here
        mlflow.log_artifact(metrics_path, artifact_path="metrics")
        print("✅ [MLflow] Run logged successfully.")

if __name__ == "__main__":
    main()
