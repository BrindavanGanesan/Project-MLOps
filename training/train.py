# training/train.py
import argparse, os, json, joblib, pathlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn


def load_params(path="config/params.yaml"):
    defaults = {
        "training": {
            "test_size": 0.25,
            "random_state": 42,
            "xgb": {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1
            }
        }
    }
    p = pathlib.Path(path)
    if not p.exists():
        return defaults

    try:
        import yaml
        with open(p, "r") as f:
            data = yaml.safe_load(f) or {}
    except:
        return defaults

    return {**defaults, **data}


def resolve(cli_val, yaml_val, default):
    return cli_val if cli_val is not None else yaml_val if yaml_val is not None else default


def main():
    # ----------------
    # Load hyperparameters
    # ----------------
    params = load_params()
    yml = params["training"]

    ap = argparse.ArgumentParser()
    ap.add_argument("--test-size", type=float, default=None)
    ap.add_argument("--random-state", type=int, default=None)
    ap.add_argument("--n-estimators", type=int, default=None)
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--learning-rate", type=float, default=None)
    args = ap.parse_args()

    test_size = resolve(args.test_size, yml["test_size"], 0.25)
    random_state = resolve(args.random_state, yml["random_state"], 42)

    n_estimators = resolve(args.n_estimators, yml["xgb"]["n_estimators"], 200)
    max_depth = resolve(args.max_depth, yml["xgb"]["max_depth"], 6)
    learning_rate = resolve(args.learning_rate, yml["xgb"]["learning_rate"], 0.1)

    resolved_params = dict(
        test_size=float(test_size),
        random_state=int(random_state),
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
    )

    # ----------------
    # Load dataset
    # ----------------
    df = pd.read_csv("data/adult.csv")

    target = "Target"
    X = df.drop(columns=[target])
    y = (df[target] == ">50K").astype(int)

    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    numeric = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric)
        ]
    )

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist"   # FAST CPU training
    )

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("clf", model)
    ])

    # ----------------
    # Train/test split
    # ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1_macro": float(f1_score(y_test, preds, average="macro"))
    }

    # ----------------
    # Persist artifacts for SageMaker
    # ----------------
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(pipeline, model_path)

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump({**metrics, "resolved_params": resolved_params}, f, indent=2)

    print("✅ Model saved:", model_path)
    print("📊 Metrics:", metrics)

    # ----------------
    # MLflow Logging
    # ----------------
    mlflow_dir = "/opt/ml/output/data/mlruns"
    os.makedirs(mlflow_dir, exist_ok=True)

    mlflow.set_tracking_uri(f"file:{mlflow_dir}")
    mlflow.set_experiment("adult-xgboost")

    with mlflow.start_run():
        mlflow.log_params(resolved_params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, "model")
        print("✅ MLflow run logged")


if __name__ == "__main__":
    main()
