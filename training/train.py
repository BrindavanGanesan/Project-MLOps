import argparse
import os
import json
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def main():
    # SageMaker provides /opt/ml paths for input/output
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.25)
    args = parser.parse_args()

    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")

    # Load dataset (Iris for demo)
    X, y = load_iris(return_X_y=True, as_frame=False)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    # Define pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    # Train
    model.fit(Xtr, ytr)

    # Evaluate
    pred = model.predict(Xte)
    metrics = {
        "accuracy": float(accuracy_score(yte, pred)),
        "f1_macro": float(f1_score(yte, pred, average="macro")),
    }

    # Save model to /opt/ml/model so SageMaker can package it
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)

    # Save metrics to output directory
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"✅ Model saved to {model_path}")
    print(f"✅ Metrics saved to {metrics_path}")
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
