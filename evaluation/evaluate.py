# evaluation/evaluate.py
import os, json, tarfile
from pathlib import Path
import argparse

def find_any(root: str, *suffixes):
    p = Path(root)
    if not p.exists():
        return None
    for f in p.rglob("*"):
        if f.is_file() and any(str(f).endswith(s) for s in suffixes):
            return str(f)
    return None

def auto_find_model():
    # 1) when passed as Processing input
    cand = find_any("/opt/ml/processing/input", ".tar.gz", ".joblib")
    if cand:
        return cand
    # 2) fallback to training/serving dir
    cand = Path("/opt/ml/model/model.joblib")
    if cand.exists():
        return str(cand)
    return None

def main():
    # import AFTER we’ve decided not to mutate the env
    import joblib
    from sklearn import datasets
    from sklearn.metrics import accuracy_score, f1_score

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="Path to model.tar.gz or model.joblib")
    ap.add_argument("--output", default="/opt/ml/processing/evaluation")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    artifact = args.model or auto_find_model()
    if not artifact:
        raise FileNotFoundError(
            "Could not locate a model. Tried /opt/ml/processing/input and /opt/ml/model."
        )
    print(f"🔎 Using model artifact: {artifact}")

    if artifact.endswith(".tar.gz"):
        extracted_dir = "/opt/ml/processing/model_extracted"
        os.makedirs(extracted_dir, exist_ok=True)
        with tarfile.open(artifact, "r:gz") as tar:
            tar.extractall(extracted_dir)
        model_path = find_any(extracted_dir, ".joblib")
        if not model_path:
            raise FileNotFoundError("No .joblib found after extracting tarball")
    else:
        model_path = artifact

    print(f"📦 Loading model: {model_path}")
    model = joblib.load(model_path)

    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    yhat = model.predict(X)

    metrics = {
        "accuracy": float(accuracy_score(y, yhat)),
        "f1_macro": float(f1_score(y, yhat, average="macro")),
    }

    out_path = os.path.join(args.output, "evaluation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("✅ Wrote metrics:", metrics, "→", out_path)

if __name__ == "__main__":
    main()
