# pipeline/run_training_job.py
import os, io, time, tarfile, shutil, boto3, botocore
import yaml
from pathlib import Path
from datetime import datetime
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.session import Session

CFG_PATH = os.path.join("config", "config.yaml")

def load_cfg(path=CFG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def parse_s3_uri(uri: str):
    assert uri.startswith("s3://"), f"Not an s3:// URI: {uri}"
    rest = uri[5:]
    bucket, key = rest.split("/", 1)
    return bucket, key

def wait_for_key(s3, bucket, key, retries=12, sleep_s=2):
    for _ in range(retries):
        try:
            s3.head_object(Bucket=bucket, Key=key)
            return True
        except botocore.exceptions.ClientError as e:
            if e.response.get("Error", {}).get("Code") == "404":
                time.sleep(sleep_s)
                continue
            raise
    return False

def mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def copy_tree(src: Path, dst: Path):
    """Merge-copy directory tree while preserving structure."""
    for root, _, files in os.walk(src):
        rel = Path(root).relative_to(src)
        tgt = dst / rel
        tgt.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(Path(root) / f, tgt / f)

def main():
    cfg = load_cfg()
    region   = cfg["aws"]["region"]
    role_arn = cfg["aws"]["role_arn"]
    bucket   = cfg["aws"]["bucket"]

    assert role_arn.startswith("arn:aws:iam::"), "role_arn looks invalid"
    assert bucket, "bucket must be set in config.yaml"

    boto_sess  = boto3.Session(region_name=region)
    sm_session = Session(boto_session=boto_sess, default_bucket=bucket)

    job_suffix   = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    base_job_name = f"thesis-train-{job_suffix}"

    use_spot = cfg.get("training", {}).get("use_spot", False)

    est = SKLearn(
        entry_point="train.py",
        source_dir="training",
        role=role_arn,
        framework_version=cfg["training"]["framework_version"],
        instance_type=cfg["training"]["instance_type"],
        instance_count=1,
        py_version="py3",
        sagemaker_session=sm_session,
        base_job_name=base_job_name,
        hyperparameters={"test-size": cfg["evaluation"]["test_size"]},
        enable_sagemaker_metrics=True,
        use_spot_instances=use_spot,
        **({"max_run": 3600, "max_wait": 7200} if use_spot else {})
    )

    print(f"Starting training job in {region} with role {role_arn} …")
    est.fit(wait=True)

    job_name = est.latest_training_job.name
    model_s3 = est.model_data
    print("\n✅ Training job finished.")
    print("Job name:", job_name)
    print("Model artifact S3:", model_s3)
    print("\nNext: you can open SageMaker → Training jobs to inspect logs/metrics.")

    # ---------- find and fetch output.tar.gz ----------
    bucket_name, model_key = parse_s3_uri(model_s3)
    output_tar_key = model_key.replace("/output/model.tar.gz", "/output/output.tar.gz")

    s3 = boto3.client("s3", region_name=region)
    print("Looking for training outputs tar:", f"s3://{bucket_name}/{output_tar_key}")
    if not wait_for_key(s3, bucket_name, output_tar_key):
        raise RuntimeError(f"output.tar.gz not found yet at s3://{bucket_name}/{output_tar_key}")

    obj = s3.get_object(Bucket=bucket_name, Key=output_tar_key)
    data_bytes = obj["Body"].read()

    # Local run folder to keep a copy of the tar + extraction
    run_root        = mkdir(Path("artifacts") / "sm-output" / job_name)
    local_output_tar = run_root / "output.tar.gz"
    local_output_tar.write_bytes(data_bytes)

    extracted_dir = mkdir(run_root / "extracted")

    # Open tar (from bytes) once; also extract for mlruns + metrics discovery
    with tarfile.open(fileobj=io.BytesIO(data_bytes), mode="r:gz") as tar:
        # Extract everything so we can locate mlruns and metrics robustly
        tar.extractall(extracted_dir)

        # -------- metrics.json -> artifacts/metrics/metrics.json (for DVC) --------
        metrics_candidates = [m for m in tar.getmembers() if m.name.endswith("metrics.json")]
        if not metrics_candidates:
            raise FileNotFoundError(
                "metrics.json not found in output.tar.gz"
            )

    # Now that it's extracted, copy the first metrics.json found
    first_metrics = next(extracted_dir.rglob("metrics.json"))
    local_metrics_dir = mkdir(Path("artifacts") / "metrics")
    local_metrics_path = local_metrics_dir / "metrics.json"
    shutil.copy2(first_metrics, local_metrics_path)
    print("✅ Saved local metrics for DVC:", local_metrics_path)

      # --- also fetch the model.tar.gz so DVC can track it ---
    local_model_dir = os.path.join("artifacts", "model")
    os.makedirs(local_model_dir, exist_ok=True)
    local_model_path = os.path.join(local_model_dir, "model.tar.gz")

    print("Downloading model.tar.gz →", local_model_path)
    s3.download_file(bucket, model_key, local_model_path)

    print("✅ Saved model artifact for DVC:", local_model_path)
    
    # -------- mlruns -> ./mlruns (merge-copy so UI sees the run immediately) --------
    mlruns_candidates = [p for p in extracted_dir.rglob("mlruns") if p.is_dir()]
    if mlruns_candidates:
        src_mlruns = mlruns_candidates[0]
        dst_mlruns = mkdir(Path("mlruns"))
        copy_tree(src_mlruns, dst_mlruns)
        print(f"📝 Copied MLflow runs → {dst_mlruns}")
        print("   Tip: run `mlflow ui --backend-store-uri ./mlruns` to view the run.")
    else:
        print("ℹ️  No `mlruns/` folder found in outputs. "
              "Ensure train.py logs to `file:/opt/ml/output/mlruns`.")

if __name__ == "__main__":
    main()
