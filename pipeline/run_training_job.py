# pipeline/run_training_job.py

import os, io, time, tarfile, shutil, boto3, botocore
import yaml
from pathlib import Path
from datetime import datetime
from sagemaker.session import Session
from sagemaker.estimator import Estimator


CFG_PATH = "config/config.yaml"


def load_cfg(path=CFG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_s3_uri(uri):
    assert uri.startswith("s3://")
    bucket, key = uri[5:].split("/", 1)
    return bucket, key


def wait_for_key(s3, bucket, key, retries=20, sleep_s=3):
    for _ in range(retries):
        try:
            s3.head_object(Bucket=bucket, Key=key)
            return True
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                time.sleep(sleep_s)
                continue
            raise
    return False


def mkdir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def copy_tree(src, dst):
    for root, _, files in os.walk(src):
        rel = Path(root).relative_to(src)
        d = Path(dst) / rel
        d.mkdir(parents=True, exist_ok=True)
        for file in files:
            shutil.copy2(Path(root) / file, d / file)


def main():

    cfg = load_cfg()
    region   = cfg["aws"]["region"]
    role_arn = cfg["aws"]["role_arn"]
    bucket   = cfg["aws"]["bucket"]
    image_uri = cfg["training"]["image_uri"]

    boto_sess  = boto3.Session(region_name=region)
    sm_session = Session(boto_session=boto_sess)

    # Use UTC timestamp for deterministic job naming
    job_suffix = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    base_job_name = f"thesis-training-{job_suffix}"

    # BYOC container: NO entry_point
    est = Estimator(
        image_uri=image_uri,
        role=role_arn,
        instance_type="ml.m5.large",
        instance_count=1,
        base_job_name=base_job_name,
        sagemaker_session=sm_session,
        hyperparameters={
            "test-size": cfg["training"]["test_size"],
            "random-state": cfg["training"]["random_state"],
        },
        output_path=f"s3://{bucket}/training-output/",
    )

    # Point SageMaker to training data
    train_s3_uri = f"s3://{bucket}/data/adult.csv"

    print(f"📡 Using training data from: {train_s3_uri}")
    print(f"🚀 Starting training job: {base_job_name}")

    # IMPORTANT: Only ONE fit() call
    est.fit(
        inputs={"training": train_s3_uri},
        wait=True
    )

    # ------------------------------
    # Download training artifacts
    # ------------------------------

    model_s3 = est.model_data
    bucket_name, model_key = parse_s3_uri(model_s3)

    output_tar_key = model_key.replace("model.tar.gz", "output.tar.gz")

    s3 = boto3.client("s3", region_name=region)

    print(f"⏳ Waiting for output.tar.gz at: s3://{bucket_name}/{output_tar_key}")

    if not wait_for_key(s3, bucket_name, output_tar_key):
        raise RuntimeError("output.tar.gz missing from SageMaker output")

    data_bytes = s3.get_object(Bucket=bucket_name, Key=output_tar_key)["Body"].read()

    run_dir = mkdir(f"artifacts/sm-output/{base_job_name}")
    output_tar = run_dir / "output.tar.gz"
    output_tar.write_bytes(data_bytes)

    extracted_dir = mkdir(run_dir / "extracted")
    with tarfile.open(fileobj=io.BytesIO(data_bytes), mode="r:gz") as tar:
        tar.extractall(extracted_dir)

    # metrics.json → DVC
    metrics_json = next(extracted_dir.rglob("metrics.json"))
    shutil.copy2(metrics_json, mkdir("artifacts/metrics") / "metrics.json")

    # model.tar.gz → DVC
    mkdir("artifacts/model")
    s3.download_file(bucket_name, model_key, "artifacts/model/model.tar.gz")

    # MLflow runs
    mlruns = list(extracted_dir.rglob("mlruns"))
    if mlruns:
        copy_tree(mlruns[0], mkdir("mlruns"))
        print("📂 Copied MLflow runs")
    else:
        print("ℹ️ No mlruns folder found")

    print("✅ Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
