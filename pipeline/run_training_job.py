import os, io, time, tarfile, boto3, botocore
import yaml
import boto3
from datetime import datetime
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.session import Session


CFG_PATH = os.path.join("config", "config.yaml")


def load_cfg(path=CFG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_cfg()
    region = cfg["aws"]["region"]
    role_arn = cfg["aws"]["role_arn"]
    bucket = cfg["aws"]["bucket"]

    # optional sanity checks
    assert role_arn.startswith("arn:aws:iam::"), "role_arn looks invalid"
    assert bucket, "bucket must be set in config.yaml"

    # create a boto3 + sagemaker session pinned to your region
    boto_sess = boto3.Session(region_name=region)
    sm_session = Session(boto_session=boto_sess, default_bucket=bucket)

    # unique job name (helps you find it in console)
    job_suffix = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    base_job_name = f"thesis-train-{job_suffix}"

    # Set use_spot from config or default to False
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
        hyperparameters={
            "test-size": cfg["evaluation"]["test_size"],  # passed into train.py
        },
        enable_sagemaker_metrics=True,
        use_spot_instances=use_spot,
        **({"max_run": 3600, "max_wait": 7200} if use_spot else {})
    )

    print(f"Starting training job in {region} with role {role_arn} …")
    est.fit(wait=True)  # block until the job completes

    print("\n✅ Training job finished.")
    print("Job name:", est.latest_training_job.name)
    print("Model artifact S3:", est.model_data)  # s3 path to model.tar.gz
    print("\nNext: you can open SageMaker → Training jobs to inspect logs/metrics.")

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

    artifact_uri = est.model_data  # s3://.../<job>/output/model.tar.gz
    bucket, model_key = parse_s3_uri(artifact_uri)
    output_tar_key = model_key.replace("/output/model.tar.gz", "/output/output.tar.gz")

    s3 = boto3.client("s3", region_name=region)
    print("Looking for training outputs tar:", f"s3://{bucket}/{output_tar_key}")
    if not wait_for_key(s3, bucket, output_tar_key):
        raise RuntimeError(f"output.tar.gz not found yet at s3://{bucket}/{output_tar_key}")

    obj = s3.get_object(Bucket=bucket, Key=output_tar_key)
    buf = io.BytesIO(obj["Body"].read())

    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        members = [m.name for m in tar.getmembers()]
        target = "data/metrics.json"
        if target not in members:
        # fall back to any metrics.json inside the tar
            cand = [n for n in members if n.endswith("/metrics.json") or n == "metrics.json"]
            if not cand:
                raise FileNotFoundError(f"metrics.json not in output.tar.gz (members sample: {members[:15]})")
            target = cand[0]
        fobj = tar.extractfile(target)
        metrics_bytes = fobj.read()

    local_metrics_dir = os.path.join("artifacts", "metrics")
    os.makedirs(local_metrics_dir, exist_ok=True)
    local_metrics_path = os.path.join(local_metrics_dir, "metrics.json")
    with open(local_metrics_path, "wb") as out:
        out.write(metrics_bytes)

    print("✅ Saved local metrics for DVC:", local_metrics_path)


if __name__ == "__main__":
    main()
