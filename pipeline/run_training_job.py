import os
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
        role=role_arn,
        framework_version=cfg["training"]["framework_version"],  
        instance_type=cfg["training"]["instance_type"],          
        instance_count=1,
        py_version="py3",
        source_dir="training",             # uploads your code to S3 automatically
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


if __name__ == "__main__":
    main()
