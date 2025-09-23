from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
import os, yaml, boto3
from sagemaker.session import Session


CFG = "config/config.yaml"
MODEL_ARTIFACT_S3 = "s3://thebrowntiger/thesis-train-20250919-173307-2025-09-19-17-33-07-901/output/model.tar.gz"

def load_cfg():
    with open(CFG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_cfg()
    region, bucket, role = cfg["aws"]["region"], cfg["aws"]["bucket"], cfg["aws"]["role_arn"]
    boto_sess = boto3.Session(region_name=region)
    sm_sess = Session(boto_session=boto_sess, default_bucket=bucket)

    processor = SKLearnProcessor(
        framework_version=cfg["training"]["framework_version"],  # e.g. "1.2-1"
        role=role,
        instance_type="ml.t3.large",
        instance_count=1,
        sagemaker_session=sm_sess,
    )

    processor.run(
    code="evaluation/evaluate.py",   # <— point directly to the script
    inputs=[
        ProcessingInput(
            source=MODEL_ARTIFACT_S3,
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/evaluation",
            destination=f"s3://{bucket}/evaluation/"
        )
    ],
    arguments=[
        "--model", "/opt/ml/processing/input/model.tar.gz"
    ],
    wait=True,
    logs=True,
)

    print("✅ Evaluation job submitted. Check S3:", f"s3://{bucket}/evaluation/evaluation.json")



if __name__ == "__main__":
    main()
