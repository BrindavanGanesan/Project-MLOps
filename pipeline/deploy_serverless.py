# pipeline/deploy_serverless.py
import os, boto3, yaml
from sagemaker.session import Session
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.serverless import ServerlessInferenceConfig

CFG = "config/config.yaml"
# Use the exact model.tar.gz S3 URI from your *successful training job*
MODEL_ARTIFACT_S3 = "s3://thebrowntiger/thesis-train-20250919-173307-2025-09-19-17-33-07-901/output/model.tar.gz"

ENDPOINT_NAME = "thesis-iris-serverless"   # change if you like

def load_cfg():
    with open(CFG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_cfg()
    region, role = cfg["aws"]["region"], cfg["aws"]["role_arn"]
    boto_sess = boto3.Session(region_name=region)
    sm_sess = Session(boto_session=boto_sess)

    model = SKLearnModel(
        model_data=MODEL_ARTIFACT_S3,
        role=role,
        entry_point="inference.py",
        source_dir="inference",
        framework_version=cfg["training"]["framework_version"],  # "1.2-1" to match training
        sagemaker_session=sm_sess,
        py_version="py3",
        name=f"{ENDPOINT_NAME}-model",
    )

    serverless_cfg = ServerlessInferenceConfig(
        memory_size_in_mb=2048,      # 1024/2048/4096/6144
        max_concurrency=5            # adjust per expected load
    )

    predictor = model.deploy(
        serverless_inference_config=serverless_cfg,
        endpoint_name=ENDPOINT_NAME
    )
    print("✅ Deployed endpoint:", predictor.endpoint_name)

if __name__ == "__main__":
    main()
