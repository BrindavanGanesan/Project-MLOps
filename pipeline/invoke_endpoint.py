# pipeline/invoke_endpoint.py
import json, boto3, yaml

CFG = "config/config.yaml"
ENDPOINT_NAME = "thesis-iris-serverless"

def load_cfg():
    with open(CFG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_cfg()
    smrt = boto3.client("sagemaker-runtime", region_name=cfg["aws"]["region"])

    payload = {"instances": [
        [5.1, 3.5, 1.4, 0.2],  # example Iris measurements
        [6.2, 2.9, 4.3, 1.3]
    ]}
    resp = smrt.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(payload)
    )
    print("Predictions:", json.loads(resp["Body"].read().decode("utf-8")))

if __name__ == "__main__":
    main()
