import boto3, yaml

def load_cfg():
    with open("config/config.yaml","r",encoding="utf-8") as f:
        import yaml; return yaml.safe_load(f)

def main():
    cfg = load_cfg()
    sm = boto3.client("sagemaker", region_name=cfg["aws"]["region"])
    ep = "thesis-iris-serverless"
    print("Deleting endpoint:", ep)
    sm.delete_endpoint(EndpointName=ep)

if __name__ == "__main__":
    main()