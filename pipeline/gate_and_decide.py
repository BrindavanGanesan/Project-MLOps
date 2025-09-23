# pipeline/gate_and_decide.py
import os, json, boto3, sys, yaml

CFG = "config/config.yaml"
EVAL_KEY = "evaluation/evaluation.json"   # adjust if you wrote to a subfolder

def load_cfg():
    with open(CFG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main(threshold=0.90):
    cfg = load_cfg()
    s3 = boto3.client("s3", region_name=cfg["aws"]["region"])
    obj = s3.get_object(Bucket=cfg["aws"]["bucket"], Key=EVAL_KEY)
    metrics = json.loads(obj["Body"].read())
    print("Metrics:", metrics)

    passed = metrics.get("f1_macro", 0.0) >= threshold
    print(f"Gate @ f1_macro>={threshold}: {'PASS ✅' if passed else 'FAIL ❌'}")
    sys.exit(0 if passed else 2)

if __name__ == "__main__":
    # optional: python gate_and_decide.py 0.92
    import sys
    thr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.90
    main(threshold=thr)
