# inference/inference.py
# Compatible with SageMaker SKLearn container script mode
import os
import joblib
import numpy as np

def model_fn(model_dir):
    # SageMaker downloads model.tar.gz and extracts to /opt/ml/model
    model_path = os.path.join(model_dir, "model.joblib")
    return joblib.load(model_path)

def input_fn(request_body, content_type="application/json"):
    # Accept JSON: {"instances": [[...feature vector...], ...]}
    # or {"inputs":[...]} for single row
    import json
    data = json.loads(request_body)
    if "instances" in data:
        arr = np.array(data["instances"])
    elif "inputs" in data:
        arr = np.array([data["inputs"]], dtype=float)
    else:
        # fall back: a flat list represents a single instance
        if isinstance(data, list):
            arr = np.array([data], dtype=float)
        else:
            raise ValueError("Unsupported input format.")
    return arr

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, accept="application/json"):
    import json
    return json.dumps({"predictions": prediction.tolist()}), accept
