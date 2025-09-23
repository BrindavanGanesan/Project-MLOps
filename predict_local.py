import joblib, numpy as np
m = joblib.load("model.joblib")
sample = np.array([[5.1,3.5,1.4,0.2]])
print("pred:", m.predict(sample).tolist())