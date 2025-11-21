import requests
import random
import time
from concurrent.futures import ThreadPoolExecutor

PREDICT_URL = "http://iris-nlb-a5756f62e3cf9fb9.elb.eu-west-1.amazonaws.com/predict"

# ---- Sample categorical values (from Adult dataset) ----
workclass_vals = [
    "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
    "State-gov", "Local-gov"
]

education_vals = [
    "HS-grad", "Bachelors", "Some-college", "Masters", "Assoc-voc",
    "Assoc-acdm", "Doctorate"
]

marital_vals = [
    "Never-married", "Married-civ-spouse", "Divorced",
    "Separated", "Widowed"
]

occupation_vals = [
    "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
    "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
    "Adm-clerical", "Transport-moving"
]

relationship_vals = [
    "Husband", "Not-in-family", "Own-child", "Unmarried", "Wife"
]

race_vals = [
    "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"
]

sex_vals = ["Male", "Female"]

country_vals = [
    "United-States", "Mexico", "Cuba", "Jamaica", "India",
    "China", "Philippines", "Canada", "Germany", "Japan",
    "Vietnam", "Thailand", "Ireland", "Dominican-Republic"
]

# ---- Generate one random record with correct aliases ----
def make_record():
    return {
        "Age": random.randint(18, 70),
        "Workclass": random.choice(workclass_vals),
        "fnlwgt": random.randint(50000, 300000),
        "Education": random.choice(education_vals),
        "Education_Num": random.randint(8, 16),
        "Martial_Status": random.choice(marital_vals),
        "Occupation": random.choice(occupation_vals),
        "Relationship": random.choice(relationship_vals),
        "Race": random.choice(race_vals),
        "Sex": random.choice(sex_vals),
        "Capital_Gain": random.choice([0, 0, 0, random.randint(500, 25000)]),  # mostly zero
        "Capital_Loss": random.choice([0, 0, random.randint(500, 2500)]),
        "Hours_per_week": random.randint(10, 80),
        "Country": random.choice(country_vals)
    }


# ---- Send prediction
def send_prediction(n):
    payload = { "records": [make_record()] }

    try:
        r = requests.post(PREDICT_URL, json=payload, timeout=5)
        if r.status_code == 200:
            print(f"[{n}] OK — Predicted: {r.json()['predictions']}")
        else:
            print(f"[{n}] ERROR: {r.status_code} -> {r.text}")
    except Exception as e:
        print(f"[{n}] EXCEPTION: {e}")


# ---- Fire 500 predictions with 20 threads ----
def run_load_test(total=500):
    print(f"🚀 Starting load… sending {total} predictions")

    with ThreadPoolExecutor(max_workers=20) as executor:
        for i in range(total):
            executor.submit(send_prediction, i+1)
            time.sleep(0.05)   # slight delay to avoid NLB surge protection

    print("✅ Load test complete — check Grafana + Prometheus!")


if __name__ == "__main__":
    run_load_test(500)
