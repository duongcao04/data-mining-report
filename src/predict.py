# src/predict.py

import joblib
import pandas as pd

MODEL_PATH = "models/model.pkl"

def load_model():
    return joblib.load(MODEL_PATH)


def predict_single(model, customer_dict):
    df = pd.DataFrame([customer_dict])
    prob = model.predict_proba(df)[0, 1]
    pred = int(model.predict(df)[0])
    return pred, prob
