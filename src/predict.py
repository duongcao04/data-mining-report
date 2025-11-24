import joblib
import os
import pandas as pd

ARTIFACTS_DIR = 'artifacts'
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'best_model.joblib')

class ChurnPredictor:
    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load model pipeline đã train từ disk"""
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
            print("Đã load model thành công.")
        else:
            print(f"Cảnh báo: Không tìm thấy model tại {MODEL_PATH}. Hãy chạy train trước.")
            self.model = None

    def predict_one(self, data_dict):
        """
        Dự đoán cho 1 khách hàng (input là dictionary)
        """
        if not self.model:
            self._load_model()
            if not self.model:
                raise Exception("Model chưa được train!")

        # Chuyển dict thành DataFrame (1 dòng)
        df = pd.DataFrame([data_dict])
        
        # Dự đoán
        prediction = self.model.predict(df)[0]
        probability = self.model.predict_proba(df)[0][1]
        
        return {
            "prediction": int(prediction),
            "churn_probability": float(probability),
            "label": "Churn" if prediction == 1 else "No Churn"
        }

    def predict_batch(self, data_list):
        """
        Dự đoán cho nhiều khách hàng
        """
        if not self.model:
            self._load_model()
        
        df = pd.DataFrame(data_list)
        predictions = self.model.predict(df)
        probabilities = self.model.predict_proba(df)[:, 1]
        
        results = []
        for pred, prob in zip(predictions, probabilities):
            results.append({
                "prediction": int(pred),
                "churn_probability": float(prob),
                "label": "Churn" if pred == 1 else "No Churn"
            })
        return results

if __name__ == "__main__":
    # Test nhanh
    predictor = ChurnPredictor()
    # Sample data khớp với columns trong CSV
    sample = {
        "Age": 30, "Gender": "Female", "Tenure": 12, 
        "Usage Frequency": 5, "Support Calls": 2, "Payment Delay": 0,
        "Subscription Type": "Basic", "Contract Length": "Monthly",
        "Total Spend": 500, "Last Interaction": 5
    }
    try:
        print(predictor.predict_one(sample))
    except Exception as e:
        print(e)