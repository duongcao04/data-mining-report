import joblib
import os
import pandas as pd

# --- CẤU HÌNH ĐƯỜNG DẪN ĐỘNG ---
# Đảm bảo tìm thấy artifacts dù chạy script từ đâu
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(CURRENT_DIR, '..', 'artifacts')
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'best_model.joblib')

class ChurnPredictor:
    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load model pipeline đã train từ disk"""
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
            print(f"Đã load model thành công từ: {MODEL_PATH}")
        else:
            print(f"CẢNH BÁO: Không tìm thấy model tại {MODEL_PATH}. Hãy chạy src/modeling.py trước.")
            self.model = None

    def predict_one(self, data_dict):
        """Dự đoán cho 1 mẫu dữ liệu"""
        if not self.model:
            self._load_model()
            if not self.model:
                raise RuntimeError("Model chưa được huấn luyện hoặc không tìm thấy file model.")

        df = pd.DataFrame([data_dict])
        
        prediction = self.model.predict(df)[0]
        probability = self.model.predict_proba(df)[0][1]
        
        return {
            "prediction": int(prediction),
            "churn_probability": float(probability),
            "label": "Churn" if prediction == 1 else "No Churn"
        }

    def predict_batch(self, data_list):
        """Dự đoán cho danh sách mẫu dữ liệu"""
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
    sample = {
        "Age": 30, "Gender": "Female", "Tenure": 12, 
        "Usage Frequency": 5, "Support Calls": 2, "Payment Delay": 0,
        "Subscription Type": "Basic", "Contract Length": "Monthly",
        "Total Spend": 500, "Last Interaction": 5
    }
    try:
        print(predictor.predict_one(sample))
    except Exception as e:
        print(f"Lỗi: {e}")