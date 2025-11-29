import joblib
import os
import sys
import pandas as pd

# --- CẤU HÌNH ĐƯỜNG DẪN ĐỘNG ---
# Đảm bảo tìm thấy model dù chạy script từ đâu
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, '..', 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'model.pkl')

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
        if 'customerID' in df.columns:
            df = df.drop(columns=['customerID'])
        
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
        if 'customerID' in df.columns:
            df = df.drop(columns=['customerID'])
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
        "tenure": 12,
        "InternetService": "DSL",
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35
    }
    try:
        print(predictor.predict_one(sample))
    except Exception as e:
        print(f"Lỗi: {e}")