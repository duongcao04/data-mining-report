import joblib
import os
import sys
import pandas as pd

<<<<<<< Updated upstream
# --- CẤU HÌNH ĐƯỜNG DẪN ĐỘNG ---
# Đảm bảo tìm thấy model dù chạy script từ đâu
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, '..', 'models')
=======
# Fix encoding cho Windows console (chỉ khi chạy trực tiếp, không khi import)
if sys.platform == 'win32' and __name__ == '__main__':
    try:
        import io
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except:
        pass

# Cấu hình đường dẫn (tự động xác định dựa trên vị trí file)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
>>>>>>> Stashed changes
MODEL_PATH = os.path.join(MODELS_DIR, 'model.pkl')

class ChurnPredictor:
    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load model pipeline đã train từ disk"""
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
<<<<<<< Updated upstream
            print(f"Đã load model thành công từ: {MODEL_PATH}")
        else:
            print(f"CẢNH BÁO: Không tìm thấy model tại {MODEL_PATH}. Hãy chạy src/modeling.py trước.")
=======
            print("Da load model thanh cong.")
        else:
            print(f"Canh bao: Khong tim thay model tai {MODEL_PATH}. Hay chay train truoc.")
>>>>>>> Stashed changes
            self.model = None

    def predict_one(self, data_dict):
        """Dự đoán cho 1 mẫu dữ liệu"""
        if not self.model:
            self._load_model()
            if not self.model:
<<<<<<< Updated upstream
                raise RuntimeError("Model chưa được huấn luyện hoặc không tìm thấy file model.")
=======
                raise Exception("Model chua duoc train!")
>>>>>>> Stashed changes

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
<<<<<<< Updated upstream
    sample = {
        "tenure": 12,
        "InternetService": "DSL",
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35
=======
    # Sample data khớp với columns trong CSV thực tế
    sample = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": "29.85"
>>>>>>> Stashed changes
    }
    try:
        print(predictor.predict_one(sample))
    except Exception as e:
        print(f"Lỗi: {e}")