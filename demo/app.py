from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import sys
import os
import json

# Thêm thư mục src vào system path để import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import run_preprocessing
from src.modeling import train_and_evaluate
from src.predict import ChurnPredictor

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API triển khai mô hình dự đoán rời bỏ theo quy trình CRISP-DM",
    version="1.0"
)

# Khởi tạo predictor
predictor = ChurnPredictor()

# --- Pydantic Models (Schema Validation) ---
class CustomerData(BaseModel):
    Age: int
    Gender: str
    Tenure: int
    Usage_Frequency: int
    Support_Calls: int
    Payment_Delay: int
    Subscription_Type: str
    Contract_Length: str
    Total_Spend: float
    Last_Interaction: int

    class Config:
        # Mapping field names if needed (Pydantic uses underscores, CSV uses spaces)
        populate_by_name = True 
        json_schema_extra = {
            "example": {
                "Age": 40,
                "Gender": "Male",
                "Tenure": 24,
                "Usage Frequency": 15,
                "Support Calls": 1,
                "Payment Delay": 5,
                "Subscription Type": "Standard",
                "Contract Length": "Annual",
                "Total Spend": 1200.5,
                "Last Interaction": 10
            }
        }

# --- Endpoints ---

@app.get("/")
def root():
    return {"message": "Welcome to Churn Prediction API. Visit /docs for Swagger UI."}

@app.get("/status")
def get_status():
    """Kiểm tra trạng thái mô hình và hệ thống"""
    model_exists = os.path.exists('artifacts/best_model.joblib')
    return {
        "status": "active",
        "model_trained": model_exists,
        "artifacts_dir": os.path.abspath('artifacts')
    }

@app.get("/eda")
def get_eda():
    """
    CRISP-DM: Data Understanding
    Trả về kết quả phân tích dữ liệu.
    """
    try:
        # Chạy lại EDA hoặc load kết quả đã lưu (ở đây chạy trực tiếp cho demo)
        stats = run_preprocessing()
        # Convert key numpy int64 to int for JSON serialization
        return {"status": "success", "eda_stats": str(stats)} 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
def train_model(background_tasks: BackgroundTasks):
    """
    CRISP-DM: Modeling
    Kích hoạt quy trình huấn luyện (chạy background để không block API).
    """
    try:
        # Chạy training ngay lập tức (trong thực tế nên dùng Celery)
        results = train_and_evaluate()
        # Reload lại predictor sau khi train xong
        global predictor
        predictor = ChurnPredictor()
        return {"status": "Training completed", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict_churn(data: CustomerData):
    """
    CRISP-DM: Deployment
    Dự đoán cho dữ liệu đầu vào.
    """
    try:
        # Convert Pydantic model to dict, handle alias (Usage_Frequency -> Usage Frequency)
        input_data = data.dict()
        # Map lại keys nếu cần thiết để khớp với CSV columns
        mapped_data = {k.replace('_', ' '): v for k, v in input_data.items()}
        
        result = predictor.predict_one(mapped_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Để chạy server: uvicorn api.app:app --reload