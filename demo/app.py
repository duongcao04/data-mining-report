from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
import json

# Thêm thư mục src vào system path để import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import thêm hàm get_business_analytics và load_data
from src.preprocessing import run_preprocessing, get_business_analytics, load_data
from src.modeling import train_and_evaluate
from src.predict import ChurnPredictor

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API triển khai mô hình dự đoán rời bỏ theo quy trình CRISP-DM",
    version="1.0"
)

# --- CẤU HÌNH CORS ---
# Cho phép frontend gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production nên giới hạn domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    Trả về kết quả phân tích dữ liệu dạng JSON.
    """
    try:
        # Chạy lại EDA
        stats = run_preprocessing()
        # SỬA LỖI: Trả về dict trực tiếp, không ép kiểu str() nữa
        return {"status": "success", "eda_stats": stats} 
    except Exception as e:
        # Log lỗi ra console server để dễ debug
        print(f"Error in /eda: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics")
def get_analytics():
    """
    Business Analytics Endpoint
    """
    try:
        df = load_data()
        analytics_data = get_business_analytics(df)
        return analytics_data
    except Exception as e:
        print(f"Error in /analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analytics Error: {str(e)}")

@app.post("/train")
def train_model(background_tasks: BackgroundTasks):
    """
    CRISP-DM: Modeling
    """
    try:
        results = train_and_evaluate()
        global predictor
        predictor = ChurnPredictor()
        return {"status": "Training completed", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict_churn(data: CustomerData):
    """
    CRISP-DM: Deployment
    """
    try:
        input_data = data.dict()
        mapped_data = {k.replace('_', ' '): v for k, v in input_data.items()}
        result = predictor.predict_one(mapped_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")