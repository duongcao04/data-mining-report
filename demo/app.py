from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pydantic import BaseModel
import sys
import os
import json
from datetime import datetime

# Đảm bảo log tiếng Việt không gây UnicodeEncodeError trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Thêm thư mục src vào system path để import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import thêm hàm get_business_analytics và load_data
from src.preprocessing import run_preprocessing, get_business_analytics, load_data, get_eda_stats
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
    customerID: Optional[str] = None
    tenure: int
    MonthlyCharges: float
    Contract: str
    InternetService: str
    PaymentMethod: str

    class Config:
        json_schema_extra = {
            "example": {
                "tenure": 12,
                "MonthlyCharges": 70.35,
                "Contract": "Month-to-month",
                "InternetService": "DSL",
                "PaymentMethod": "Electronic check"
            }
        }

# --- Endpoints ---

@app.get("/")
def root():
    return {"message": "Welcome to Churn Prediction API. Visit /docs for Swagger UI."}

@app.get("/status")
def get_status():
    """Kiểm tra trạng thái mô hình và hệ thống"""
    model_exists = os.path.exists('models/model.pkl')
    return {
        "status": "active",
        "model_trained": model_exists,
        "model_path": os.path.abspath('models/model.pkl'),
        "models_dir": os.path.abspath('models')
    }

@app.get("/eda")
def get_eda():
    """
    CRISP-DM: Data Understanding
    Trả về kết quả phân tích dữ liệu dạng JSON.
    """
    try:
        stats = get_eda_stats()
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

@app.get("/train/status")
def get_train_status():
    """
    Trả về kết quả training mới nhất đã lưu trên ổ đĩa.
    """
    results_path = os.path.join('models', 'evaluation_results.json')
    if not os.path.exists(results_path):
        raise HTTPException(status_code=404, detail="Chưa có kết quả training. Bấm Train Model để bắt đầu.")

    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        last_updated = datetime.fromtimestamp(os.path.getmtime(results_path)).isoformat()
        return {"status": "ok", "last_updated": last_updated, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Không đọc được kết quả training: {str(e)}")

@app.post("/predict")
def predict_churn(data: CustomerData):
    """
    CRISP-DM: Deployment
    """
    try:
        input_data = data.dict()
        result = predictor.predict_one(input_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")