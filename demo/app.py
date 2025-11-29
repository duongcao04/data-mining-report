from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
<<<<<<< Updated upstream
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
=======
from pydantic import BaseModel, create_model
from typing import Optional, Dict, Any
import sys
import os
import json
import numpy as np
import pandas as pd
>>>>>>> Stashed changes

# Thêm thư mục src vào system path để import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

<<<<<<< Updated upstream
# Import thêm hàm get_business_analytics và load_data
from src.preprocessing import run_preprocessing, get_business_analytics, load_data, get_eda_stats
=======
from src.preprocessing import run_preprocessing, load_data, build_preprocessor
>>>>>>> Stashed changes
from src.modeling import train_and_evaluate
from src.predict import ChurnPredictor

def convert_to_json_serializable(obj):
    """Convert numpy/pandas types to native Python types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    return obj

def get_feature_columns():
    """Lấy danh sách các features từ dữ liệu thực tế"""
    try:
        df = load_data()
        preprocessor, X, y = build_preprocessor(df)
        return list(X.columns)
    except Exception as e:
        print(f"Warning: Khong the load du lieu: {e}")
        return None

# Lấy danh sách features từ dữ liệu thực tế
feature_columns = None  # sẽ được load lazy

def ensure_feature_columns(force_reload: bool = False):
    """Đảm bảo feature_columns đã được load."""
    global feature_columns
    if feature_columns and not force_reload:
        return feature_columns
    features = get_feature_columns()
    feature_columns = features
    return features

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API triển khai mô hình dự đoán rời bỏ theo quy trình CRISP-DM",
    version="1.0"
)

<<<<<<< Updated upstream
# --- CẤU HÌNH CORS ---
# Cho phép frontend gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production nên giới hạn domain cụ thể
=======
# Thêm CORS middleware để cho phép web demo kết nối
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả origins (trong production nên giới hạn)
>>>>>>> Stashed changes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo predictor
predictor = ChurnPredictor()

<<<<<<< Updated upstream
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

=======
>>>>>>> Stashed changes
# --- Endpoints ---

@app.get("/")
def root():
    features = ensure_feature_columns()
    return {
        "message": "Welcome to Churn Prediction API. Visit /docs for Swagger UI.",
        "features": features if features else "Loading..."
    }

@app.get("/status")
def get_status():
    """Kiểm tra trạng thái mô hình và hệ thống"""
<<<<<<< Updated upstream
    model_exists = os.path.exists('models/model.pkl')
    return {
        "status": "active",
        "model_trained": model_exists,
        "model_path": os.path.abspath('models/model.pkl'),
        "models_dir": os.path.abspath('models')
=======
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(BASE_DIR, 'models')
    model_path = os.path.join(models_dir, 'best_model.joblib')
    model_exists = os.path.exists(model_path)
    
    features = ensure_feature_columns()
    return {
        "status": "active",
        "model_trained": model_exists,
        "models_dir": models_dir,
        "features": features if features else []
>>>>>>> Stashed changes
    }

@app.get("/features")
def get_features():
    """Lấy danh sách các features cần thiết cho prediction"""
    features = ensure_feature_columns()
    if features:
        try:
            df = load_data()
            preprocessor, X, y = build_preprocessor(df)
            features_info = {}
            for col in X.columns:
                dtype = str(X[col].dtype)
                if dtype in ['int64', 'int32']:
                    features_info[col] = {
                        "type": "integer",
                        "sample_values": X[col].unique()[:5].tolist() if X[col].dtype == 'int64' else []
                    }
                elif dtype in ['float64', 'float32']:
                    features_info[col] = {
                        "type": "float",
                        "min": float(X[col].min()),
                        "max": float(X[col].max()),
                        "mean": float(X[col].mean())
                    }
                else:
                    features_info[col] = {
                        "type": "string",
                        "unique_values": X[col].unique().tolist()[:10]
                    }
            
            return {
                "features": feature_columns,
                "features_info": features_info
            }
        except Exception as e:
            return {
                "features": feature_columns,
                "error": str(e)
            }
    else:
        return {
            "error": "Features chua duoc load",
            "suggestion": "Chay python src/preprocessing.py (neu chua) hoac dam bao dataset ton tai."
        }

@app.post("/features/reload")
def reload_features():
    """Reload danh sách features"""
    features = ensure_feature_columns(force_reload=True)
    if features:
        return {"status": "reloaded", "features": features}
    return {"status": "error", "message": "Khong the load features. Kiem tra log server."}

@app.get("/eda")
def get_eda():
    """
    CRISP-DM: Data Understanding
    Trả về kết quả phân tích dữ liệu dạng JSON.
    """
    try:
<<<<<<< Updated upstream
        stats = get_eda_stats()
        # SỬA LỖI: Trả về dict trực tiếp, không ép kiểu str() nữa
        return {"status": "success", "eda_stats": stats} 
=======
        # Chạy lại EDA hoặc load kết quả đã lưu (ở đây chạy trực tiếp cho demo)
        stats = run_preprocessing()
        # Convert numpy/pandas types to JSON serializable format
        serializable_stats = convert_to_json_serializable(stats)
        return {"status": "success", "eda_stats": serializable_stats} 
>>>>>>> Stashed changes
    except Exception as e:
        # Log lỗi ra console server để dễ debug
        print(f"Error in /eda: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

<<<<<<< Updated upstream
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
=======
# Biến để track training status
training_status = {"is_training": False, "progress": None}
>>>>>>> Stashed changes

@app.post("/train")
def train_model(background_tasks: BackgroundTasks):
    """
    CRISP-DM: Modeling
    """
<<<<<<< Updated upstream
    try:
        results = train_and_evaluate()
        global predictor
        predictor = ChurnPredictor()
        return {"status": "Training completed", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
=======
    global training_status
    
    if training_status["is_training"]:
        return {"status": "Training already in progress", "message": "Vui long doi training hoan tat"}
    
    def run_training():
        """Hàm chạy training trong background"""
        global training_status, predictor
        try:
            training_status["is_training"] = True
            training_status["progress"] = "Starting..."
            
            # Chạy training
            results = train_and_evaluate()
            
            # Reload lại predictor sau khi train xong
            predictor = ChurnPredictor()
            
            training_status["progress"] = "Completed"
            training_status["is_training"] = False
            
            return results
        except Exception as e:
            training_status["progress"] = f"Error: {str(e)}"
            training_status["is_training"] = False
            raise e
    
    # Thêm task vào background
    background_tasks.add_task(run_training)
    
    return {
        "status": "Training started",
        "message": "Training dang chay trong background. API van hoat dong binh thuong. Kiem tra /train/status de xem tien do."
    }

@app.get("/train/status")
def get_training_status():
    """Kiểm tra trạng thái training"""
    return training_status
>>>>>>> Stashed changes

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
def predict_churn(data: Dict[str, Any]):
    """
    CRISP-DM: Deployment
<<<<<<< Updated upstream
    """
    try:
        input_data = data.dict()
        result = predictor.predict_one(input_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
=======
    Dự đoán cho dữ liệu đầu vào.
    Nhận dữ liệu dạng JSON với các keys khớp với tên cột trong dataset.
    
    Ví dụ request body:
    {
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
    }
    """
    try:
        if not predictor.model:
            raise HTTPException(status_code=400, detail="Model chua duoc train! Hay chay /train truoc.")
        
        # Kiểm tra xem có đủ features không
        if feature_columns:
            missing_features = set(feature_columns) - set(data.keys())
            if missing_features:
                return {
                    "error": "Thieu cac features sau",
                    "missing": list(missing_features),
                    "required": feature_columns,
                    "received": list(data.keys())
                }
        
        # Dự đoán
        result = predictor.predict_one(data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Để chạy server: uvicorn demo.app:app --reload
>>>>>>> Stashed changes
