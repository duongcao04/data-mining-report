import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Cấu hình đường dẫn
DATA_PATH = '../data/customer_churn_dataset-testing-master.csv'
ARTIFACTS_DIR = 'artifacts'
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def load_data(path=DATA_PATH):
    """
    CRISP-DM: Data Understanding
    Tải dữ liệu từ file CSV.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {path}")
    
    df = pd.read_csv(path)
    print(f"Đã tải dữ liệu: {df.shape}")
    return df

def perform_eda(df):
    """
    CRISP-DM: Data Understanding
    Thực hiện phân tích khám phá dữ liệu (EDA) và trả về báo cáo dạng dictionary.
    """
    eda_report = {
        "shape": df.shape,
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "description": df.describe().to_dict(),
        "target_distribution": df['Churn'].value_counts(normalize=True).to_dict(),
    }
    
    # Tính Correlation Matrix chỉ cho các cột số
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().to_dict()
    eda_report["correlation"] = corr_matrix
    
    print("Đã hoàn thành EDA.")
    return eda_report

def build_preprocessor(df):
    """
    CRISP-DM: Data Preparation
    Xây dựng pipeline tiền xử lý: Imputer -> Scaler/Encoder.
    """
    # Xác định cột target và features
    target = 'Churn'
    if target in df.columns:
        X = df.drop(columns=[target, 'CustomerID']) # Bỏ CustomerID vì không có ý nghĩa dự đoán
        y = df[target]
    else:
        X = df.drop(columns=['CustomerID'], errors='ignore')
        y = None

    # Phân loại cột
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    print(f"Features số: {list(numeric_features)}")
    print(f"Features phân loại: {list(categorical_features)}")

    # Pipeline cho dữ liệu số: Điền khuyết bằng trung vị -> Chuẩn hóa
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline cho dữ liệu phân loại: Điền khuyết bằng 'missing' -> OneHotEncoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Kết hợp lại
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor, X, y

def run_preprocessing():
    """Hàm chạy chính cho bước tiền xử lý"""
    df = load_data()
    
    # Thực hiện EDA
    eda_stats = perform_eda(df)
    
    # Xây dựng preprocessor
    preprocessor, X, y = build_preprocessor(df)
    
    # Fit preprocessor trên toàn bộ dữ liệu (hoặc tập train nếu chia kỹ hơn ở bước này)
    preprocessor.fit(X)
    
    # Lưu preprocessor
    joblib.dump(preprocessor, os.path.join(ARTIFACTS_DIR, 'preprocessor.joblib'))
    print("Đã lưu preprocessor vào artifacts/preprocessor.joblib")
    
    return eda_stats

if __name__ == "__main__":
    run_preprocessing()