import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- CẤU HÌNH ĐƯỜNG DẪN ĐỘNG (DYNAMIC PATHS) ---
# Lấy đường dẫn tuyệt đối của thư mục chứa file này (src/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Đi ra thư mục cha (..) rồi vào folder 'data'
DATA_PATH = os.path.join(CURRENT_DIR, '..', 'data', 'customer_churn_dataset-testing-master.csv')

# Đi ra thư mục cha (..) rồi vào folder 'artifacts'
ARTIFACTS_DIR = os.path.join(CURRENT_DIR, '..', 'artifacts')

# Tạo thư mục artifacts nếu chưa tồn tại
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def load_data(path=DATA_PATH):
    """
    CRISP-DM: Data Understanding
    Tải dữ liệu từ file CSV.
    """
    if not os.path.exists(path):
        # Fallback: Nếu không thấy file ở đường dẫn tính toán, thử tìm ở đường dẫn gốc user cung cấp
        fallback_path = '/mnt/data/customer_churn_dataset-testing-master.csv'
        if os.path.exists(fallback_path):
            print(f"Không tìm thấy data tại {path}, dùng fallback: {fallback_path}")
            path = fallback_path
        else:
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {path} hoặc {fallback_path}")
    
    df = pd.read_csv(path)
    print(f"Đã tải dữ liệu: {df.shape}")
    return df

def perform_eda(df):
    """
    CRISP-DM: Data Understanding
    Thực hiện phân tích khám phá dữ liệu (EDA).
    """
    eda_report = {
        "shape": df.shape,
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "description": df.describe().to_dict(),
        "target_distribution": df['Churn'].value_counts(normalize=True).to_dict(),
    }
    
    # Chỉ tính correlation cho các cột số
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        eda_report["correlation"] = numeric_df.corr().to_dict()
    
    print("Đã hoàn thành EDA.")
    return eda_report

def build_preprocessor(df):
    """
    CRISP-DM: Data Preparation
    Xây dựng pipeline tiền xử lý.
    """
    target = 'Churn'
    if target in df.columns:
        X = df.drop(columns=[target, 'CustomerID'], errors='ignore')
        y = df[target]
    else:
        X = df.drop(columns=['CustomerID'], errors='ignore')
        y = None

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    print(f"Features số: {list(numeric_features)}")
    print(f"Features phân loại: {list(categorical_features)}")

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor, X, y

def run_preprocessing():
    """Hàm chính chạy pipeline"""
    df = load_data()
    eda_stats = perform_eda(df)
    preprocessor, X, y = build_preprocessor(df)
    preprocessor.fit(X)
    
    save_path = os.path.join(ARTIFACTS_DIR, 'preprocessor.joblib')
    joblib.dump(preprocessor, save_path)
    print(f"Đã lưu preprocessor vào {save_path}")
    
    return eda_stats

if __name__ == "__main__":
    run_preprocessing()