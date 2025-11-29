import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

<<<<<<< Updated upstream
# --- CẤU HÌNH ĐƯỜNG DẪN ĐỘNG ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, '..', 'data', 'Customer-Churn.csv')
MODELS_DIR = os.path.join(CURRENT_DIR, '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_COLUMNS = [
    'tenure',
    'MonthlyCharges',
    'Contract',
    'InternetService',
    'PaymentMethod'
]
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
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Customer-Churn.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
>>>>>>> Stashed changes

def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        fallback_path = '/mnt/data/Customer-Churn.csv'
        if os.path.exists(fallback_path):
            path = fallback_path
        else:
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {path}")

    df = pd.read_csv(path)
<<<<<<< Updated upstream
    df.columns = df.columns.str.strip()
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    if 'SeniorCitizen' in df.columns:
        df['SeniorCitizen'] = pd.to_numeric(df['SeniorCitizen'], errors='coerce')

=======
    print(f"Da tai du lieu: {df.shape}")
>>>>>>> Stashed changes
    return df

def perform_eda(df):
    """
    CRISP-DM: Data Understanding
    Trả về dữ liệu đã xử lý để vẽ biểu đồ EDA trên Dashboard.
    """
<<<<<<< Updated upstream
    # 1. Phân phối Target (Churn)
    target_dist = {}
    if 'Churn' in df.columns:
        target_dist = df['Churn'].value_counts(normalize=True).to_dict()
    
    # 2. Tương quan (Correlation) với Churn
    correlations = {}
    if 'Churn' in df.columns:
        churn_flag = df['Churn'].map({'Yes': 1, 'No': 0})
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        numeric_df['Churn_Flag'] = churn_flag
        numeric_df = numeric_df.dropna(subset=['Churn_Flag'])
        if not numeric_df.empty:
            corr_series = numeric_df.corr()['Churn_Flag'].drop('Churn_Flag').sort_values(ascending=False)
            correlations = {k: float(v) for k, v in corr_series.items()}

    # 3. Phân tích Categorical: Tỷ lệ Churn theo nhóm
    # Hàm helper để tính tỷ lệ churn
    def get_churn_rate_by_col(col_name):
        if col_name not in df.columns or 'Churn' not in df.columns:
            return {}
        grouped = df.groupby(col_name)['Churn'].apply(lambda x: (x == 'Yes').mean())
        return {str(k): float(v * 100) for k, v in grouped.items()}

    churn_by_contract = get_churn_rate_by_col('Contract')
    churn_by_subscription = get_churn_rate_by_col('InternetService')
    churn_by_payment = get_churn_rate_by_col('PaymentMethod')

    # 4. Phân phối biến số (Numerical Distribution) - Ví dụ: Tenure
    # Tạo histogram dữ liệu cho Tenure (chia làm 5 khoảng)
    tenure_hist = {}
    if 'tenure' in df.columns:
        tenure_values = df['tenure'].dropna()
        if len(tenure_values) > 0:
            counts, bin_edges = np.histogram(tenure_values, bins=5)
            for i in range(len(counts)):
                label = f"{int(bin_edges[i])}-{int(bin_edges[i+1])} tháng"
                tenure_hist[label] = int(counts[i])

    return {
        "summary": {
            "total_rows": int(df.shape[0]),
            "total_cols": int(df.shape[1]),
            "missing_values": int(df.isnull().sum().sum())
        },
        "target_distribution": target_dist,
        "correlations": correlations,
        "categorical_analysis": {
            "churn_by_contract": churn_by_contract,
            "churn_by_subscription": churn_by_subscription,
            "churn_by_payment": churn_by_payment
        },
        "numerical_distribution": {
            "tenure_distribution": tenure_hist
        }
    }

def get_business_analytics(df):
    """
    Tính toán chỉ số kinh doanh (Analytics)
    """
    total_revenue = float(df['TotalCharges'].sum())
    avg_revenue = float(df['MonthlyCharges'].mean())
    total_customers = int(len(df))
    churn_rate = float((df['Churn'] == 'Yes').mean() * 100)

    revenue_by_contract = df.groupby('Contract')['TotalCharges'].sum().to_dict()
    revenue_by_contract = {str(k): float(v) for k, v in revenue_by_contract.items()}

    customer_by_internet = df['InternetService'].value_counts().to_dict()
    customer_by_internet = {str(k): int(v) for k, v in customer_by_internet.items()}

    avg_charge_by_churn = df.groupby('Churn')['MonthlyCharges'].mean().to_dict()
    avg_charge_by_churn = {str(k): float(v) for k, v in avg_charge_by_churn.items()}

    return {
        "kpi": {
            "total_revenue": total_revenue,
            "avg_revenue_per_user": avg_revenue,
            "total_customers": total_customers,
            "churn_rate": churn_rate
        },
        "charts": {
            "revenue_by_contract": revenue_by_contract,
            "customer_by_internet": customer_by_internet,
            "avg_monthly_charge_by_churn": avg_charge_by_churn
        }
    }
=======
    # Convert shape tuple to list for JSON serialization
    eda_report = {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "missing_values": {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
        "description": {k: {k2: float(v2) for k2, v2 in v.items()} 
                       for k, v in df.describe().to_dict().items()},
        "target_distribution": {str(k): float(v) for k, v in df['Churn'].value_counts(normalize=True).to_dict().items()} if 'Churn' in df.columns else {},
    }
    
    # Tính Correlation Matrix chỉ cho các cột số
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = {k: {k2: float(v2) for k2, v2 in v.items()} 
                   for k, v in numeric_df.corr().to_dict().items()}
    eda_report["correlation"] = corr_matrix
    
    print("Da hoan thanh EDA.")
    return eda_report
>>>>>>> Stashed changes

def build_preprocessor(df):
    df = df.copy()
    target = 'Churn'
    if target in df.columns:
<<<<<<< Updated upstream
        y = df[target].map({'Yes': 1, 'No': 0})
        X = df.drop(columns=[target, 'customerID'], errors='ignore')
        mask = y.notna()
        y = y[mask]
        X = X.loc[mask]
    else:
=======
        # Bỏ customerID (có thể viết hoa hoặc thường) vì không có ý nghĩa dự đoán
        id_cols = [col for col in df.columns if 'customerid' in col.lower() or 'customer_id' in col.lower()]
        X = df.drop(columns=[target] + id_cols)
        # Convert Churn từ Yes/No sang 1/0 nếu cần
        if df[target].dtype == 'object':
            y = (df[target] == 'Yes').astype(int)
        else:
            y = df[target]
    else:
        id_cols = [col for col in df.columns if 'customerid' in col.lower() or 'customer_id' in col.lower()]
        X = df.drop(columns=id_cols, errors='ignore')
>>>>>>> Stashed changes
        y = None
        X = df.drop(columns=['customerID'], errors='ignore')

    available_features = [col for col in FEATURE_COLUMNS if col in X.columns]
    X = X[available_features]

    numeric_features = [col for col in available_features if col in X.select_dtypes(include=['int64', 'float64']).columns]
    categorical_features = [col for col in available_features if col in X.select_dtypes(include=['object', 'category']).columns]

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

def get_eda_stats():
    """Tải dữ liệu và trả về thống kê phục vụ EDA (không fit lại preprocessor)."""
    df = load_data()
    return perform_eda(df)


def run_preprocessing():
    df = load_data()
    eda_stats = perform_eda(df)
    preprocessor, X, y = build_preprocessor(df)
    preprocessor.fit(X)
    
<<<<<<< Updated upstream
    save_path = os.path.join(MODELS_DIR, 'preprocessor.joblib')
    joblib.dump(preprocessor, save_path)
=======
    # Lưu preprocessor
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, 'preprocessor.joblib'))
    print("Da luu preprocessor vao models/preprocessor.joblib")
    
>>>>>>> Stashed changes
    return eda_stats

if __name__ == "__main__":
    run_preprocessing()