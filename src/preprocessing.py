import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- CẤU HÌNH ĐƯỜNG DẪN ĐỘNG ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, '..', 'data', 'customer_churn_dataset-testing-master.csv')
ARTIFACTS_DIR = os.path.join(CURRENT_DIR, '..', 'artifacts')
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        fallback_path = '/mnt/data/customer_churn_dataset-testing-master.csv'
        if os.path.exists(fallback_path):
            path = fallback_path
        else:
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {path}")
    return pd.read_csv(path)

def perform_eda(df):
    """
    CRISP-DM: Data Understanding
    Trả về dữ liệu đã xử lý để vẽ biểu đồ EDA trên Dashboard.
    """
    # 1. Phân phối Target (Churn)
    target_dist = df['Churn'].value_counts(normalize=True).to_dict()
    
    # 2. Tương quan (Correlation) với Churn
    numeric_df = df.select_dtypes(include=[np.number])
    correlations = {}
    if not numeric_df.empty and 'Churn' in numeric_df.columns:
        # Lấy top 5 tương quan dương và âm (bỏ qua chính cột Churn)
        corr_series = numeric_df.corr()['Churn'].drop('Churn').sort_values(ascending=False)
        # Convert sang float để tránh lỗi JSON
        correlations = {k: float(v) for k, v in corr_series.items()}

    # 3. Phân tích Categorical: Tỷ lệ Churn theo nhóm
    # Hàm helper để tính tỷ lệ churn
    def get_churn_rate_by_col(col_name):
        if col_name not in df.columns: return {}
        # Group by cột đó, tính mean của Churn (tỷ lệ rời bỏ)
        return {str(k): float(v * 100) for k, v in df.groupby(col_name)['Churn'].mean().items()}

    churn_by_contract = get_churn_rate_by_col('Contract Length')
    churn_by_subscription = get_churn_rate_by_col('Subscription Type')
    churn_by_gender = get_churn_rate_by_col('Gender')

    # 4. Phân phối biến số (Numerical Distribution) - Ví dụ: Tenure
    # Tạo histogram dữ liệu cho Tenure (chia làm 5 khoảng)
    tenure_hist = {}
    if 'Tenure' in df.columns:
        counts, bin_edges = np.histogram(df['Tenure'].dropna(), bins=5)
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
            "churn_by_gender": churn_by_gender
        },
        "numerical_distribution": {
            "tenure_distribution": tenure_hist
        }
    }

def get_business_analytics(df):
    """
    Tính toán chỉ số kinh doanh (Analytics)
    """
    total_revenue = float(df['Total Spend'].sum())
    avg_revenue = float(df['Total Spend'].mean())
    total_customers = int(len(df))
    churn_rate = float(df['Churn'].mean() * 100)

    revenue_by_sub = df.groupby('Subscription Type')['Total Spend'].sum().to_dict()
    revenue_by_sub = {k: float(v) for k, v in revenue_by_sub.items()}

    customer_by_contract = df['Contract Length'].value_counts().to_dict()
    customer_by_contract = {k: int(v) for k, v in customer_by_contract.items()}

    avg_spend_by_churn = df.groupby('Churn')['Total Spend'].mean().to_dict()
    avg_spend_by_churn = {int(k): float(v) for k, v in avg_spend_by_churn.items()}

    return {
        "kpi": {
            "total_revenue": total_revenue,
            "avg_revenue_per_user": avg_revenue,
            "total_customers": total_customers,
            "churn_rate": churn_rate
        },
        "charts": {
            "revenue_by_subscription": revenue_by_sub,
            "customer_by_contract": customer_by_contract,
            "avg_spend_churn_vs_loyal": avg_spend_by_churn
        }
    }

def build_preprocessor(df):
    target = 'Churn'
    if target in df.columns:
        X = df.drop(columns=[target, 'CustomerID'], errors='ignore')
        y = df[target]
    else:
        X = df.drop(columns=['CustomerID'], errors='ignore')
        y = None

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

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
    df = load_data()
    eda_stats = perform_eda(df)
    preprocessor, X, y = build_preprocessor(df)
    preprocessor.fit(X)
    
    save_path = os.path.join(ARTIFACTS_DIR, 'preprocessor.joblib')
    joblib.dump(preprocessor, save_path)
    return eda_stats

if __name__ == "__main__":
    run_preprocessing()