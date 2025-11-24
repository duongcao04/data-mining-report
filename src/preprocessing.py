# src/preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_data():
    df = pd.read_csv("data/churn.csv")
    return df


def clean_data(df):
    """
    Làm sạch dữ liệu:
    - Chuyển TotalCharges sang dạng số
    - Xử lý NaN
    - Xóa các cột không cần thiết
    """
    # Convert TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Encode Churn
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # 🔥 XÓA NHỮNG CỘT KHÔNG DÙNG (để app không bị thiếu input)
    drop_cols = [
        "StreamingMovies",
        "DeviceProtection",
        "TechSupport",
        "OnlineBackup",
        "StreamingTV",
        "OnlineSecurity",
        "MultipleLines"
    ]

    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    return df


def build_preprocessor(df):
    """
    Tạo preprocessor chuẩn: one-hot encoding + scaling
    """
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    
    categorical_cols = [
        col for col in df.columns
        if col not in numeric_cols + ["Churn", "customerID"]
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )
    return preprocessor


def split(df):
    """
    Chia dữ liệu train/test
    """
    X = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test
