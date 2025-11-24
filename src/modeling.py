# src/modeling.py

import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score

def build_models(preprocessor):
    """
    Tạo nhiều mô hình để thử nghiệm
    """
    models = {
        "logistic_regression": Pipeline([
            ("preprocess", preprocessor),
            ("clf", LogisticRegression(max_iter=5000))
        ]),
        "random_forest": Pipeline([
            ("preprocess", preprocessor),
            ("clf", RandomForestClassifier(
                n_estimators=250,
                random_state=42,
                class_weight="balanced"
            ))
        ])
    }
    return models


def train_best_model(models, X_train, y_train):
    """
    Chạy cross-validation và chọn model có ROC-AUC cao nhất
    """
    best_model = None
    best_score = -1

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
        print(f"{name}: ROC-AUC mean = {scores.mean():.4f}")

        if scores.mean() > best_score:
            best_score = scores.mean()
            best_model = model
    
    print(f"\n=> Best model selected with ROC-AUC = {best_score:.4f}")
    best_model.fit(X_train, y_train)
    return best_model


def evaluate(model, X_test, y_test):
    """
    Đánh giá model sau khi train
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score = {auc:.4f}")
    return auc


def save_model(model, path="models/model.pkl"):
    """
    Lưu model đã train để dự đoán
    """
    joblib.dump(model, path)
    print(f"Model saved to {path}")  
