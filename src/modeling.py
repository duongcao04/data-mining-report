import pandas as pd
import joblib
import os
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

# Import từ module preprocessing
try:
    from src.preprocessing import load_data, build_preprocessor
except ImportError:
    # Fallback nếu chạy trực tiếp file này mà không setup package
    from preprocessing import load_data, build_preprocessor

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

def train_and_evaluate():
    """
    CRISP-DM: Modeling & Evaluation
    Train 3 mô hình, so sánh và chọn best model.
    """
    print("Bat dau quy trinh huan luyen...")
    
    # 1. Load dữ liệu
    df = load_data()
    preprocessor, X, y = build_preprocessor(df)
    
    # 2. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Định nghĩa các mô hình
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, random_state=42) # probability=True để tính ROC-AUC
    }
    
    results = {}
    best_model_name = None
    best_score = 0.0
    best_pipeline = None

    # 4. Train và Evaluate từng mô hình
    model_list = list(models.items())
    for idx, (name, model) in enumerate(model_list, 1):
        print(f"Dang train {name} ({idx}/{len(model_list)})...")
        
        # Tạo pipeline đầy đủ: Preprocessor -> Model
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', model)])
        
        clf.fit(X_train, y_train)
        
        # Dự đoán
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        
        # Tính metrics (CRISP-DM: Evaluation)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }
        
        results[name] = metrics
        print(f"Ket qua {name}: {metrics}")
        
        # Chọn mô hình tốt nhất dựa trên F1-Score (hoặc ROC-AUC tùy bài toán)
        if metrics['f1'] > best_score:
            best_score = metrics['f1']
            best_model_name = name
            best_pipeline = clf

    print(f"\n>>> Mo hinh tot nhat: {best_model_name} voi F1-Score: {best_score:.4f}")
    
    # 5. Lưu mô hình tốt nhất (CRISP-DM: Deployment Preparation)
    model_path = os.path.join(MODELS_DIR, 'model.pkl')
    joblib.dump(best_pipeline, model_path)
    print(f"Da luu mo hinh tot nhat vao {model_path}")
    
    # Lưu báo cáo kết quả
    with open(os.path.join(MODELS_DIR, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    return results

if __name__ == "__main__":
    train_and_evaluate()