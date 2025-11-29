import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


df = pd.read_csv("../data/Customer-Churn.csv")

print(df.head())
print(df.shape)
print(df.columns)
print(df.isnull().sum())
print(df.dtypes)

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

#
features = ["tenure", "MonthlyCharges", "Contract", "InternetService", "PaymentMethod"]
target = "Churn"

X = df[features]
y = df[target]


print(X.columns.tolist())
print( target)
print("Kích thước X:", X.shape)
print("Kích thước y:", y.shape)
print(X.head())


# chia train/test( 80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print("X_train:", X_train.shape, " | y_train:", y_train.shape)
print("X_test :", X_test.shape,  " | y_test :", y_test.shape)

# (tuỳ chọn) kiểm tra tỉ lệ Churn trước/sau khi chia
print("Tỉ lệ Churn toàn bộ:", y.mean().round(3))
print("Tỉ lệ Churn train   :", y_train.mean().round(3))
print("Tỉ lệ Churn test    :", y_test.mean().round(3))


# xử lý dữ liệu

numeric_features = ["tenure", "MonthlyCharges"]
categorical_features = ["Contract", "InternetService", "PaymentMethod"]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
    ],
    remainder='passthrough'
)


# 1. Logistic Regression Pipeline
pipe_lr = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])
# 2. Random Forest Pipeline
pipe_rf = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced_subsample"))
])

# 3. Support Vector Machine (SVC) Pipeline
pipe_svm = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"))
])

models = {
    "Logistic Regression": pipe_lr,
    "Random Forest": pipe_rf,
    "SVM": pipe_svm
}

results = []

for name, model in models.items():
    print(f"\n Huấn luyện mô hình: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1
    })

# Kết quả
results_df = pd.DataFrame(results).sort_values(by="F1", ascending=False)
print("\n So sánh mô hình:")
print(results_df.to_string(index=False))