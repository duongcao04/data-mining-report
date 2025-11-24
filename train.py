# train.py

from src.preprocessing import load_data, clean_data, build_preprocessor, split
from src.modeling import build_models, train_best_model, evaluate, save_model

# 1. Load dữ liệu
df = load_data()
df = clean_data(df)

# 2. Preprocessing
preprocessor = build_preprocessor(df)

# 3. Chia dữ liệu
X_train, X_test, y_train, y_test = split(df)

# 4. Build và train models
models = build_models(preprocessor)
best_model = train_best_model(models, X_train, y_train)

# 5. Evaluate model tốt nhất
evaluate(best_model, X_test, y_test)

# 6. Save model
save_model(best_model)
