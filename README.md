Customer Churn Prediction Project (CRISP-DM)

Dự án này xây dựng một hệ thống Machine Learning để dự đoán khách hàng rời bỏ (Churn), tuân thủ chặt chẽ 6 giai đoạn của quy trình CRISP-DM.

1. Quy trình CRISP-DM trong dự án

Business Understanding:

Mục tiêu: Xác định khách hàng có nguy cơ rời bỏ để có chiến lược giữ chân.

Đầu ra: Nhãn dự đoán (Churn/No Churn) và xác suất.

Data Understanding:

Thực hiện tại /src/preprocessing.py.

API Endpoint /eda trả về các thống kê mô tả, giá trị thiếu, và ma trận tương quan.

Data Preparation:

Xử lý tại /src/preprocessing.py.

Pipeline: Xử lý Missing Values (Imputer) -> Chuẩn hóa (StandardScaler) -> Mã hóa biến phân loại (OneHotEncoder).

Artifact: preprocessor.joblib.

Modeling:

Thực hiện tại /src/modeling.py.

Train 3 thuật toán: Logistic Regression, Random Forest, SVM.

Sử dụng Cross-validation.

Evaluation:

So sánh mô hình dựa trên F1-Score, Accuracy, ROC-AUC.

Chọn mô hình tốt nhất và lưu vào artifacts/best_model.joblib.

Kết quả chi tiết lưu tại artifacts/evaluation_results.json.

Deployment:

API được xây dựng bằng FastAPI tại api/app.py.

Cung cấp các endpoint để Train lại mô hình và Dự đoán realtime.

2. Cấu trúc thư mục

├── demo/
│   └── app.py            # FastAPI Server
├── artifacts/            # Chứa model và preprocessor đã lưu
├── src/
│   ├── preprocessing.py  # Load data, EDA, Feature Engineering
│   ├── modeling.py       # Train và Evaluate models
│   └── predict.py        # Class dự đoán
├── README.md             # Hướng dẫn
└── requirements.txt      # Thư viện


3. Hướng dẫn cài đặt và chạy

Bước 1: Tạo và kích hoạt Virtual Environment

Để tránh xung đột thư viện, bạn nên tạo một môi trường ảo riêng biệt cho dự án.

1. Tạo môi trường ảo:

python -m venv venv


2. Kích hoạt môi trường:

Trên Windows:

.\venv\Scripts\activate


Trên macOS / Linux:

source venv/bin/activate


Bước 2: Cài đặt thư viện

Sau khi kích hoạt môi trường ảo, hãy cài đặt các dependencies:

pip install -r requirements.txt


Bước 3: Huấn luyện mô hình (Lần đầu)

Bạn có thể chạy script trực tiếp hoặc qua API.

# Cách 1: Chạy script
python src/modeling.py


Bước 4: Khởi động API

uvicorn demo.app:app --reload


Server sẽ chạy tại: http://127.0.0.1:8000

4. Sử dụng API

Tài liệu API (Swagger UI): Truy cập http://127.0.0.1:8000/docs

Training: POST /train

EDA: GET /eda

Predict: POST /predict

Body JSON mẫu:

{
  "Age": 30,
  "Gender": "Female",
  "Tenure": 12,
  "Usage_Frequency": 5,
  "Support_Calls": 2,
  "Payment_Delay": 0,
  "Subscription_Type": "Basic",
  "Contract_Length": "Monthly",
  "Total_Spend": 500,
  "Last_Interaction": 5
}
