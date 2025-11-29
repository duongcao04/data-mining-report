Customer Churn Prediction Project (CRISP-DM)

> 📋 **Thứ tự chạy:** Xem [RUN_ORDER.md](RUN_ORDER.md) để biết thứ tự các bước chạy dự án

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

Chọn mô hình tốt nhất và lưu vào models/model.pkl.

Kết quả chi tiết lưu tại models/evaluation_results.json.

Deployment:

API được xây dựng bằng FastAPI tại demo/app.py.

Cung cấp các endpoint để Train lại mô hình và Dự đoán realtime.

2. Cấu trúc thư mục và mô tả chi tiết

├── demo/
│   └── app.py            # FastAPI Server
├── models/               # model.pkl, preprocessor.joblib, evaluation_results.json
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

```bash
pip install -r requirements.txt
```

**Chi tiết các thư viện:** Xem phần Requirements trong README này.

**Danh sách thư viện chính:**
- **Xử lý dữ liệu**: pandas, numpy
- **Machine Learning**: scikit-learn, joblib
- **API**: fastapi, uvicorn, pydantic
- **Dashboard**: streamlit
- **Visualization**: matplotlib, seaborn
- **Notebook**: jupyter, notebook, ipykernel


Bước 3: Tiền xử lý dữ liệu (Preprocessing)

```bash
python src/preprocessing.py
```

**Kết quả:** Tạo `models/preprocessor.joblib`

Bước 4: Huấn luyện mô hình (Training)

```bash
python src/modeling.py
```

**Lưu ý:** Quá trình này có thể mất vài phút. Sau khi hoàn tất, mô hình sẽ được lưu vào `models/model.pkl`

> 📖 **Xem thứ tự chạy chi tiết:** [RUN_ORDER.md](RUN_ORDER.md)

uvicorn demo.app:app --reload

**Option A: FastAPI**
```bash
uvicorn demo.app:app --reload
```
Truy cập: http://127.0.0.1:8000/docs

**Option B: Streamlit Dashboard**
```bash
streamlit run demo/dashboard.py
```
Truy cập: http://localhost:8501

**Option C: Jupyter Notebook**
```bash
jupyter notebook notebooks/notebook.ipynb
```

**Option D: Web Demo**
1. Khởi động FastAPI (Option A)
2. Mở `demo/index.html` trong trình duyệt

4. Các hình thức triển khai

Dự án hỗ trợ 4 hình thức triển khai chính:

## 4.1. FastAPI - RESTful API

**Khởi động:**
```bash
uvicorn demo.app:app --reload
```

{
  "tenure": 12,
  "InternetService": "DSL",
  "Contract": "Month-to-month",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.35
}
