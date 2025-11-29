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

<<<<<<< Updated upstream
Chọn mô hình tốt nhất và lưu vào models/model.pkl.

Kết quả chi tiết lưu tại models/evaluation_results.json.
=======
Chọn mô hình tốt nhất và lưu vào `models/model.pkl`.

Kết quả chi tiết lưu tại `models/evaluation_results.json`.
>>>>>>> Stashed changes

Deployment:

API được xây dựng bằng FastAPI tại demo/app.py.

Cung cấp các endpoint để Train lại mô hình và Dự đoán realtime.

2. Cấu trúc thư mục và mô tả chi tiết

<<<<<<< Updated upstream
├── demo/
│   └── app.py            # FastAPI Server
├── models/               # model.pkl, preprocessor.joblib, evaluation_results.json
├── src/
│   ├── preprocessing.py  # Load data, EDA, Feature Engineering
│   ├── modeling.py       # Train và Evaluate models
│   └── predict.py        # Class dự đoán
├── README.md             # Hướng dẫn
└── requirements.txt      # Thư viện
=======
## 2.1. data/

Chứa các dữ liệu đầu vào của dự án, ví dụ như bộ dữ liệu `Customer-Churn.csv`.

**Lưu ý:** Không upload dữ liệu lớn nếu có. Bạn có thể chỉ cung cấp một phần nhỏ hoặc hướng dẫn người dùng tải dữ liệu từ nguồn khác nếu bộ dữ liệu quá lớn.

## 2.2. notebooks/

Jupyter Notebooks hoặc Google Colab Notebooks chứa các bước phân tích dữ liệu, khám phá dữ liệu (EDA), và các thí nghiệm mô hình hóa.

**notebook.ipynb:** Bao gồm các bước:
- **Khám phá dữ liệu (EDA):** Xem thông tin cơ bản của dữ liệu, kiểm tra các giá trị thiếu, phân tích mối tương quan giữa các đặc trưng.
- **Tiền xử lý:** Chuẩn hóa, mã hóa và xử lý các giá trị thiếu.
- **Huấn luyện và đánh giá mô hình:** Áp dụng các mô hình như Logistic Regression, Random Forest, và SVM, sau đó đánh giá chúng bằng các chỉ số như Accuracy, F1-Score, ROC-AUC.

## 2.3. src/

Chứa các module Python chính của dự án:

**preprocessing.py:** Mã xử lý dữ liệu trước khi đưa vào mô hình.
- Xử lý các giá trị thiếu
- Chuẩn hóa các đặc trưng số
- Mã hóa các cột phân loại
- Thực hiện EDA và trả về báo cáo thống kê

**modeling.py:** Huấn luyện mô hình và đánh giá các mô hình học máy.
- Huấn luyện các mô hình như Logistic Regression, Random Forest, và SVM
- Đánh giá mô hình bằng các chỉ số như F1-Score, Accuracy, và ROC-AUC
- Lưu mô hình tốt nhất và báo cáo đánh giá

**predict.py:** Dự đoán churn cho dữ liệu mới sử dụng mô hình đã huấn luyện.
- Tải mô hình và preprocessor
- Chuẩn hóa và mã hóa dữ liệu đầu vào
- Dự đoán churn và trả về xác suất

## 2.4. demo/

FastAPI app để triển khai mô hình học máy dưới dạng API.

**app.py:** Cung cấp API endpoints để:
- Huấn luyện mô hình (`POST /train`)
- Xem kết quả EDA (`GET /eda`)
- Dự đoán churn cho dữ liệu mới (`POST /predict`)
- Kiểm tra trạng thái hệ thống (`GET /status`)
- Lấy danh sách features (`GET /features`)

**index.html:** Web demo với giao diện HTML/JavaScript:
- Tự động load features từ API
- Form động dựa trên dữ liệu thực tế
- Hiển thị kết quả dự đoán trực quan
- Không cần server riêng, chỉ cần mở file HTML trong trình duyệt

## 2.5. models/

Lưu trữ mô hình đã huấn luyện.

**model.pkl:** Mô hình học máy đã được huấn luyện và lưu trữ dưới dạng file pickle để sử dụng lại mà không cần huấn luyện lại.

**Lưu ý:** Thư mục `models/` sẽ được tự động tạo khi chạy preprocessing và modeling, chứa:
- `preprocessor.joblib`: Pipeline tiền xử lý dữ liệu
- `model.pkl`: Mô hình tốt nhất đã được huấn luyện
- `evaluation_results.json`: Kết quả đánh giá các mô hình

## 2.6. requirements.txt

Danh sách các thư viện cần thiết cho dự án:
- `pandas`: Xử lý và phân tích dữ liệu
- `numpy`: Tính toán số học
- `scikit-learn`: Machine learning models và preprocessing
- `joblib`: Lưu và tải mô hình
- `fastapi`: Framework để xây dựng API
- `uvicorn`: ASGI server để chạy FastAPI
- `pydantic`: Validation dữ liệu cho API
- `matplotlib`: Vẽ biểu đồ
- `seaborn`: Visualization nâng cao

Các thư viện này sẽ được sử dụng cho tiền xử lý dữ liệu, huấn luyện mô hình, triển khai API, và tạo ứng dụng demo.

## 2.7. README.md

Hướng dẫn cách cài đặt môi trường, cách sử dụng mã nguồn, và cách chạy API hoặc ứng dụng demo.

Bao gồm các bước để:
- Tạo và kích hoạt môi trường ảo
- Cài đặt thư viện từ requirements.txt
- Huấn luyện mô hình lần đầu
- Khởi động FastAPI hoặc Streamlit app

## 2.8. report.pdf (Tùy chọn)

Báo cáo mô tả quy trình CRISP-DM:
- **Business Understanding:** Giới thiệu về mục tiêu dự đoán churn và các chỉ số quan trọng
- **Data Understanding:** Khám phá bộ dữ liệu (EDA)
- **Data Preparation:** Tiền xử lý dữ liệu (chuẩn hóa, mã hóa)
- **Modeling:** Huấn luyện các mô hình và đánh giá chúng
- **Evaluation:** Đánh giá các mô hình và chọn mô hình tốt nhất
- **Deployment:** Triển khai mô hình vào ứng dụng (API hoặc Streamlit)
>>>>>>> Stashed changes


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

<<<<<<< Updated upstream
uvicorn demo.app:app --reload
=======
Bước 6: Khởi động ứng dụng
>>>>>>> Stashed changes

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

<<<<<<< Updated upstream
{
  "tenure": 12,
  "InternetService": "DSL",
  "Contract": "Month-to-month",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.35
}
=======
**Truy cập:**
- API Documentation (Swagger UI): http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

**Các endpoint:**
- `GET /`: Trang chủ
- `GET /status`: Kiểm tra trạng thái hệ thống
- `GET /eda`: Xem kết quả phân tích dữ liệu (EDA)
- `POST /train`: Huấn luyện lại mô hình
- `POST /predict`: Dự đoán churn cho dữ liệu mới

**Ví dụ sử dụng API:**
```bash
# Dự đoán
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

## 4.2. Streamlit Dashboard

**Khởi động:**
```bash
streamlit run demo/dashboard.py
```

**Truy cập:** http://localhost:8501

**Tính năng:**
- 📈 **Phân tích dữ liệu (EDA)**: Visualization dữ liệu, ma trận tương quan, phân phối
- 🤖 **Dự đoán**: Giao diện thân thiện để nhập thông tin và dự đoán churn
- ⚙️ **Quản lý mô hình**: Huấn luyện mô hình, xem kết quả đánh giá
- 📊 **Dashboard tương tác**: Biểu đồ, metrics, và thống kê trực quan

## 4.3. Ứng dụng Web Demo (HTML/JS)

**Sử dụng:**
1. Khởi động FastAPI (xem mục 4.1)
2. Mở file `demo/index.html` trong trình duyệt
3. Form sẽ tự động load features từ API
4. Nhập thông tin khách hàng và nhấn "Dự đoán"

**Tính năng:**
- ✅ Tự động load features từ API (không hardcode)
- ✅ Form động dựa trên dữ liệu thực tế
- ✅ Giao diện web đẹp, responsive
- ✅ Kết nối với FastAPI backend
- ✅ Hiển thị kết quả trực quan với progress bar
- ✅ Cảnh báo dựa trên xác suất churn

## 4.4. Báo cáo HTML

**Tạo báo cáo:**
```bash
# Báo cáo có thể được tạo từ notebook hoặc API endpoint /eda
```

**Kết quả:** File HTML được lưu trong thư mục `reports/`

**Nội dung báo cáo:**
- Business Understanding
- Data Understanding (EDA với biểu đồ)
- Data Preparation
- Modeling & Evaluation (so sánh các mô hình)
- Deployment

**Mở báo cáo:** Mở file HTML trong trình duyệt web

## 4.5. Jupyter Notebook

**Khởi động:**
```bash
jupyter notebook notebooks/notebook.ipynb
```

**Nội dung:**
- Phân tích đầy đủ theo quy trình CRISP-DM
- EDA với visualization
- Preprocessing và modeling
- So sánh và đánh giá mô hình

## 5. So sánh các hình thức triển khai

| Hình thức | Ưu điểm | Sử dụng khi |
|-----------|---------|-------------|
| **FastAPI** | RESTful, dễ tích hợp, Swagger UI | Tích hợp vào hệ thống, mobile app, microservices |
| **Streamlit Dashboard** | Giao diện đẹp, tương tác, dễ dùng | Demo, presentation, phân tích nhanh |
| **Web Demo HTML** | Tự động load features, không hardcode | Demo động, dễ sử dụng |
| **Báo cáo HTML** | Tĩnh, dễ in, chia sẻ | Báo cáo cuối kỳ, documentation |
| **Jupyter Notebook** | Tương tác, reproducible | Phân tích, thí nghiệm, học tập |
>>>>>>> Stashed changes
