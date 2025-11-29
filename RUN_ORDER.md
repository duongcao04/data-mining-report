# 📋 Thứ tự chạy dự án

## Quy trình chạy đầy đủ (Lần đầu)

### Bước 1: Chuẩn bị môi trường

```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt virtual environment
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Cài đặt thư viện
pip install -r requirements.txt
```

---

### Bước 2: Tiền xử lý dữ liệu (Preprocessing)

```bash
python src/preprocessing.py
```

**Kết quả:**
- Tải và phân tích dữ liệu
- Tạo preprocessor pipeline
- Lưu `models/preprocessor.joblib`

**Thời gian:** ~10-30 giây

---

### Bước 3: Huấn luyện mô hình (Training)

```bash
python src/modeling.py
```

**Kết quả:**
- Train 3 mô hình: Logistic Regression, Random Forest, SVM
- So sánh và chọn mô hình tốt nhất
- Lưu `models/model.pkl`
- Lưu `models/evaluation_results.json`

**Thời gian:** ~2-5 phút (tùy vào kích thước dữ liệu)

---

### Bước 4: Chạy ứng dụng

**Option A: FastAPI (Khuyến nghị)**

```bash
uvicorn demo.app:app --reload
```

**Truy cập:**
- API: http://127.0.0.1:8000
- Swagger UI: http://127.0.0.1:8000/docs
- Web Demo: Mở `demo/index.html` trong trình duyệt

**Option B: Jupyter Notebook**

```bash
jupyter notebook notebooks/notebook.ipynb
```

---

## Quy trình chạy nhanh (Lần sau)

Nếu đã train mô hình rồi, chỉ cần:

```bash
# 1. Kích hoạt venv (nếu chưa)
.\venv\Scripts\activate

# 2. Chạy API
uvicorn demo.app:app --reload

# 3. Mở web demo
# Mở demo/index.html trong trình duyệt
```

---

## Thứ tự các bước (Tóm tắt)

```
1. Setup môi trường
   └─> python -m venv venv
   └─> .\venv\Scripts\activate
   └─> pip install -r requirements.txt

2. Preprocessing
   └─> python src/preprocessing.py
   └─> Tạo: models/preprocessor.joblib

3. Training
   └─> python src/modeling.py
   └─> Tạo: models/model.pkl

4. Chạy ứng dụng
   └─> uvicorn demo.app:app --reload
   └─> Mở demo/index.html
```

---

## Kiểm tra nhanh

### Kiểm tra môi trường:
```bash
python --version  # Phải >= 3.8
pip list | Select-String "pandas"  # Kiểm tra thư viện
```

### Kiểm tra dữ liệu:
```bash
# Kiểm tra file CSV có tồn tại không
python -c "import os; print('OK' if os.path.exists('data/Customer-Churn.csv') else 'ERROR')"
```

### Kiểm tra model:
```bash
# Kiểm tra model đã train chưa
python -c "import os; print('OK' if os.path.exists('models/model.pkl') else 'CHUA TRAIN')"
```

---

## Lưu ý quan trọng

### ⚠️ Phải chạy theo thứ tự:
1. **Preprocessing** → Tạo preprocessor
2. **Training** → Tạo model (cần preprocessor)
3. **API/Demo** → Sử dụng model (cần model đã train)

### ❌ Không thể bỏ qua bước:
- Không thể train nếu chưa preprocessing
- Không thể predict nếu chưa train

### ✅ Có thể bỏ qua nếu đã có:
- Nếu đã có `models/preprocessor.joblib` → Bỏ qua preprocessing
- Nếu đã có `models/model.pkl` → Bỏ qua training

---

## Troubleshooting

### Lỗi: "ModuleNotFoundError"
→ Chưa cài thư viện hoặc chưa kích hoạt venv
```bash
.\venv\Scripts\activate
pip install -r requirements.txt
```

### Lỗi: "FileNotFoundError: Không tìm thấy file dữ liệu"
→ Kiểm tra file `data/Customer-Churn.csv` có tồn tại không

### Lỗi: "Model chưa được train!"
→ Chạy training trước:
```bash
python src/modeling.py
```

### Lỗi khi chạy API: "Address already in use"
→ Port 8000 đang được dùng, đổi port:
```bash
uvicorn demo.app:app --reload --port 8001
```

---

## Thời gian ước tính

| Bước | Thời gian |
|------|-----------|
| Setup môi trường | 2-5 phút |
| Preprocessing | 10-30 giây |
| Training | 2-5 phút |
| Chạy API | Ngay lập tức |
| **Tổng cộng** | **~5-10 phút** |

---

## Checklist

Trước khi chạy, đảm bảo:
- [ ] Python >= 3.8 đã được cài đặt
- [ ] File `data/Customer-Churn.csv` có trong thư mục data/
- [ ] Đã tạo và kích hoạt virtual environment
- [ ] Đã cài đặt tất cả thư viện từ requirements.txt

Sau khi chạy, kiểm tra:
- [ ] `models/preprocessor.joblib` đã được tạo
- [ ] `models/model.pkl` đã được tạo
- [ ] API chạy được tại http://127.0.0.1:8000
- [ ] Web demo load được features từ API

---

**Chúc bạn thành công! 🎉**

