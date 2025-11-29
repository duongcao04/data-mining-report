
# PHÂN TÍCH DỮ LIỆU IRIS (EDA)

# import thư viên 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.precision", 3)                       # Hiển thị số thập phân
iris = load_iris()                                          # nạp dữ liệu iris có sẳn trong thư viện scikit-learn

X = pd.DataFrame(iris.data, columns=iris.feature_names)     # bảng đặc trưng 
y = pd.Series(iris.target, name="target")                   # phân loại nhãn 

df = pd.concat([X, y], axis=1)                              # ghép lại thành bảng

# 1:HIỂN THỊ THÔNG TIN CƠ BẢN 
print("1:THÔNG TIN DỮ LIỆU IRIS")
print("Đặc trưng:", iris.feature_names)
print("Nhãn:", iris.target_names.tolist())
print("\n5 dòng đầu tiên:\n", df.head())
print("\nKích thước:", df.shape)
print("\nKiểu dữ liệu và thông tin:")
print(df.info())

# 2:TRUNG BÌNH ,ĐỘ LỆNH CỦA TỪNG ĐẶC TRƯNG
means = X.mean()                                            #trung bình cộng
stds = X.std(ddof=1)                                        #độ lệch
print("2:THỐNG KÊ MÔ TẢ")
print("Trung bình mỗi đặc trưng:\n", means)
print("\nĐộ lệch chuẩn mỗi đặc trưng:\n", stds)
print("\nMô tả nhanh (describe):\n", X.describe().loc[["mean", "std"]])

# 3:BIỂU ĐỒ CẶP (PAIRPLOT) 

print("3:VẼ BIỂU ĐỒ CẶP (PAIRPLOT)")
# thêm tên loài hoa cho dễ nhìn
df_named = df.copy()                                        #Sao chép DataFrame
df_named["species"] = df_named["target"].map(
    {i: name for i, name in enumerate(iris.target_names)}
)
# vẽ biểu đồ cặp giữa các đặc trưng
sns.pairplot(
    data=df_named,                                          #Dữ liệu để vẽ
    vars=iris.feature_names,                                #Chọn 4 đặc trưng cần so sánh
    hue="species",                                          #Màu khác nhau cho từng loài hoa
    diag_kind="kde",                                        #vẽ biểu đồ phân bố
    plot_kws={"alpha": 0.8, "s": 35}                        #	Độ trong suốt (alpha) và kích thước điểm (s)
)
plt.suptitle("Biểu đồ cặp (Pairplot) giữa các đặc trưng của Iris", y=1.02)   #tiêu đề
plt.show()                                                  #hiển thị biểu đồ.

# 4:CHUẨN HÓA DỮ LIỆU (MIN-MAX SCALER) ---
print("4:CHUẨN HÓA DỮ LIỆU (MIN-MAX SCALER)")


scaler = MinMaxScaler()                                     #tìm giá trị nhỏ nhất và lớn nhất của từng cột,
X_scaled = scaler.fit_transform(X)                          #Chuẩn hóa dữ liệu
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)        #Chuyển lại thành DataFrame

print("\nTrước chuẩn hóa:\n", X.head(3))
print("\nSau chuẩn hóa:\n", X_scaled.head(3))

# Ghép lại với cột nhãn để lưu thành dataset hoàn chỉnh
df_scaled = pd.concat([X_scaled, y], axis=1)

#  LƯU FILE CSV 
df_scaled.to_csv("iris_scaled.csv", index=False, encoding="utf-8")
print("\n Đã lưu file iris_scaled.csv trong thư mục hiện tại.")
print("File chứa dữ liệu đã được chuẩn hóa về thang [0,1].")
