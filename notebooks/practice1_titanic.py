import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


print("--- 1. Nạp dữ liệu và hiển thị 5 dòng đầu tiên ---")
df = pd.read_csv("../data/titanic.csv")
print(df.head(5))
print("\n" + "="*50 + "\n")

print("--- 2. Xác định số dòng và số cột ---")
print("Tập dữ liệu có số dòng là:", df.shape[0])
print("Tập dữ liệu có số cột là:", df.shape[1])
print("\n" + "="*50 + "\n")

print("--- 3. Thống kê mô tả và đếm giá trị thiếu ---")
print("Thống kê các cột dữ liệu số:")
print(df.describe())
print("\nĐếm số lượng giá trị thiếu (missing values) trên mỗi cột:")
print(df.isnull().sum())
print("\n=> Nhận xét: Cột 'Age', 'Cabin' và 'Embarked' đang có giá trị bị thiếu.")
print("\n" + "="*50 + "\n")

print("--- 4. Vẽ histogram cho thuộc tính Age và boxplot cho Fare. ---")
plt.figure(figsize=(10,4))
print("--- Histogram cho Age ---")

plt.subplot(1, 2, 1)
sns.histplot(df['Age'], bins=30, kde=True, color='green')
plt.title("Phân bố độ tuổi hành khách (Age)")
plt.xlabel("Age")
plt.ylabel("Số lượng")

print("--- Boxplot cho Fare ---")

plt.subplot(1, 2, 2)
sns.boxplot(x=df['Fare'], color='orange')
plt.title("Phân bố giá vé (Fare)")
plt.xlabel("Fare")
plt.show()
print("\n" + "="*50 + "\n")

print("--- 5. Điền giá trị trung vị cho Age bị thiếu ---")

age_filled   =df['Age'].fillna(df['Age'].median(), inplace=True)
print("Số giá trị thiếu của cột 'age' sau khi điền", age_filled )
print("\n" + "="*50 + "\n")

print("---6. Chuẩn hóa các cột Age và Fare ---")
scaler = MinMaxScaler()
df[['Age_norm', 'Fare_norm']] = scaler.fit_transform(df[['Age', 'Fare']])
print(df[['Age', 'Age_norm', 'Fare', 'Fare_norm']].head())
print("\n" + "="*50 + "\n")
print("---7. Mã hóa cột Sex và Embarked thành dạng số ---")

    #Label Encoding
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    #One-Hot Encoding
df = pd.get_dummies(df, columns=['Embarked'], prefix='Embarked')
df[['Embarked_C', 'Embarked_Q', 'Embarked_S']] = df[['Embarked_C', 'Embarked_Q', 'Embarked_S']].astype(int)
print(df.head())
print("\n" + "="*50 + "\n")

print("--- 8. Lưu tập dữ liệu sau xử lý ---")
df.to_csv("../data/titanic_clean.csv", index=False)
print(" Lưu thành công")
