import pandas as pd
import numpy as np
import os

print("Starting Step 3: Feature Engineering...")

# 1. Load clean data
data_path = 'data/processed/clean_auto_mpg.csv'
df = pd.read_csv(data_path)

# 2. Extract Target (y) and Features (X)
y = df['mpg']
X = df.drop(columns=['mpg'])

# 3. Encoding Categorical Variables
# 'origin' is categorical (1: USA, 2: Europe, 3: Asia)
X['origin'] = X['origin'].map({1: 'USA', 2: 'Europe', 3: 'Asia'})
X = pd.get_dummies(X, columns=['origin'], drop_first=False)
# Convert boolean to int
for col in ['origin_Asia', 'origin_Europe', 'origin_USA']:
    if col in X.columns:
        X[col] = X[col].astype(int)

# 4. Domain Feature Creation
# Adding some interaction features based on domain knowledge
X['weight_per_hp'] = X['weight'] / X['horsepower']
X['displacement_per_cylinder'] = X['displacement'] / X['cylinders']

# Plotting the new feature insight
import matplotlib.pyplot as plt
import seaborn as sns
os.makedirs('Figures/FE', exist_ok=True)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X['weight_per_hp'], y=y, alpha=0.7, color='purple')
plt.title("Tương quan giữa Weight per HP và MPG")
plt.xlabel("Weight per HP (lbs / horsepower)")
plt.ylabel("MPG (Miles per gallon)")
plt.tight_layout()
plt.savefig('Figures/FE/new_features.png')
plt.close()

# 5. Save Feature Set
os.makedirs('data/processed', exist_ok=True)
X.to_csv('data/processed/X_features.csv', index=False)
y.to_csv('data/processed/y_target.csv', index=False)

print(f"Feature set saved. Shape of X: {X.shape}, Shape of y: {y.shape}")

# Generate Buoc3.md
buoc3_content = f"""# Feature Engineering Report (Phase 3)

## 1. Goal
Tạo, chọn lọc và biến đổi đặc trưng để giúp mô hình học tốt hơn.

## 2. Input
- Dataset sạch: `clean_auto_mpg.csv`

## 3. Tasks Performed
- **Tách X, y**: Target là `mpg`, các cột còn lại là Features.
- **Encoding**: Biến categorical `origin` (1, 2, 3) đã được One-Hot Encoding thành `origin_USA`, `origin_Europe`, `origin_Asia`.
- **Feature Creation**: Tạo thêm 2 đặc trưng từ Domain Knowledge:
  - `weight_per_hp` = weight / horsepower (tỉ lệ trọng lượng trên mã lực, ảnh hưởng lớn đến tiêu thụ nhiên liệu).
  - `displacement_per_cylinder` = displacement / cylinders (dung tích mỗi xi-lanh).
- **Polynomial Features & Scaling**: Đã xem xét nhưng QUYẾT ĐỊNH CHƯA THỰC HIỆN ở bước này. Lý do: 
  - Scaling trên toàn bộ dữ liệu trước khi chia train/test sẽ gây **Data Leakage**.
  - Polynomial Features nên được đưa vào `sklearn.pipeline.Pipeline` để có thể dễ dàng **tune bậc đa thức (degree)** ở Bước 7.

## 4. Output
- Kích thước Feature Set (X): {X.shape[0]} mẫu, {X.shape[1]} đặc trưng.
- File lưu trữ: 
  - `data/processed/X_features.csv`
  - `data/processed/y_target.csv`

## 5. Insight & Decision
- **Insight**: Việc tạo One-Hot Encoding và các feature kết hợp (`weight_per_hp`) diễn ra độc lập trên từng dòng (row-wise), nên không gây Data Leakage. Feature set hiện tại đã phong phú hơn và phản ánh đúng bản chất dữ liệu.
- **Decision**: Hoàn thành Bước 3. Dữ liệu đã sẵn sàng để chuyển sang **Bước 4 — Data Split** nhằm chia tập Train/Val/Test một cách an toàn.
"""

with open('Buoc3.md', 'w', encoding='utf-8') as f:
    f.write(buoc3_content)

print("Saved report to Buoc3.md")
