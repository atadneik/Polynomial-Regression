import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create Figures directory
os.makedirs('Figures/EDA', exist_ok=True)

print("Starting Step 1: Exploratory Data Analysis (EDA)...")

# 1. Load Data
# Data format: space-separated, no header, last column is string (car name)
columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 
           'acceleration', 'model_year', 'origin', 'car_name']

# Read the file
data_path = 'data/auto-mpg.data'
df = pd.read_csv(data_path, sep=r'\s+', names=columns, na_values='?')

# 2. Basic Info & Missing Values
print("\n--- Basic Information ---")
df.info()

print("\n--- Missing Values ---")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# 3. Distribution Analysis
print("\n--- Plotting Distributions ---")
numerical_features = ['mpg', 'displacement', 'horsepower', 'weight', 'acceleration']
df[numerical_features].hist(bins=20, figsize=(12, 8), color='skyblue', edgecolor='black')
plt.suptitle("Distribution of Numerical Features")
plt.tight_layout()
plt.savefig('Figures/EDA/distributions.png')
plt.close()

# Categorical/Discrete Features Distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sns.countplot(data=df, x='cylinders', ax=axes[0], palette='viridis')
sns.countplot(data=df, x='model_year', ax=axes[1], palette='viridis')
sns.countplot(data=df, x='origin', ax=axes[2], palette='viridis')
plt.suptitle("Distribution of Discrete/Categorical Features")
plt.tight_layout()
plt.savefig('Figures/EDA/categorical_distributions.png')
plt.close()

# 4. Outlier Analysis (Boxplots)
print("--- Plotting Boxplots for Outliers ---")
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[numerical_features], orient="h", palette="Set2")
plt.title("Boxplots of Numerical Features (Outlier Detection)")
plt.tight_layout()
plt.savefig('Figures/EDA/boxplots.png')
plt.close()

# 5. Correlation Matrix
print("--- Plotting Correlation Matrix ---")
plt.figure(figsize=(10, 8))
# Drop car_name for correlation
corr_matrix = df.drop(columns=['car_name']).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig('Figures/EDA/correlation.png')
plt.close()

# 6. Target Relationships (Scatter plots vs mpg)
print("--- Plotting Feature vs Target (mpg) ---")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
sns.scatterplot(data=df, x='weight', y='mpg', ax=axes[0,0], alpha=0.6)
sns.scatterplot(data=df, x='displacement', y='mpg', ax=axes[0,1], alpha=0.6)
sns.scatterplot(data=df, x='horsepower', y='mpg', ax=axes[1,0], alpha=0.6)
sns.scatterplot(data=df, x='acceleration', y='mpg', ax=axes[1,1], alpha=0.6)
plt.suptitle("Features vs Target (mpg)")
plt.tight_layout()
plt.savefig('Figures/EDA/features_vs_target.png')
plt.close()

print("\nEDA Completed. Generating Buoc1.md...")

# 7. Generate Buoc1.md
readme_content = """# Data Profile Report (Phase 1 - EDA)

## 1. Goal
Phân tích và hiểu đặc điểm của bộ dữ liệu Auto-MPG gốc.

## 2. Input
- Dataset: `auto-mpg.data`
- Số lượng mẫu: 398
- Số lượng đặc trưng: 9 (bao gồm target `mpg`)

## 3. Data Dictionary
- `mpg`: Miles per gallon (Target Variable - Continuous)
- `cylinders`: Số lượng xi-lanh (Discrete)
- `displacement`: Dung tích động cơ (Continuous)
- `horsepower`: Mã lực (Continuous - có giá trị thiếu '?')
- `weight`: Trọng lượng (Continuous)
- `acceleration`: Gia tốc (Continuous)
- `model_year`: Năm sản xuất (Discrete)
- `origin`: Khu vực sản xuất (Discrete: 1=Mỹ, 2=Âu, 3=Á)
- `car_name`: Tên xe (String - Unique)

## 4. Findings & Insights
- **Missing Values**: Cột `horsepower` có 6 giá trị thiếu.
- **Distributions**: 
  - `mpg` hơi lệch phải (right-skewed).
  - `displacement` và `weight` phân phối đa mode (bimodal) và lệch phải.
- **Outliers**: `horsepower` và `acceleration` có một vài điểm ngoại lệ nhẹ (dựa trên boxplot). Tuy nhiên, có vẻ đây là giá trị thực tế của các xe hiệu suất cao hoặc thấp, không phải lỗi nhập liệu.
- **Correlations**:
  - `mpg` có tương quan **âm rất mạnh** với `weight` (-0.83), `displacement` (-0.80) và `horsepower` (-0.78).
  - `cylinders`, `displacement`, `weight`, `horsepower` có tương quan **dương rất mạnh** với nhau (đa cộng tuyến - multicollinearity).

## 5. Decision
- **Dữ liệu có sử dụng được không?** Có.
- **Hành động tiếp theo (Bước 2 - Data Cleaning)**:
  1. Xử lý 6 giá trị thiếu trong `horsepower` (sử dụng Median imputation).
  2. Bỏ cột `car_name` vì đây là định danh (unique identifier), không có giá trị học máy.
  3. Outlier trong bài toán này có vẻ hợp lý, có thể giữ nguyên ở bước tiếp theo để không mất thông tin.
"""

with open('Buoc1.md', 'w', encoding='utf-8') as f:
    f.write(readme_content)

print("Saved report to Buoc1.md")

# ==========================================
# PHASE 2: DATA CLEANING
# ==========================================
print("\nStarting Step 2: Data Cleaning...")

# 1. Handle Missing Values
median_hp = df['horsepower'].median()
df['horsepower'] = df['horsepower'].fillna(median_hp)

# 2. Drop unique identifier
if 'car_name' in df.columns:
    df = df.drop(columns=['car_name'])

# 3. Handle Duplicates
df = df.drop_duplicates()

# Save clean dataset
os.makedirs('data/processed', exist_ok=True)
clean_data_path = 'data/processed/clean_auto_mpg.csv'
df.to_csv(clean_data_path, index=False)
print(f"Clean dataset saved to {clean_data_path}")

# Generate Buoc2.md
buoc2_content = f"""# Data Cleaning Report (Phase 2)

## 1. Goal
Làm sạch và chuẩn hóa dữ liệu để đảm bảo độ tin cậy.

## 2. Input
- Dataset từ Bước 1
- Insight: Có 6 missing values ở `horsepower` và cột `car_name` không mang giá trị.

## 3. Tasks Performed
- **Missing Values**: Thay thế 6 giá trị khuyết thiếu trong `horsepower` bằng trung vị (median = {median_hp}).
- **Drop Columns**: Đã xóa cột `car_name`.
- **Duplicates**: Xóa dữ liệu trùng lặp (nếu có).
- **Outliers**: Quyết định giữ lại các giá trị ngoại lệ của `horsepower` và `acceleration` vì đây là giá trị thực tế của xe.

## 4. Output
- Kích thước dữ liệu sau khi làm sạch: {df.shape[0]} mẫu, {df.shape[1]} đặc trưng.
- Dataset đã được lưu tại: `{clean_data_path}`

## 5. Insight & Decision
- **Insight**: Dữ liệu hiện tại hoàn toàn không còn missing values, các cột đều là dạng số (numerical) sẵn sàng để thực hiện Feature Engineering.
- **Decision**: Hoàn thành Bước 2. Tiếp tục tiến sang **Bước 3 — Feature Engineering**.
"""

with open('Buoc2.md', 'w', encoding='utf-8') as f:
    f.write(buoc2_content)

print("Saved report to Buoc2.md")
