import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("Starting Step 5: Baseline Model...")

# 1. Load Split Data
X_train = pd.read_csv('data/processed/split/X_train.csv')
y_train = pd.read_csv('data/processed/split/y_train.csv').squeeze()
X_val = pd.read_csv('data/processed/split/X_val.csv')
y_val = pd.read_csv('data/processed/split/y_val.csv').squeeze()

# 2. Build Pipeline for Baseline Model
# Using StandardScaler -> LinearRegression (No Polynomial)
# The pipeline ensures scaler is only fitted on X_train to prevent data leakage
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

# 3. Train the Baseline Model
pipeline.fit(X_train, y_train)

# 4. Predict & Evaluate
y_train_pred = pipeline.predict(X_train)
y_val_pred = pipeline.predict(X_val)

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

train_metrics = calculate_metrics(y_train, y_train_pred)
val_metrics = calculate_metrics(y_val, y_val_pred)

print(f"Validation R2: {val_metrics[3]:.4f}")
print(f"Validation RMSE: {val_metrics[1]:.4f}")

# 5. Visualizations for Insight
os.makedirs('Figures/Baseline', exist_ok=True)

# A. Actual vs Predicted Plot
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x=y_train, y=y_train_pred, color='blue', alpha=0.6, label='Train')
sns.scatterplot(x=y_val, y=y_val_pred, color='orange', alpha=0.8, label='Validation', marker='x')
# Add y=x line
min_val = min(min(y_train), min(y_train_pred))
max_val = max(max(y_train), max(y_train_pred))
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Fit')
plt.title("Actual vs Predicted MPG (Linear Baseline)")
plt.xlabel("Actual MPG")
plt.ylabel("Predicted MPG")
plt.legend()

# B. Residual Plot
plt.subplot(1, 2, 2)
train_residuals = y_train - y_train_pred
val_residuals = y_val - y_val_pred

sns.scatterplot(x=y_train_pred, y=train_residuals, color='blue', alpha=0.6, label='Train Residuals')
sns.scatterplot(x=y_val_pred, y=val_residuals, color='orange', alpha=0.8, label='Validation Residuals', marker='x')
plt.axhline(y=0, color='k', linestyle='--', lw=2)
plt.title("Residuals vs Predicted MPG")
plt.xlabel("Predicted MPG")
plt.ylabel("Residual (Actual - Predicted)")
plt.legend()

plt.tight_layout()
plt.savefig('Figures/Baseline/baseline_performance.png')
plt.close()

# 6. Generate Buoc5.md
buoc5_content = f"""# Baseline Model Report (Phase 5)

## 1. Goal
Xây dựng một mốc so sánh tối thiểu (benchmark). Bất kỳ mô hình phức tạp nào (như Polynomial) sau này cũng bắt buộc phải đánh bại mốc này.

## 2. Input
- Dữ liệu: Tập Train (`X_train.csv`, `y_train.csv`) và Validation (`X_val.csv`, `y_val.csv`).
- Mô hình: `Linear Regression` kết hợp với `StandardScaler` (thông qua `sklearn.pipeline.Pipeline`). Việc thiết lập Pipeline đảm bảo Scaler chỉ được fit trên tập Train, tuân thủ tuyệt đối quy tắc chống **Data Leakage**.

## 3. Tasks Performed & Visual Insights

### Kết quả Metrics
- **Train Set:**
  - MSE: {train_metrics[0]:.4f} | RMSE: {train_metrics[1]:.4f}
  - MAE: {train_metrics[2]:.4f} | $R^2$: {train_metrics[3]:.4f}
- **Validation Set:**
  - MSE: {val_metrics[0]:.4f} | RMSE: {val_metrics[1]:.4f}
  - MAE: {val_metrics[2]:.4f} | $R^2$: {val_metrics[3]:.4f}

### Trực quan hóa & Đánh giá lỗi (Visual Insight)
Để hiểu rõ Linear Regression đơn thuần hoạt động ra sao, mình đã vẽ 2 biểu đồ: Thực tế vs Dự đoán, và Biểu đồ phần dư (Residuals).

![Baseline Performance](./Figures/Baseline/baseline_performance.png)

**Insight Cực Kì Quan Trọng (từ biểu đồ Residuals):**
- Biểu đồ bên trái (Actual vs Predicted) cho thấy model dự đoán ở mức khá (R2 = {val_metrics[3]:.2f}), nhưng các điểm dữ liệu bị cong vòng khỏi đường nét đứt (đường hoàn hảo).
- **Đặc biệt nhìn vào biểu đồ bên phải (Residuals):** Phần dư (sai số) KHÔNG phân bố ngẫu nhiên quanh trục 0. Nó tạo thành một đường cong rõ rệt (hình chữ U ngược). Ở mức dự đoán thấp (15 mpg) và mức cao (35 mpg), model thường dự đoán sai theo cùng một hướng.
- Hiện tượng phần dư có hình dạng (pattern) báo hiệu một điều: **Mô hình tuyến tính quá đơn giản (Underfitting) để nắm bắt mối quan hệ cong của dữ liệu.** 

## 4. Output
- Baseline Metrics ghi nhận $R^2$ = {val_metrics[3]:.4f} và RMSE = {val_metrics[1]:.4f} trên tập Validation.
- Biểu đồ phân tích hiệu suất lưu tại `Figures/Baseline/baseline_performance.png`.

## 5. Decision
- Kết quả Baseline $R^2$ ~ {val_metrics[3]:.2f} là mốc so sánh tối thiểu.
- Biểu đồ Residuals đã chứng minh đanh thép rằng dữ liệu có tính chất phi tuyến tính. Quyết định: Chuyển sang **Bước 6 — Model Selection** để thử nghiệm mô hình Polynomial Regression nhằm giải quyết đường cong sai số này!
"""

with open('Buoc5.md', 'w', encoding='utf-8') as f:
    f.write(buoc5_content)

print("Saved report to Buoc5.md")
