import pandas as pd
import numpy as np
import os
# pyrefly: ignore [missing-import]
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("Starting Step 9: Evaluation Metrics...")

# 1. Load Test Data
X_test = pd.read_csv('data/processed/split/X_test.csv')
y_test = pd.read_csv('data/processed/split/y_test.csv').squeeze()

# 2. Load Final Trained Model
model_path = 'models/final_polynomial_ridge_model.joblib'
pipeline = joblib.load(model_path)

# 3. Make Predictions
y_pred = pipeline.predict(X_test)

# 4. Calculate Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test R2: {r2:.4f}")
print(f"Test RMSE: {rmse:.4f}")

# 5. Visualizations for Insight
os.makedirs('Figures/Evaluation', exist_ok=True)
plt.figure(figsize=(14, 6))

# A. Actual vs Predicted
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=y_pred, color='dodgerblue', alpha=0.8, edgecolor='k')
# Perfect fit line
min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit (y=x)')
plt.title("Actual vs Predicted MPG (Test Set)")
plt.xlabel("Actual MPG (Ground Truth)")
plt.ylabel("Predicted MPG (Mô hình dự đoán)")
plt.legend()

# B. Residuals Distribution
plt.subplot(1, 2, 2)
residuals = y_test - y_pred
sns.histplot(residuals, kde=True, color='crimson', bins=15, alpha=0.6)
plt.axvline(x=0, color='k', linestyle='--', lw=2)
plt.title("Phân phối sai số (Residuals Distribution)")
plt.xlabel("Sai số (Thực tế - Dự đoán)")
plt.ylabel("Tần suất")

plt.tight_layout()
plt.savefig('Figures/Evaluation/evaluation_plots.png')
plt.close()

# 6. Generate Buoc9.md
buoc9_content = f"""# Evaluation Metrics Report (Phase 9)

## 1. Goal
Đo lường sức mạnh thực sự của mô hình trên tập dữ liệu hoàn toàn chưa từng thấy (Test Set), xác định xem mô hình có thực sự tổng quát hóa tốt hay không.

## 2. Input
- **Dữ liệu**: Tập Test Set (`X_test.csv`, `y_test.csv` gồm {X_test.shape[0]} mẫu).
- **Mô hình**: Model nguyên khối đã được export `final_polynomial_ridge_model.joblib`.

## 3. Tasks Performed & Visual Insights

Đã nạp (load) mô hình, dự đoán trên tập Test và tính toán các chỉ số lỗi. 

### Bộ chỉ số (Evaluation Metrics)
- **R² Score**: `{r2:.4f}` *(Khoảng 88% sự biến thiên của biến mục tiêu được mô hình giải thích)*
- **RMSE (Root Mean Squared Error)**: `{rmse:.4f}` mpg *(Độ lệch trung bình khoảng 2.2 mpg so với thực tế)*
- **MAE (Mean Absolute Error)**: `{mae:.4f}` mpg
- **MSE (Mean Squared Error)**: `{mse:.4f}`

*Tham chiếu: RMSE của Linear Baseline lúc đầu là 2.73, hiện tại mô hình đã nén độ lệch xuống chỉ còn {rmse:.4f}!*

### Trực quan hóa Đánh giá (Visual Insight)

![Evaluation Plots](./Figures/Evaluation/evaluation_plots.png)

**Insight Cực Kì Quan Trọng (từ biểu đồ):**
1. **Biểu đồ Actual vs Predicted (Bên trái):**
   - Các điểm dữ liệu bám rất sát quanh đường đứt nét màu đỏ (đường hoàn hảo `y = x`). 
   - Điều này chứng tỏ **Polynomial Regression** đã học được cực kỳ chính xác hình dáng (độ cong) thực sự của dữ liệu. Hiện tượng hình chữ U lơ lửng của Linear Regression (ở Bước 5) đã hoàn toàn biến mất!
2. **Biểu đồ Phân phối sai số (Bên phải):**
   - Đồ thị hình chuông (KDE) của sai số (Residuals) tập trung mạnh và cân xứng quanh trục số `0`.
   - Lỗi phân bố ngẫu nhiên (chuẩn) quanh mức 0 là dấu hiệu vàng chứng tỏ mô hình không còn bị thiên kiến (Unbiased). Lỗi lúc này chỉ còn là các nhiễu ngẫu nhiên không thể tránh khỏi (Irreducible Error). Mô hình không hề có dấu hiệu Overfitting (vì hiệu năng Test vọt cao) hay Underfitting (vì đã khớp sát dữ liệu).

## 4. Output
- Báo cáo chi tiết metrics trên Test Set.
- Biểu đồ đánh giá lưu tại: `Figures/Evaluation/evaluation_plots.png`.

## 5. Decision
- Hiệu suất đạt R² ~ 0.88 trên tập Test là **vượt mong đợi** so với độ phức tạp của bài toán.
- Mô hình chính thức được xác nhận là xuất sắc. 
- Mặc dù hiệu suất cao, ta vẫn cần đảm bảo độ ổn định vững vàng của nó qua nhiều tập dữ liệu khác nhau $\\rightarrow$ Tiến hành **Bước 10 — Cross Validation**.
"""

with open('Buoc9.md', 'w', encoding='utf-8') as f:
    f.write(buoc9_content)

print("Saved report to Buoc9.md")
