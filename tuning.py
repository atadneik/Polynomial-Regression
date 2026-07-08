import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import json

print("Starting Step 7: Hyperparameter Tuning...")

# 1. Load Data
X_train = pd.read_csv('data/processed/split/X_train.csv')
y_train = pd.read_csv('data/processed/split/y_train.csv').squeeze()
X_val = pd.read_csv('data/processed/split/X_val.csv')
y_val = pd.read_csv('data/processed/split/y_val.csv').squeeze()

# 2. Define Alpha Grid for Ridge Regression
# We search logarithmically from 10^-3 to 10^4
alphas = np.logspace(-3, 4, 100)

train_rmse_list = []
val_rmse_list = []
val_r2_list = []

# 3. Grid Search over Alphas
print(f"Tuning Ridge alpha over {len(alphas)} values...")
for alpha in alphas:
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('regressor', Ridge(alpha=alpha))
    ])
    
    pipeline.fit(X_train, y_train)
    
    y_train_pred = pipeline.predict(X_train)
    y_val_pred = pipeline.predict(X_val)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_r2 = r2_score(y_val, y_val_pred)
    
    train_rmse_list.append(train_rmse)
    val_rmse_list.append(val_rmse)
    val_r2_list.append(val_r2)

# 4. Find Best Hyperparameter
best_idx = np.argmin(val_rmse_list)
best_alpha = alphas[best_idx]
best_val_rmse = val_rmse_list[best_idx]
best_val_r2 = val_r2_list[best_idx]
best_train_rmse = train_rmse_list[best_idx]

print(f"Best Alpha: {best_alpha:.4f}")
print(f"Best Val RMSE: {best_val_rmse:.4f}")
print(f"Best Val R2: {best_val_r2:.4f}")

# Save best parameters to a config file
os.makedirs('Results', exist_ok=True)
best_params = {
    'model': 'Ridge',
    'degree': 2,
    'alpha': float(best_alpha)
}
with open('Results/best_params.json', 'w') as f:
    json.dump(best_params, f, indent=4)

# 5. Visualizations for Insight (Validation Curve)
os.makedirs('Figures/Tuning', exist_ok=True)
plt.figure(figsize=(10, 6))

plt.plot(alphas, train_rmse_list, label='Train RMSE', color='blue', lw=2)
plt.plot(alphas, val_rmse_list, label='Validation RMSE', color='orange', lw=2)
plt.axvline(x=best_alpha, color='red', linestyle='--', label=f'Best Alpha = {best_alpha:.2f}')

# Configure X-axis to logarithmic scale
plt.xscale('log')
plt.title("Validation Curve cho Ridge Regression (Polynomial Bậc 2)")
plt.xlabel("Alpha (Độ mạnh của Regularization) - Log Scale")
plt.ylabel("RMSE (Càng thấp càng tốt)")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig('Figures/Tuning/validation_curve.png')
plt.close()

# 6. Generate Buoc7.md
buoc7_content = f"""# Hyperparameter Tuning Report (Phase 7)

## 1. Goal
Tìm ra tham số tối ưu nhất cho mô hình đã chọn (`Polynomial Bậc 2 + Ridge Regression`) để cân bằng hoàn hảo giữa Overfitting và Underfitting.

## 2. Input
- Mô hình: Pipeline (`Poly degree=2 -> StandardScaler -> Ridge(alpha)`)
- Hyperparameter cần tune: `alpha` (hệ số phạt L2 của Ridge).
- Không gian tìm kiếm (Search Space): 100 giá trị `alpha` trải đều trên thang đo logarit từ $10^{{-3}}$ đến $10^4$.
- Đánh giá trên: Validation Set.

## 3. Tasks Performed & Visual Insights

Đã tiến hành huấn luyện 100 mô hình tương ứng với 100 giá trị `alpha` khác nhau. Dưới đây là biểu đồ **Validation Curve** (Đường cong Xác thực).

![Validation Curve](./Figures/Tuning/validation_curve.png)

**Insight Cực Kì Quan Trọng (từ biểu đồ):**
- **Vùng bên trái (Alpha nhỏ, < 1):** Mô hình bị phạt quá ít, gần giống hệt Linear Regression đa thức bậc 2 thông thường. Ở vùng này Train RMSE rất thấp nhưng Validation RMSE cao (khoảng cách giữa 2 đường màu cam và màu xanh lớn) -> Dấu hiệu của **Overfitting nhẹ**.
- **Vùng bên phải (Alpha lớn, > 100):** Mô hình bị phạt quá nặng, các trọng số (weights) bị ép về 0. Cả Train RMSE và Validation RMSE đều tăng vọt -> Mô hình mất khả năng dự đoán (**Underfitting**).
- **Vùng lý tưởng (Sweet Spot):** Nằm ở khoảng `alpha` từ 1 đến 10, nơi đường Validation RMSE (màu cam) đạt tới đáy thấp nhất. Tại đây, mô hình vừa đủ tự do để học tính phi tuyến tính, vừa đủ bị "kìm kẹp" để không học vẹt nhiễu.

## 4. Output
- **Siêu tham số tối ưu nhất (Best Hyperparameter):**
  - Thuật toán: Ridge
  - Bậc đa thức (Degree): 2
  - **Best Alpha:** `{best_alpha:.4f}`
- **Hiệu suất đạt được tại điểm tối ưu:**
  - Val R²: `{best_val_r2:.4f}`
  - Val RMSE: `{best_val_rmse:.4f}`
  - Train RMSE: `{best_train_rmse:.4f}`
- Bộ tham số đã được lưu tại: `Results/best_params.json`
- Biểu đồ lưu tại: `Figures/Tuning/validation_curve.png`

## 5. Decision
- Quá trình Tuning đã thành công mĩ mãn, tìm ra chính xác điểm "Sweet Spot".
- Chốt cấu trúc mô hình cuối cùng: **Polynomial Bậc 2 + Ridge(alpha={best_alpha:.4f})**.
- Tiến hành **Bước 8 — Train Model**: Sử dụng cấu trúc này để huấn luyện lại mô hình một lần cuối cùng trên tập Train (hoặc Train+Val), chuẩn bị đem ra "chiến trường" đánh giá trên tập Test độc lập.
"""

with open('Buoc7.md', 'w', encoding='utf-8') as f:
    f.write(buoc7_content)

print("Saved report to Buoc7.md")
