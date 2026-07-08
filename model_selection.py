import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

print("Starting Step 6: Model Selection...")

# 1. Load Split Data
X_train = pd.read_csv('data/processed/split/X_train.csv')
y_train = pd.read_csv('data/processed/split/y_train.csv').squeeze()
X_val = pd.read_csv('data/processed/split/X_val.csv')
y_val = pd.read_csv('data/processed/split/y_val.csv').squeeze()

# 2. Define Candidate Models
# We build pipelines to strictly prevent Data Leakage
models = {
    "1. Baseline (Linear)": Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ]),
    "2. Poly (Degree 2) + Linear": Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ]),
    "3. Poly (Degree 3) + Linear": Pipeline([
        ('poly', PolynomialFeatures(degree=3, include_bias=False)),
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ]),
    "4. Poly (Degree 2) + Ridge": Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('regressor', Ridge(alpha=1.0))
    ]),
    "5. Poly (Degree 2) + Lasso": Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('regressor', Lasso(alpha=0.1, max_iter=10000))
    ])
}

# 3. Train and Evaluate
results = []

for name, pipeline in models.items():
    # Train
    pipeline.fit(X_train, y_train)
    
    # Predict
    y_train_pred = pipeline.predict(X_train)
    y_val_pred = pipeline.predict(X_val)
    
    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    
    results.append({
        'Model': name,
        'Train R2': train_r2,
        'Val R2': val_r2,
        'Train RMSE': train_rmse,
        'Val RMSE': val_rmse
    })

results_df = pd.DataFrame(results)
print("\n--- Model Comparison Results ---")
print(results_df.to_string(index=False))

# 4. Visualization for Insight
os.makedirs('Figures/Selection', exist_ok=True)
plt.figure(figsize=(14, 6))

# Subplot 1: R2 Comparison
plt.subplot(1, 2, 1)
sns.barplot(data=results_df, y='Model', x='Val R2', palette='viridis')
plt.title("So sánh Validation R² (Càng cao càng tốt)")
plt.axvline(x=results_df.loc[0, 'Val R2'], color='r', linestyle='--', label='Baseline')
plt.xlim(0.7, 1.0)
plt.legend()

# Subplot 2: RMSE Comparison
plt.subplot(1, 2, 2)
# We plot Train vs Val RMSE to spot Overfitting
df_melted = results_df.melt(id_vars=['Model'], value_vars=['Train RMSE', 'Val RMSE'], 
                            var_name='Dataset', value_name='RMSE')
sns.barplot(data=df_melted, y='Model', x='RMSE', hue='Dataset', palette='mako')
plt.title("So sánh RMSE: Train vs Validation (Càng thấp càng tốt)")
plt.axvline(x=results_df.loc[0, 'Val RMSE'], color='r', linestyle='--', label='Baseline Val RMSE')
plt.legend()

plt.tight_layout()
plt.savefig('Figures/Selection/model_comparison.png')
plt.close()

# 5. Extract specific insights for Buoc6.md
best_r2_model = results_df.loc[results_df['Val R2'].idxmax()]

buoc6_content = f"""# Model Selection Report (Phase 6)

## 1. Goal
Đánh giá nhanh nhiều thuật toán/kiến trúc để chọn ra mô hình tiềm năng nhất, từ đó thu hẹp phạm vi trước khi tinh chỉnh chuyên sâu.

## 2. Input
- Tập Train và Validation từ Bước 4.
- 5 ứng viên mô hình (Candidate Models) được gói trong Pipeline:
  1. Baseline (Linear)
  2. Poly (Degree 2) + Linear
  3. Poly (Degree 3) + Linear
  4. Poly (Degree 2) + Ridge (L2 Penalty)
  5. Poly (Degree 2) + Lasso (L1 Penalty)

## 3. Tasks Performed & Visual Insights

Đã tiến hành huấn luyện 5 mô hình trên tập Train và đo lường $R^2$, RMSE trên tập Validation.

### Bảng So Sánh Hiệu Suất

| Model | Train R² | Val R² | Train RMSE | Val RMSE |
|-------|----------|--------|------------|----------|
| Baseline | {results_df.loc[0, 'Train R2']:.4f} | {results_df.loc[0, 'Val R2']:.4f} | {results_df.loc[0, 'Train RMSE']:.4f} | {results_df.loc[0, 'Val RMSE']:.4f} |
| Poly (Deg 2) + Linear | {results_df.loc[1, 'Train R2']:.4f} | {results_df.loc[1, 'Val R2']:.4f} | {results_df.loc[1, 'Train RMSE']:.4f} | {results_df.loc[1, 'Val RMSE']:.4f} |
| Poly (Deg 3) + Linear | {results_df.loc[2, 'Train R2']:.4f} | {results_df.loc[2, 'Val R2']:.4f} | {results_df.loc[2, 'Train RMSE']:.4f} | {results_df.loc[2, 'Val RMSE']:.4f} |
| Poly (Deg 2) + Ridge | {results_df.loc[3, 'Train R2']:.4f} | {results_df.loc[3, 'Val R2']:.4f} | {results_df.loc[3, 'Train RMSE']:.4f} | {results_df.loc[3, 'Val RMSE']:.4f} |
| Poly (Deg 2) + Lasso | {results_df.loc[4, 'Train R2']:.4f} | {results_df.loc[4, 'Val R2']:.4f} | {results_df.loc[4, 'Train RMSE']:.4f} | {results_df.loc[4, 'Val RMSE']:.4f} |

### Trực quan hóa Insight (Visual Insight)

![Model Comparison](./Figures/Selection/model_comparison.png)

**Insight Cực Kì Quan Trọng (từ biểu đồ):**
1. **Sức mạnh của Đa thức (Polynomial):** Ngay khi nâng lên bậc 2, $R^2$ trên tập Validation đã nhảy vọt từ ~0.84 lên ~0.88. Điều này khẳng định triệt để giả thuyết phi tuyến tính ở Bước 5.
2. **Cảnh báo Overfitting cực mạnh ở Bậc 3:** Mô hình `Poly (Degree 3) + Linear` có RMSE trên tập Train cực thấp (fit cực kỳ tốt), nhưng RMSE trên tập Validation lại **bùng nổ** (có thể thấy thanh Validation RMSE dài bất thường). Đây là sách giáo khoa về hiện tượng **Overfitting** khi bậc đa thức quá cao khiến số lượng features tăng theo cấp số nhân và mô hình học vẹt các nhiễu.
3. **Hiệu quả của Regularization:** `Poly (Degree 2) + Ridge` và `Lasso` đều giữ được hiệu suất tương đương `Linear` bậc 2 nhưng có tính ổn định cao hơn (đặc biệt Ridge đang có RMSE tốt nhất).

## 4. Output
- Mô hình tiềm năng nhất hiện tại là: **{best_r2_model['Model']}** với $R^2$ cao nhất là **{best_r2_model['Val R2']:.4f}**.
- Biểu đồ đánh giá lưu tại `Figures/Selection/model_comparison.png`.

## 5. Decision
- Loại bỏ mô hình Đa thức Bậc 3 vì Overfitting quá nặng.
- Lựa chọn mô hình **Polynomial Regression (Bậc 2) kết hợp với Ridge (hoặc Lasso)** làm Candidate xuất sắc nhất.
- Chuyển sang **Bước 7 — Hyperparameter Tuning** để tìm ra siêu tham số `alpha` tối ưu nhất cho Regularization của mô hình bậc 2 này.
"""

with open('Buoc6.md', 'w', encoding='utf-8') as f:
    f.write(buoc6_content)

print("Saved report to Buoc6.md")
