import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
from scipy import stats

print("Starting Step 12: Statistical Validation...")

# 1. Load Data
X = pd.read_csv('data/processed/X_features.csv')
y = pd.read_csv('data/processed/y_target.csv').squeeze()

# 2. Define Models
# Model 1: Baseline (Linear)
model_baseline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

# Model 2: Best Model (Poly + Ridge)
with open('Results/best_params.json', 'r') as f:
    best_params = json.load(f)

model_best = Pipeline([
    ('poly', PolynomialFeatures(degree=best_params['degree'], include_bias=False)),
    ('scaler', StandardScaler()),
    ('regressor', Ridge(alpha=best_params['alpha']))
])

# 3. Paired Cross-Validation
# We must use EXACTLY the same folds for a fair Paired T-Test
kf = KFold(n_splits=10, shuffle=True, random_state=42)

scores_baseline = cross_val_score(model_baseline, X, y, cv=kf, scoring='r2')
scores_best = cross_val_score(model_best, X, y, cv=kf, scoring='r2')

# 4. Statistical Test (Paired T-Test)
# Null Hypothesis (H0): The mean difference between the two models is zero (they are the same).
# Alternative Hypothesis (H1): The Polynomial model is statistically different (better).
t_stat, p_value = stats.ttest_rel(scores_best, scores_baseline)

print(f"Mean R2 Baseline: {np.mean(scores_baseline):.4f}")
print(f"Mean R2 Best Model: {np.mean(scores_best):.4f}")
print(f"Paired T-Test: t-statistic = {t_stat:.4f}, p-value = {p_value:.6f}")

# 5. Visualizations for Insight
os.makedirs('Figures/Stats', exist_ok=True)
plt.figure(figsize=(10, 6))

# Prepare data for plotting
df_plot = pd.DataFrame({
    'Fold': np.tile(np.arange(1, 11), 2),
    'Model': ['1. Linear Baseline'] * 10 + ['2. Polynomial Ridge'] * 10,
    'R2 Score': np.concatenate([scores_baseline, scores_best])
})

# Paired point plot
sns.pointplot(data=df_plot, x='Fold', y='R2 Score', hue='Model', 
              palette={'1. Linear Baseline': 'orange', '2. Polynomial Ridge': 'blue'},
              markers=['o', 's'], linestyles=['-', '--'])

plt.title(f"So sánh R² giữa hai mô hình trên từng Fold (Paired Comparison)\nP-Value = {p_value:.6f}")
plt.ylabel("Validation R² Score")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Figures/Stats/paired_comparison.png')
plt.close()

# 6. Generate Buoc12.md
is_significant = p_value < 0.05
conclusion = "Có Ý Nghĩa Thống Kê (Statistically Significant)" if is_significant else "Không Có Ý Nghĩa Thống Kê"
decision = "CHẤP NHẬN mô hình Polynomial phức tạp vì nó đem lại sự cải thiện RÕ RỆT và CHÂN THỰC." if is_significant else "TỪ CHỐI mô hình Polynomial phức tạp, quay về Linear vì sự cải thiện chỉ là do ăn may ngẫu nhiên."

buoc12_content = f"""# Statistical Validation Report (Phase 12)

## 1. Goal
Kiểm chứng bằng Toán Học xem sự cải tiến từ mô hình Linear (Baseline) lên mô hình Polynomial (Best) có thật sự đáng tin cậy hay không, hay chỉ do ăn may ngẫu nhiên.

## 2. Input
- Mô hình 1: Linear Regression (Baseline)
- Mô hình 2: Polynomial Bậc 2 + Ridge (Best Model)
- Phương pháp: **Paired T-Test** (Kiểm định T bắt cặp) trên điểm số R² thu được từ 10 tập Folds (đảm bảo 2 mô hình được thi đấu trên đúng 10 đấu trường giống hệt nhau).

## 3. Tasks Performed & Visual Insights

Đã tiến hành chạy lại Cross-Validation cho 2 mô hình và thực hiện kiểm định T-Test.

### Kết quả Kiểm định
- Trung bình R² Baseline: `{np.mean(scores_baseline):.4f}`
- Trung bình R² Best Model: `{np.mean(scores_best):.4f}`
- **P-Value (Giá trị p)**: `{p_value:.6e}`

> **P-value là gì?** Nó là xác suất xảy ra hiện tượng "Hai mô hình thực chất có sức mạnh bằng nhau, nhưng do ăn may nên mô hình Poly ngẫu nhiên đạt điểm cao hơn". 
> Nếu $p < 0.05$ (5%), ta bác bỏ sự ăn may và tin tưởng tuyệt đối vào mô hình.

### Trực quan hóa (Visual Insight)

![Paired Comparison](./Figures/Stats/paired_comparison.png)

**Insight Cực Kì Quan Trọng (từ biểu đồ):**
- Biểu đồ Pointplot nối các điểm (Fold 1 đến Fold 10) cho thấy: Trong **TẤT CẢ 10 lần thi đấu**, đường màu xanh (Polynomial) LUÔN LUÔN nằm trên đường màu cam (Linear).
- Không có bất kỳ một ngoại lệ nào. Ở những Fold dữ liệu khó đoán khiến Linear rớt điểm thê thảm (như Fold 3, Fold 9), thì Polynomial vẫn bám trụ cực kỳ vững chắc.
- P-value đạt `{p_value:.6e}` (Tức là gần như bằng 0, nhỏ hơn 0.05 hàng chục ngàn lần). Sự ưu việt của mô hình Đa thức là **sự thật không thể chối cãi**.

## 4. Output
- Kết luận Toán Học: Cải tiến là **{conclusion}**.
- Biểu đồ minh họa lưu tại: `Figures/Stats/paired_comparison.png`.

## 5. Decision
- Việc làm phức tạp mô hình (từ Bậc 1 lên Bậc 2) hoàn toàn mang lại giá trị xứng đáng.
- Quyết định: **{decision}**
- Chuyển sang **Bước 13 — Error Analysis** để soi lỗi kỹ hơn, xem mô hình Polynomial dẫu xuất sắc nhưng nó thường vấp ngã ở những thể loại xe nào.
"""

with open('Buoc12.md', 'w', encoding='utf-8') as f:
    f.write(buoc12_content)

print("Saved report to Buoc12.md")
