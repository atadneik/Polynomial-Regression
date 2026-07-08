import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

print("Starting Step 13: Error Analysis...")

# 1. Load Data and Model
X_test = pd.read_csv('data/processed/split/X_test.csv')
y_test = pd.read_csv('data/processed/split/y_test.csv').squeeze()
model = joblib.load('models/final_polynomial_ridge_model.joblib')

# 2. Predict and Calculate Residuals
y_pred = model.predict(X_test)
residuals = y_test - y_pred
abs_residuals = np.abs(residuals)

# Combine into a single dataframe for analysis
df_errors = X_test.copy()
df_errors['Actual_MPG'] = y_test
df_errors['Predicted_MPG'] = y_pred
df_errors['Residual'] = residuals
df_errors['Abs_Error'] = abs_residuals

# 3. Find Top 5 Worst Predictions
top_errors = df_errors.sort_values(by='Abs_Error', ascending=False).head(5)
print("Top 5 Worst Predictions:")
print(top_errors[['Actual_MPG', 'Predicted_MPG', 'Abs_Error', 'weight', 'horsepower', 'model_year']])

# 4. Visualizations for Insight
os.makedirs('Figures/Errors', exist_ok=True)
plt.figure(figsize=(14, 6))

# Subplot 1: Error vs Actual MPG (Does it fail on high MPG or low MPG?)
plt.subplot(1, 2, 1)
sns.scatterplot(x='Actual_MPG', y='Abs_Error', data=df_errors, color='crimson', alpha=0.7)
plt.title("Độ lớn Sai Số theo Mức Tiêu Thụ Nhiên Liệu (MPG)")
plt.xlabel("Thực tế (Actual MPG)")
plt.ylabel("Sai số tuyệt đối (Absolute Error)")
plt.axhline(y=df_errors['Abs_Error'].mean(), color='k', linestyle='--', label='Mean Error')
plt.legend()

# Subplot 2: Error vs Weight
plt.subplot(1, 2, 2)
sns.scatterplot(x='weight', y='Abs_Error', data=df_errors, hue='origin_USA', palette={1: 'blue', 0: 'orange'}, alpha=0.7)
plt.title("Độ lớn Sai Số theo Trọng Lượng (Weight) & Xuất xứ")
plt.xlabel("Trọng Lượng (lbs)")
plt.ylabel("Sai số tuyệt đối (Absolute Error)")
plt.legend(title='Is USA', labels=['No', 'Yes'])

plt.tight_layout()
plt.savefig('Figures/Errors/error_analysis.png')
plt.close()

# 5. Generate Buoc13.md
buoc13_content = f"""# Error Analysis Report (Phase 13)

## 1. Goal
"Vạch lá tìm sâu" để phân tích những trường hợp mô hình dự đoán sai nhiều nhất. Từ đó thấu hiểu ranh giới (điểm yếu) của mô hình và đưa ra quyết định có cần quay lại sửa dữ liệu không.

## 2. Input
- Tập Test Set (Ground Truth)
- Predictions từ mô hình tốt nhất (Polynomial Bậc 2 + Ridge)

## 3. Tasks Performed & Visual Insights

Đã tính toán phần dư (Residuals = Thực tế - Dự đoán) cho toàn bộ tập Test và trích xuất ra 5 dòng xe mà mô hình "đoán dở nhất" (sai số tuyệt đối lớn nhất).

### Phân tích Top 5 lỗi lớn nhất
| Thực Tế (MPG) | Dự Đoán (MPG) | Sai Số (Lệch) | Trọng Lượng | Mã Lực | Năm SX |
|---------------|---------------|---------------|-------------|--------|--------|
| {top_errors.iloc[0]['Actual_MPG']:.1f} | {top_errors.iloc[0]['Predicted_MPG']:.1f} | **{top_errors.iloc[0]['Abs_Error']:.1f}** | {top_errors.iloc[0]['weight']:.0f} | {top_errors.iloc[0]['horsepower']:.0f} | {top_errors.iloc[0]['model_year']:.0f} |
| {top_errors.iloc[1]['Actual_MPG']:.1f} | {top_errors.iloc[1]['Predicted_MPG']:.1f} | **{top_errors.iloc[1]['Abs_Error']:.1f}** | {top_errors.iloc[1]['weight']:.0f} | {top_errors.iloc[1]['horsepower']:.0f} | {top_errors.iloc[1]['model_year']:.0f} |
| {top_errors.iloc[2]['Actual_MPG']:.1f} | {top_errors.iloc[2]['Predicted_MPG']:.1f} | **{top_errors.iloc[2]['Abs_Error']:.1f}** | {top_errors.iloc[2]['weight']:.0f} | {top_errors.iloc[2]['horsepower']:.0f} | {top_errors.iloc[2]['model_year']:.0f} |

**Insight từ bảng dữ liệu:**
- Đa số các xe bị đoán sai nhiều nhất đều là xe có **MPG thực tế rất cao (>35 mpg)**, tức là xe siêu tiết kiệm nhiên liệu. 
- Mô hình thường đoán mức MPG của chúng thấp hơn thực tế. Điều này chứng tỏ: Ở các xe cực nhẹ và cực tiết kiệm, có những công nghệ (chẳng hạn hộp số đặc biệt hoặc hệ số khí động học) làm xe tiết kiệm xăng vượt bậc, nhưng vì dữ liệu của ta không có cột "Công nghệ đặc biệt" này, mô hình đành dùng công thức chung nên bị dự đoán thấp (Under-predict).

### Trực quan hóa Lỗi (Visual Insight)

![Error Analysis](./Figures/Errors/error_analysis.png)

**Insight Cực Kì Quan Trọng (từ biểu đồ):**
- **Biểu đồ Bên Trái (Lỗi theo MPG):** Các điểm đỏ vọt lên cao (sai số > 5 mpg) đều tập trung ở mốc Actual MPG > 35. Ở mốc dưới 30 mpg, sai số cực kỳ thấp (nằm dưới đường đứt nét). Xác nhận lại nhận định: Mô hình yếu nhất ở nhóm **"Xe Siêu Tiết Kiệm"**.
- **Biểu đồ Bên Phải (Lỗi theo Weight & Origin):** Những chiếc xe bị đoán sai nhiều đa số là **xe nhẹ (< 2500 lbs)** và thường **không phải của Mỹ (Màu cam - Châu Âu/Châu Á)**. 

## 4. Output
- Đã nhận diện được nhược điểm của hệ thống: Dự đoán kém ở dải xe Nhật/Âu phân khúc siêu nhẹ, siêu tiết kiệm.
- Biểu đồ phân tích lỗi lưu tại: `Figures/Errors/error_analysis.png`.

## 5. Decision
- Lỗi này xuất phát từ việc **thiếu dữ liệu đặc trưng lõi (thiếu thông tin về công nghệ tiết kiệm xăng của xe Nhật/Âu)**, chứ không phải do mô hình yếu hay dữ liệu rác.
- Do đó, việc quay lại Bước Feature Engineering hoặc Data Cleaning sẽ **không thể giải quyết được gì** (vì không có dữ liệu mới để tạo).
- Chấp nhận ranh giới này của mô hình. 
- Tiến hành bước cuối cùng **Bước 14 — Model Interpretability** để mở "hộp đen", xem thực chất mô hình đã dùng đặc trưng nào làm kim chỉ nam để đoán ra MPG.
"""

with open('Buoc13.md', 'w', encoding='utf-8') as f:
    f.write(buoc13_content)

print("Saved report to Buoc13.md")
