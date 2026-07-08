import pandas as pd
import numpy as np
import os
import json
import time
import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

print("Starting Step 8: Train Final Model...")

# 1. Load Data
# Khác với lúc dò tìm tham số (chỉ dùng Train set), khi đã chốt mô hình, 
# ta sẽ GỘP (merge) tập Train và Validation lại để có nhiều data nhất có thể cho việc fit mô hình cuối.
X_train = pd.read_csv('data/processed/split/X_train.csv')
y_train = pd.read_csv('data/processed/split/y_train.csv').squeeze()
X_val = pd.read_csv('data/processed/split/X_val.csv')
y_val = pd.read_csv('data/processed/split/y_val.csv').squeeze()

X_train_full = pd.concat([X_train, X_val], ignore_index=True)
y_train_full = pd.concat([y_train, y_val], ignore_index=True)

print(f"Full Training Data Shape: {X_train_full.shape}")

# 2. Load Best Hyperparameters
with open('Results/best_params.json', 'r') as f:
    best_params = json.load(f)

print(f"Loaded Hyperparameters: {best_params}")

# 3. Construct Final Pipeline
final_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=best_params['degree'], include_bias=False)),
    ('scaler', StandardScaler()),
    ('regressor', Ridge(alpha=best_params['alpha']))
])

# 4. Train the Model and measure time
start_time = time.time()
final_pipeline.fit(X_train_full, y_train_full)
end_time = time.time()

training_time = (end_time - start_time) * 1000 # in milliseconds

print(f"Model trained in {training_time:.2f} ms")

# 5. Export the Trained Model
os.makedirs('models', exist_ok=True)
model_path = 'models/final_polynomial_ridge_model.joblib'
joblib.dump(final_pipeline, model_path)
print(f"Model exported to {model_path}")

# 6. Generate Buoc8.md
buoc8_content = f"""# Final Model Training Report (Phase 8)

## 1. Goal
Huấn luyện và đóng gói (export) mô hình hoàn chỉnh với cấu hình tối ưu nhất đã tìm được ở Bước 7, sẵn sàng cho việc đánh giá và triển khai.

## 2. Input
- **Dataset**: Tập `X_train_full` (Gộp từ Train và Validation, tổng cộng {X_train_full.shape[0]} mẫu).
- **Hyperparameters**: Load từ `Results/best_params.json`:
  - `model`: {best_params['model']}
  - `degree`: {best_params['degree']}
  - `alpha`: {best_params['alpha']}

## 3. Tasks Performed & Insights

- **Chiến thuật Gộp Dữ Liệu (Merge Data):** Thay vì chỉ train trên tập `X_train` (278 mẫu) như lúc dò tham số, mình đã gộp thêm tập `X_val` (60 mẫu) vào để tạo thành `X_train_full` (338 mẫu). 
  - *Insight*: Khi cấu trúc mô hình đã được chốt, tập Validation không còn tác dụng để tune nữa. Việc gộp nó vào tập Train giúp mô hình học được nhiều dữ liệu hơn (tăng 21% dung lượng data), từ đó củng cố độ vững vàng (robustness) trước khi chạm trán tập Test.
- **Theo dõi sự hội tụ (Convergence & Time):**
  - Khởi tạo `Pipeline` với toàn bộ cấu trúc: Sinh đa thức bậc 2 $\\rightarrow$ Chuẩn hóa Z-score $\\rightarrow$ Hồi quy Ridge.
  - *Insight*: Thời gian huấn luyện (fit) toàn bộ Pipeline chỉ mất **{training_time:.2f} mili-giây (ms)**. Ridge Regression là thuật toán dạng Closed-form solution (hoặc dùng Cholesky solver cực nhanh) nên sự hội tụ diễn ra gần như ngay lập tức, không tốn tài nguyên tính toán như các mô hình Neural Network. Rất lý tưởng để triển khai thực tế.

## 4. Output
- Mô hình đã được huấn luyện thành công.
- Export mô hình nguyên khối dưới định dạng Joblib: `models/final_polynomial_ridge_model.joblib`

## 5. Decision
- Mô hình đã sẵn sàng, hội tụ cực tốt và đã được đóng gói an toàn.
- Chuyển sang "Chiến trường thực sự": **Bước 9 — Evaluation Metrics** để mở niêm phong tập Test và đánh giá hiệu năng thật sự của mô hình dự án này!
"""

with open('Buoc8.md', 'w', encoding='utf-8') as f:
    f.write(buoc8_content)

print("Saved report to Buoc8.md")
