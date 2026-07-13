# ĐỀ CƯƠNG BÀI TIỂU LUẬN: MÔ HÌNH HỒI QUY ĐA THỨC (POLYNOMIAL REGRESSION) ÁP DỤNG QUY TRÌNH 14 BƯỚC

Dưới đây là cấu trúc chi tiết để bạn viết báo cáo tiểu luận. Bản nâng cấp này đã được bổ sung hướng dẫn **CHÈN MÃ CODE (CODE SNIPPETS)** vào đúng các trọng điểm kỹ thuật, giúp giảng viên chấm bài thấy rõ năng lực lập trình và sự am hiểu thuật toán của bạn.

---

## PHẦN 1: MỞ ĐẦU
**1.1. Giới thiệu bài toán:** 
- Đặt vấn đề: Tầm quan trọng của việc dự đoán mức tiêu thụ nhiên liệu (MPG).
- Mục tiêu: Áp dụng quy trình chuẩn 14-step ML Pipeline.
**1.2. Giới thiệu tập dữ liệu (Auto MPG):**
- Trình bày nguồn gốc dữ liệu, 398 mẫu, các đặc trưng ban đầu.

---

## PHẦN 2: CƠ SỞ LÝ THUYẾT
**2.1. Từ Linear đến Polynomial Regression:**
- Giới thiệu công thức toán học và lý do cần mở rộng từ không gian tuyến tính sang phi tuyến.
**2.2. Ridge Regression & Regularization:**
- Giải thích hình phạt L2 (L2 Penalty) và cách nó giúp kiểm soát sự bùng nổ của các trọng số đa thức.
**2.3. Bias-Variance Tradeoff:**
- Sự đánh đổi giữa độ chệch và phương sai khi tăng bậc đa thức (Underfitting vs Overfitting).

---

## PHẦN 3: CHUẨN BỊ VÀ TIỀN XỬ LÝ DỮ LIỆU (DATA PREPARATION)

**3.1. Khám phá dữ liệu (EDA - Bước 1)**
- *Hình ảnh:* Biểu đồ phân phối, Ma trận tương quan.
- *Insight:* Đa cộng tuyến và tính phi tuyến tính.

**3.2. Làm sạch dữ liệu (Data Cleaning - Bước 2)**
- *Đoạn code cần chèn:* Kỹ thuật điền giá trị khuyết (Median Imputation).
  ```python
  # Thay thế missing value ở horsepower bằng Median để chống nhiễu từ outlier
  median_hp = df['horsepower'].median()
  df['horsepower'] = df['horsepower'].fillna(median_hp)
  df.drop(columns=['car_name'], inplace=True) # Bỏ cột định danh
  ```

**3.3. Kỹ thuật Đặc trưng (Feature Engineering - Bước 3)**
- *Đoạn code cần chèn:* Tạo tính năng mới từ domain knowledge.
  ```python
  # One-Hot Encoding cho origin
  X = pd.get_dummies(X, columns=['origin'])
  # Feature Creation: Tỉ lệ Trọng lượng / Mã lực
  X['weight_per_hp'] = X['weight'] / X['horsepower']
  X['displacement_per_cylinder'] = X['displacement'] / X['cylinders']
  ```
- *Insight:* Giải thích tại sao `weight_per_hp` lại quan trọng.

**3.4. Phân chia dữ liệu (Data Split - Bước 4)**
- *Đoạn code cần chèn:* Chia Train/Val/Test an toàn.
  ```python
  # Chia 70% Train, 30% Temp
  X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
  # Chia tiếp Temp thành 15% Val và 15% Test
  X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
  ```
- *Hình ảnh:* Biểu đồ Density plot so sánh phân phối của 3 tập.

---

## PHẦN 4: XÂY DỰNG VÀ LỰA CHỌN MÔ HÌNH (MODELING)

**4.1. Mô hình Cơ sở (Baseline Model - Bước 5)**
- *Hình ảnh:* Biểu đồ Residuals hình chữ U. Luận điểm chốt hạ chứng minh tính phi tuyến tính.

**4.2. Lựa chọn Mô hình (Model Selection - Bước 6)**
- So sánh các ứng viên (Linear, Poly bậc 2, Poly bậc 3, Poly Ridge, Poly Lasso).
- *Hình ảnh:* Biểu đồ so sánh RMSE giữa Bậc 2 và thảm họa Bậc 3 (Overfitting). Khẳng định chọn Poly Bậc 2 + Ridge.

**4.3. Tinh chỉnh Siêu tham số (Hyperparameter Tuning - Bước 7)**
- *Đoạn code cần chèn:* Quá trình quét logarit tìm Alpha.
  ```python
  alphas = np.logspace(-3, 4, 100) # Quét 100 giá trị từ 0.001 đến 10000
  for alpha in alphas:
      pipeline.set_params(regressor__alpha=alpha)
      pipeline.fit(X_train, y_train)
      # Tính RMSE...
  ```
- *Hình ảnh:* Biểu đồ Validation Curve võng hình thung lũng.

**4.4. Huấn luyện Mô hình Cuối cùng (Final Model Training - Bước 8)**
- *Đoạn code cần chèn (CỰC KỲ QUAN TRỌNG):* Code thiết lập Pipeline chống Data Leakage.
  ```python
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import StandardScaler, PolynomialFeatures
  from sklearn.linear_model import Ridge
  
  # Pipeline gom biến đổi đa thức, chuẩn hóa và hồi quy vào 1 khối
  final_pipeline = Pipeline([
      ('poly', PolynomialFeatures(degree=2, include_bias=False)),
      ('scaler', StandardScaler()),
      ('regressor', Ridge(alpha=0.2984))
  ])
  # Huấn luyện trên tập gộp Train + Val
  final_pipeline.fit(X_train_full, y_train_full)
  ```

---

## PHẦN 5: KIỂM ĐỊNH VÀ ĐÁNH GIÁ (EVALUATION & VALIDATION)

**5.1. Đánh giá trên tập Test (Evaluation Metrics - Bước 9)**
- *Hình ảnh:* Đồ thị Actual vs Predicted và Biểu đồ sai số hình chuông chuẩn.
- *Kết quả:* R² ~ 0.94.

**5.2. Đánh giá chéo K-Fold (Cross Validation - Bước 10)**
- *Đoạn code cần chèn:* Code chạy CV chuẩn mực.
  ```python
  from sklearn.model_selection import cross_validate, KFold
  kf = KFold(n_splits=10, shuffle=True, random_state=42)
  cv_results = cross_validate(final_pipeline, X, y, cv=kf, scoring='r2')
  ```
- *Hình ảnh:* Biểu đồ Boxplot của 10-Fold CV (Giải thích mốc kỳ vọng 87%).

**5.3. Quản lý Thí nghiệm (Experiment Management - Bước 11)**
- Ghi log các tham số, siêu tham số, đặc trưng sử dụng và kết quả đánh giá (JSON format).
- Khẳng định tính tái lập của thí nghiệm và xác nhận mức trần 87% do giới hạn vật lý.

**5.4. Kiểm định thống kê (Statistical Validation - Bước 12)**
- *Đoạn code cần chèn:* Dùng T-Test để lấy P-Value.
  ```python
  from scipy import stats
  # Kiểm định T bắt cặp (Paired T-Test) giữa Linear và Polynomial
  t_stat, p_value = stats.ttest_rel(scores_poly, scores_linear)
  ```
- *Phân tích Insight:* P-value < 0.05 (thực tế 0.0016) chứng minh việc mô hình Poly mạnh hơn Linear là có ý nghĩa thống kê.

**5.5. Phân tích Lỗi và Ranh giới mô hình (Error Analysis - Bước 13)**
- *Insight:* Lý thuyết **Omitted Variable Bias** (Thiếu thông tin cản gió, hộp số) làm mô hình đánh giá sai các xe nhỏ, tiết kiệm nhiên liệu xuất xứ Nhật/Âu.

---

## PHẦN 6: GIẢI THÍCH MÔ HÌNH VÀ KẾT LUẬN

**6.1. Mở hộp đen Mô hình (Model Interpretability - Bước 14)**
- *Đoạn code cần chèn:* Cách dịch ngược tên đặc trưng từ Polynomial.
  ```python
  poly = final_pipeline.named_steps['poly']
  ridge = final_pipeline.named_steps['regressor']
  poly_features = poly.get_feature_names_out(original_features)
  coefficients = ridge.coef_
  ```
- *Hình ảnh:* Đồ thị thanh ngang Feature Importance.
- *Phân tích:* Khẳng định `weight_per_hp` là đặc trưng Vàng, và mô hình cực kỳ ghét `weight`.

**6.2. Tổng kết chung**
- Bảng Tổng Kết kết quả số liệu (So sánh Test R², Test RMSE, CV R², P-Value giữa Baseline và Polynomial Ridge).
- Tổng kết 14 bước. Bài học về Data Leakage và chống Overfitting.
