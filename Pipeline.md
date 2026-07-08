# Machine Learning Pipeline Rulebook

## Purpose
Pipeline chuẩn theo nguyên tắc **Controlled Experiment**: chỉ thay đổi 1 biến ở mỗi lần thử nghiệm.

## 1. EDA
- Goal: Hiểu dữ liệu.
- Tasks: Missing, outlier, distribution, correlation, leakage.
- Output: Data Profile.
- Insight: Hiểu chất lượng dữ liệu và giả thuyết ban đầu.

## 2. Data Cleaning
- Goal: Làm sạch dữ liệu.
- Tasks: Missing, duplicate, outlier, datatype.
- Output: Clean Dataset.
- Insight: Dữ liệu đủ tin cậy.

## 3. Feature Engineering
- Goal: Tạo đặc trưng.
- Tasks: Encoding, scaling, feature creation, selection.
- Output: Feature Set.
- Insight: Feature nào mang nhiều thông tin.

## 4. Data Split
- Goal: Chia Train/Validation/Test.
- Output: Dataset Split.
- Insight: Không leakage.

## 5. Baseline Model
- Goal: Tạo mốc so sánh.
- Output: Baseline Metrics.
- Insight: Có benchmark.

## 6. Model Selection
- Goal: Chọn thuật toán.
- Output: Candidate Models.
- Insight: Thuật toán phù hợp.

## 7. Hyperparameter Tuning
- Goal: Tối ưu tham số.
- Output: Best Hyperparameters.
- Insight: Cấu hình tốt nhất.

## 8. Train Model
- Goal: Train model cuối.
- Output: Trained Model.
- Insight: Model sẵn sàng đánh giá.

## 9. Evaluation Metrics
- Goal: Đánh giá.
- Output: Metrics Report.
- Insight: Điểm mạnh/yếu.

## 10. Cross Validation
- Goal: Kiểm tra độ ổn định.
- Output: Mean ± Std.
- Insight: Khả năng tổng quát hóa.

## 11. Experiment Management
- Goal: Log toàn bộ.
- Output: Experiment History.
- Insight: Có thể tái lập.

## 12. Statistical Validation
- Goal: Kiểm chứng thống kê.
- Output: Statistical Report.
- Insight: Improvement có ý nghĩa hay không.

## 13. Error Analysis
- Goal: Phân tích lỗi.
- Output: Error Categories.
- Insight: Biết cần cải thiện gì.

## 14. Model Interpretability
- Goal: Giải thích mô hình.
- Tasks: SHAP, LIME, Feature Importance.
- Output: Explainability Report + Best Model.
- Insight: Tin tưởng và triển khai.

# Vibe Coding Checklist

Mỗi bước phải có:
1. Goal
2. Input
3. Tasks
4. Output
5. Insight
6. Decision
7. Artifact

## Rules
- Chỉ thay đổi 1 biến mỗi experiment.
- Không bỏ qua bước.
- Log dataset, feature, model, metric.
- Chỉ chuyển bước khi artifact đã hoàn thành.
