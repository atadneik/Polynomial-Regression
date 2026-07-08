# Data Cleaning Report (Phase 2)

## 1. Goal
Làm sạch và chuẩn hóa dữ liệu để đảm bảo mô hình không học từ dữ liệu nhiễu ("Garbage In, Garbage Out").

## 2. Input
- Dataset từ Bước 1.
- Insight: 6 dòng khuyết `horsepower`. Cột `car_name` là nhiễu.

## 3. Tasks Performed & Insights
- **Xử lý Missing Values**: Thay thế 6 giá trị khuyết thiếu trong `horsepower` bằng trung vị (median = 93.5). 
  - *Insight*: Dùng median thay vì mean giúp chống lại sự kéo lệch của các xe có horsepower quá cao (outlier).
- **Xóa cột định danh**: Xóa `car_name`.
  - *Insight*: Tên xe là duy nhất cho mỗi dòng, nếu để nguyên mô hình sẽ cố gắng overfit vào từng tên xe, dẫn đến mất khả năng tổng quát hóa (generalization).
- **Outliers**: Giữ nguyên.
  - *Insight*: Chấp nhận tính đa dạng của dữ liệu vì các xe thể thao thực sự có horsepower cao, mô hình cần học cách dự đoán cho cả những xe này.

## 4. Output
- Dữ liệu sạch: 398 mẫu, 8 đặc trưng. Đã lưu tại `data/processed/clean_auto_mpg.csv`.

## 5. Decision
- Hoàn thành làm sạch. Dữ liệu đã là dạng số chuẩn. 
- Chuyển sang **Bước 3 — Feature Engineering**.
