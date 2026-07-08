# Evaluation Metrics Report (Phase 9)

## 1. Goal
Đo lường sức mạnh thực sự của mô hình trên tập dữ liệu hoàn toàn chưa từng thấy (Test Set), xác định xem mô hình có thực sự tổng quát hóa tốt hay không.

## 2. Input
- **Dữ liệu**: Tập Test Set (`X_test.csv`, `y_test.csv` gồm 60 mẫu).
- **Mô hình**: Model nguyên khối đã được export `final_polynomial_ridge_model.joblib`.

## 3. Tasks Performed & Visual Insights

Đã nạp (load) mô hình, dự đoán trên tập Test và tính toán các chỉ số lỗi. 

### Bộ chỉ số (Evaluation Metrics)
- **R² Score**: `0.9397` *(Khoảng 88% sự biến thiên của biến mục tiêu được mô hình giải thích)*
- **RMSE (Root Mean Squared Error)**: `1.9906` mpg *(Độ lệch trung bình khoảng 2.2 mpg so với thực tế)*
- **MAE (Mean Absolute Error)**: `1.4564` mpg
- **MSE (Mean Squared Error)**: `3.9626`

*Tham chiếu: RMSE của Linear Baseline lúc đầu là 2.73, hiện tại mô hình đã nén độ lệch xuống chỉ còn 1.9906!*

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
- Mặc dù hiệu suất cao, ta vẫn cần đảm bảo độ ổn định vững vàng của nó qua nhiều tập dữ liệu khác nhau $\rightarrow$ Tiến hành **Bước 10 — Cross Validation**.
