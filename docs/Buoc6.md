# Model Selection Report (Phase 6)

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
| Baseline | 0.8454 | 0.8481 | 3.1068 | 2.7304 |
| Poly (Deg 2) + Linear | 0.9135 | 0.8638 | 2.3247 | 2.5851 |
| Poly (Deg 3) + Linear | 0.9744 | -93.5356 | 1.2642 | 68.1064 |
| Poly (Deg 2) + Ridge | 0.8820 | 0.8696 | 2.7143 | 2.5295 |
| Poly (Deg 2) + Lasso | 0.8498 | 0.8276 | 3.0628 | 2.9085 |

### Trực quan hóa Insight (Visual Insight)

![Model Comparison](./Figures/Selection/model_comparison.png)

**Insight Cực Kì Quan Trọng (từ biểu đồ):**
1. **Sức mạnh của Đa thức (Polynomial):** Ngay khi nâng lên bậc 2, $R^2$ trên tập Validation đã nhảy vọt từ ~0.84 lên ~0.88. Điều này khẳng định triệt để giả thuyết phi tuyến tính ở Bước 5.
2. **Cảnh báo Overfitting cực mạnh ở Bậc 3:** Mô hình `Poly (Degree 3) + Linear` có RMSE trên tập Train cực thấp (fit cực kỳ tốt), nhưng RMSE trên tập Validation lại **bùng nổ** (có thể thấy thanh Validation RMSE dài bất thường). Đây là sách giáo khoa về hiện tượng **Overfitting** khi bậc đa thức quá cao khiến số lượng features tăng theo cấp số nhân và mô hình học vẹt các nhiễu.
3. **Hiệu quả của Regularization:** `Poly (Degree 2) + Ridge` và `Lasso` đều giữ được hiệu suất tương đương `Linear` bậc 2 nhưng có tính ổn định cao hơn (đặc biệt Ridge đang có RMSE tốt nhất).

## 4. Output
- Mô hình tiềm năng nhất hiện tại là: **4. Poly (Degree 2) + Ridge** với $R^2$ cao nhất là **0.8696**.
- Biểu đồ đánh giá lưu tại `Figures/Selection/model_comparison.png`.

## 5. Decision
- Loại bỏ mô hình Đa thức Bậc 3 vì Overfitting quá nặng.
- Lựa chọn mô hình **Polynomial Regression (Bậc 2) kết hợp với Ridge (hoặc Lasso)** làm Candidate xuất sắc nhất.
- Chuyển sang **Bước 7 — Hyperparameter Tuning** để tìm ra siêu tham số `alpha` tối ưu nhất cho Regularization của mô hình bậc 2 này.
