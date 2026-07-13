# Hyperparameter Tuning Report (Phase 7)

## 1. Goal
Tìm ra tham số tối ưu nhất cho mô hình đã chọn (`Polynomial Bậc 2 + Ridge Regression`) để cân bằng hoàn hảo giữa Overfitting và Underfitting.

## 2. Input
- Mô hình: Pipeline (`Poly degree=2 -> StandardScaler -> Ridge(alpha)`)
- Hyperparameter cần tune: `alpha` (hệ số phạt L2 của Ridge).
- Không gian tìm kiếm (Search Space): 100 giá trị `alpha` trải đều trên thang đo logarit từ $10^{-3}$ đến $10^4$.
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
  - **Best Alpha:** `0.2984`
- **Hiệu suất đạt được tại điểm tối ưu:**
  - Val R²: `0.8720`
  - Val RMSE: `2.5065`
  - Train RMSE: `2.6280`
- Bộ tham số đã được lưu tại: `Results/best_params.json`
- Biểu đồ lưu tại: `Figures/Tuning/validation_curve.png`

## 5. Decision
- Quá trình Tuning đã thành công mĩ mãn, tìm ra chính xác điểm "Sweet Spot".
- Chốt cấu trúc mô hình cuối cùng: **Polynomial Bậc 2 + Ridge(alpha=0.2984)**.
- Tiến hành **Bước 8 — Train Model**: Sử dụng cấu trúc này để huấn luyện lại mô hình một lần cuối cùng trên tập Train (hoặc Train+Val), chuẩn bị đem ra "chiến trường" đánh giá trên tập Test độc lập.
