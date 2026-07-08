# Data Split Report (Phase 4)

## 1. Goal
Phân chia tập dữ liệu để huấn luyện, tinh chỉnh (tune) và đánh giá mô hình một cách công bằng nhất, đảm bảo không có Data Leakage.

## 2. Input
- Feature Set: `X_features.csv`
- Target: `y_target.csv`
- Kích thước ban đầu: 398 mẫu.

## 3. Tasks Performed & Visual Insights

### Tỉ lệ phân chia (Split Ratio)
Sử dụng hàm `train_test_split` của scikit-learn với `random_state=42` để đảm bảo khả năng tái lập (reproducibility):
- **Train Set (70%)**: 278 mẫu. Dùng để huấn luyện mô hình.
- **Validation Set (15%)**: 60 mẫu. Dùng để tinh chỉnh siêu tham số (Hyperparameter Tuning).
- **Test Set (15%)**: 60 mẫu. Được "cất kỹ", CHỈ dùng để đánh giá hiệu năng mô hình cuối cùng.

### Đánh giá độ đồng đều phân phối
Để trả lời câu hỏi: *"Việc phân chia ngẫu nhiên có làm hỏng cấu trúc dữ liệu không?"*, mình đã vẽ biểu đồ phân phối (Density Plot) cho cả 3 tập.

![Distribution Comparison](./Figures/Split/distribution_comparison.png)

**Insight**:
- Nhìn vào biểu đồ, ba đường cong đại diện cho Train (xanh dương), Validation (xanh lá) và Test (đỏ) có hình dáng **rất giống nhau**, đỉnh (peak) đều rơi vào khoảng 15-20 mpg và có phần đuôi dài (right-skewed) tương tự nhau.
- Điều này chứng tỏ phép chia ngẫu nhiên (random split) đã bảo toàn được cấu trúc phân phối gốc một cách hoàn hảo. Mô hình học từ tập Train sẽ không bị "sốc" khi dự đoán trên tập Validation hay Test.

## 4. Output
- Các tập dữ liệu đã được tách riêng rẽ và lưu tại `data/processed/split/`:
  - `X_train.csv`, `y_train.csv`
  - `X_val.csv`, `y_val.csv`
  - `X_test.csv`, `y_test.csv`

## 5. Decision
- Dữ liệu chia hoàn toàn đạt tiêu chuẩn công bằng và an toàn (không leakage).
- Bước 4 hoàn tất. Đã sẵn sàng tiến hành **Bước 5 — Baseline Model**.
