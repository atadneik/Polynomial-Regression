# TRAIN — Polynomial Regression (Gradient Descent) cho Auto MPG

Huấn luyện mô hình **hồi quy đa thức viết tay** bằng **Gradient Descent** để dự
đoán `mpg`, dựa trên các feature đã được EDA chọn lọc (xem `README_EDA.md`).

- **Script:** `train.py`
- **Dữ liệu:** `data/auto-mpg.data` (392 mẫu sau khi bỏ 6 dòng `horsepower` thiếu)
- **Output:** `Results/*.{txt,csv}` + `Figures/Train/*.png`
- **Chạy:** `python3 train.py`

---

## 1. Cấu hình

| Thành phần | Giá trị | Lý do |
|---|---|---|
| Feature | `weight`, `horsepower`, `displacement` | quan hệ cong mạnh với `mpg` (EDA) |
| Bậc đa thức | **2** | R² tăng mạnh ở bậc 2, chững ở bậc 3 |
| Thuật toán | Gradient Descent | tự cài, không dùng `sklearn` |
| Learning rate | `0.1` | hội tụ ổn định sau khi chuẩn hóa |
| Số vòng lặp | `5000` | đủ để cost phẳng |
| Tỉ lệ test | `20%` | seed `42`, cố định để tái lập |

---

## 2. Pipeline

```
load_data()        bỏ NaN horsepower, lấy 3 feature + target
   │
poly_features()    sinh đặc trưng đa thức bậc 2 (bình phương + tương tác)
   │               3 feature -> 9 cột: x1,x2,x3, x1²,x1x2,x1x3,x2²,x2x3,x3²
standardize()      z-score theo thống kê TRAIN (tránh rò rỉ dữ liệu)
   │
add_bias()         thêm cột 1 -> tổng 10 hệ số
   │
gradient_descent() tối ưu MSE, lưu cost history
   │
metrics()          R² / RMSE / MAE trên train & test
```

### Vì sao chuẩn hóa?
Đặc trưng đa thức có thang đo rất lệch (`weight²` ~ 10⁷ trong khi `horsepower`
~ 10²). Không chuẩn hóa thì Gradient Descent dao động/phân kỳ. Z-score đưa mọi
cột về mean 0, std 1 → learning rate chung dùng được cho tất cả.

> **Lưu ý chống rò rỉ:** `mean`/`std` tính **chỉ trên tập train**, rồi áp lại
> cho test — không dùng thống kê toàn bộ dữ liệu.

---

## 3. Công thức

**Giả thuyết:** `ŷ = Xw` (X gồm cột bias)

**Hàm mất mát (MSE/2):**

```
J(w) = (1/2n) · Σ (ŷᵢ − yᵢ)²
```

**Cập nhật trọng số:**

```
w := w − lr · (1/n) · Xᵀ(Xw − y)
```

---

## 4. Kết quả

| Mô hình | Hệ số | R² train | R² **test** | RMSE test | MAE test |
|---|---|---|---|---|---|
| Linear (bậc 1) | 4 | 0.704 | 0.698 | 4.37 | 3.20 |
| **Polynomial (bậc 2)** | 10 | 0.753 | **0.754** | **3.94** | **2.82** |

> **Bậc 2 nâng R² test 0.698 → 0.754 (+0.056)**, RMSE giảm từ 4.37 → 3.94 mpg.
> R² train ≈ R² test → mô hình **không overfit**, khớp với kết luận EDA.

---

## 5. File sinh ra

| File | Nội dung |
|---|---|
| `Results/weights.txt` | 10 trọng số (bias + 9 đặc trưng đa thức) |
| `Results/gd_cost_history.csv` | cost theo từng vòng lặp |
| `Results/train_metrics.txt` | log đầy đủ chỉ số train/test |
| `Figures/Train/cost_curve.png` | đường hội tụ (trục y log) |
| `Figures/Train/pred_vs_actual.png` | dự đoán vs thực tế trên tập test |

---

## 6. Tùy chỉnh

Sửa hằng số đầu `train.py`:

```python
FEATURES = ["weight", "horsepower", "displacement"]  # đổi/thêm feature
DEGREE   = 2        # bậc đa thức
LR       = 0.1      # learning rate
N_ITERS  = 5000     # số vòng lặp
TEST_RATIO = 0.2    # tỉ lệ test
SEED     = 42       # seed tái lập
```
