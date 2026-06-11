# Dự đoán mức tiêu thụ nhiên liệu xe hơi bằng Polynomial Regression

Dự án đi từ **phân tích dữ liệu (EDA)** để chọn đúng mô hình, rồi **tự cài
Polynomial Regression bằng Gradient Descent** để dự đoán `mpg` (miles per gallon)
của xe hơi — không dùng `scikit-learn`.

- **Dữ liệu:** Auto MPG (UCI) — `data/auto-mpg.data` (398 dòng × 9 cột)
- **Script:** `eda.py` (phân tích) · `train.py` (huấn luyện)
- **Chạy:** `python3 eda.py` rồi `python3 train.py`
- **Output:** `Figures/EDA/*.png`, `Figures/Train/*.png`, `Results/*.{txt,csv}`

> **Câu hỏi cốt lõi:** Quan hệ giữa thông số xe và `mpg` là **đường thẳng** hay
> **đường cong**? Nếu cong thì Linear Regression không đủ, phải dùng
> **Polynomial Regression**. Toàn bộ dự án xoay quanh việc trả lời câu hỏi này
> bằng số liệu, rồi xây mô hình phù hợp.

---

## 1. Bộ dữ liệu

Mỗi dòng là **một mẫu xe**; ta dùng các thông số kỹ thuật để dự đoán `mpg`
(càng cao càng tiết kiệm xăng).

| Thuộc tính | Kiểu | Ý nghĩa |
|---|---|---|
| `mpg` | liên tục | **Biến mục tiêu** — thứ cần dự đoán |
| `weight` | liên tục | Trọng lượng xe |
| `horsepower` | liên tục | Công suất động cơ (mã lực) |
| `displacement` | liên tục | Dung tích xi-lanh |
| `acceleration` | liên tục | Khả năng tăng tốc |
| `cylinders` | rời rạc | Số xi-lanh (4, 6, 8…) |
| `model_year` | rời rạc | Năm sản xuất |
| `origin` | rời rạc | Vùng: 1=USA, 2=Europe, 3=Japan |
| `car_name` | chuỗi | Tên xe — chỉ định danh, **không dùng** |

**Chất lượng:** chỉ cột `horsepower` thiếu **6 ô** (ghi `'?'` → `NaN`). Số lượng
nhỏ nên loại 6 dòng đó → còn **392 mẫu** cho phân tích định lượng và huấn luyện.

---

## 2. EDA — Chọn mô hình bằng bằng chứng

### 2.1 Tương quan với `mpg`

Hệ số tương quan `r ∈ [-1, 1]` đo mức độ và chiều của quan hệ tuyến tính.

| Feature | r với `mpg` | Đọc thế nào |
|---|---|---|
| `weight` | **−0.83** | càng nặng càng tốn xăng — liên hệ mạnh |
| `displacement` | **−0.80** | dung tích lớn → tốn xăng |
| `horsepower` | **−0.78** | mã lực cao → tốn xăng |
| `acceleration` | +0.42 | liên hệ **yếu** → loại |

> **⚠️ Đa cộng tuyến:** bốn biến kích cỡ động cơ (`cylinders`, `displacement`,
> `horsepower`, `weight`) tương quan với nhau tới **|r| = 0.93** — "nói cùng một
> câu chuyện". Dùng cả nhóm gây dư thừa và làm hệ số kém ổn định → chỉ giữ
> **3 biến đại diện**: `weight`, `horsepower`, `displacement`.
>
> *Hình: `step3_heatmap.png`*

### 2.2 Bằng chứng "đường cong" — độ cong còn sót (cốt lõi)

1. Khớp **đường thẳng** cho `mpg` theo từng feature.
2. Tính **residual** = thật − dự đoán đường thẳng.
3. Nếu đường thẳng đúng, residual rải ngẫu nhiên quanh 0; nếu residual **uốn
   hình chữ U** → quan hệ thật là **đường cong** mà đường thẳng bỏ sót.
4. "Độ cong còn sót" = % phương sai residual mà một đường cong bậc 2 giải thích
   được. **Càng cao → càng cần đa thức.**

| Feature | Độ cong còn sót | Ý nghĩa |
|---|---|---|
| `horsepower` | **20.7%** | rất cong — đường thẳng bỏ sót nhiều |
| `displacement` | **11.5%** | cong rõ |
| `weight` | 7.3% | cong vừa |
| `acceleration` | 1.8% | gần như thẳng |

> *Hình: `step4_residuals_linear.png`* — đường đỏ uốn rõ chữ U ở
> `horsepower`/`displacement`, gần như nằm ngang ở `acceleration`.

### 2.3 R² tăng khi nâng bậc đa thức

R² = % biến thiên `mpg` mà mô hình giải thích được (0 = vô dụng, 1 = hoàn hảo).

| Feature | bậc 1 | bậc 2 | bậc 3 | Nhận xét |
|---|---|---|---|---|
| `weight` | 0.693 | **0.715** | 0.715 | tăng ở bậc 2, bậc 3 đứng yên |
| `displacement` | 0.648 | **0.689** | 0.690 | tăng ở bậc 2, bậc 3 đứng yên |
| `horsepower` | 0.606 | **0.688** | 0.688 | tăng mạnh ở bậc 2 |
| `acceleration` | 0.179 | 0.194 | 0.196 | thấp ở mọi bậc → biến yếu |

R² **nhảy lên ở bậc 2** nhưng **chững ở bậc 3** (cong hơn nữa chỉ tổ overfit) →
dấu hiệu kinh điển nói **bậc 2 là điểm dừng tối ưu**.

### 2.4 Kết luận EDA

1. **Quan hệ cong, không thẳng** — residual đường thẳng uốn chữ U.
2. **Bậc 2 là tối ưu** — R² chững ở bậc 3.
3. **Loại `acceleration`** (yếu) và tránh dùng cả nhóm động cơ (đa cộng tuyến).

→ **Auto MPG phù hợp để minh họa Polynomial Regression**, với 3 feature tốt
nhất: `weight`, `horsepower`, `displacement`.

---

## 3. TRAIN — Huấn luyện mô hình

### 3.1 Cấu hình

| Thành phần | Giá trị | Lý do |
|---|---|---|
| Feature | `weight`, `horsepower`, `displacement` | quan hệ cong mạnh (EDA) |
| Bậc đa thức | **2** | R² tăng mạnh ở bậc 2, chững ở bậc 3 |
| Thuật toán | Gradient Descent | tự cài, không dùng `sklearn` |
| Learning rate | `0.1` | hội tụ ổn định sau chuẩn hóa |
| Số vòng lặp | `5000` | đủ để cost phẳng |
| Tỉ lệ test | `20%` | seed `42`, cố định để tái lập |

### 3.2 Pipeline

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

### 3.3 Công thức

**Giả thuyết:** `ŷ = Xw` (X gồm cột bias)

**Hàm mất mát (MSE/2):**

```
J(w) = (1/2n) · Σ (ŷᵢ − yᵢ)²
```

**Cập nhật trọng số:**

```
w := w − lr · (1/n) · Xᵀ(Xw − y)
```

### 3.4 Vì sao phải chuẩn hóa?

Đặc trưng đa thức lệch thang đo nặng (`weight²` ~ 10⁷ trong khi `horsepower`
~ 10²). Không chuẩn hóa thì Gradient Descent dao động/phân kỳ. Z-score đưa mọi
cột về mean 0, std 1 → dùng chung một learning rate.

> **Chống rò rỉ:** `mean`/`std` tính **chỉ trên tập train** rồi áp lại cho test —
> không dùng thống kê toàn bộ dữ liệu.

---

## 4. Kết quả

| Mô hình | Hệ số | R² train | R² **test** | RMSE test | MAE test |
|---|---|---|---|---|---|
| Linear (bậc 1) | 4 | 0.704 | 0.698 | 4.37 | 3.20 |
| **Polynomial (bậc 2)** | 10 | 0.753 | **0.754** | **3.94** | **2.82** |

> **Bậc 2 nâng R² test 0.698 → 0.754 (+0.056)**, RMSE giảm 4.37 → 3.94 mpg.
> R² train ≈ R² test → mô hình **không overfit**, khớp kết luận EDA.

*Hình: `Figures/Train/cost_curve.png` (hội tụ), `pred_vs_actual.png` (dự đoán vs thực tế).*

---

## 5. File sinh ra

| File | Nội dung |
|---|---|
| `Figures/EDA/step1_outliers_boxplot.png` | phân bố & ngoại lai 4 biến liên tục |
| `Figures/EDA/step2_distributions.png` | histogram `mpg` và các feature |
| `Figures/EDA/step2_discrete_count.png` | đếm theo `cylinders`/`model_year`/`origin` |
| `Figures/EDA/step3_feature_vs_target.png` | scatter + so sánh thẳng vs cong bậc 2 |
| `Figures/EDA/step3_heatmap.png` | ma trận tương quan (cảnh báo đa cộng tuyến) |
| `Figures/EDA/step4_residuals_linear.png` | residual đường thẳng uốn chữ U |
| `Results/eda_summary.txt` | tóm tắt số liệu EDA |
| `Results/weights.txt` | 10 trọng số (bias + 9 đặc trưng) |
| `Results/gd_cost_history.csv` | cost theo từng vòng lặp |
| `Results/train_metrics.txt` | log đầy đủ chỉ số train/test |
| `Figures/Train/cost_curve.png` | đường hội tụ (trục y log) |
| `Figures/Train/pred_vs_actual.png` | dự đoán vs thực tế trên tập test |

---

## 6. Tùy chỉnh

Sửa hằng số đầu `train.py`:

```python
FEATURES   = ["weight", "horsepower", "displacement"]  # đổi/thêm feature
DEGREE     = 2        # bậc đa thức
LR         = 0.1      # learning rate
N_ITERS    = 5000     # số vòng lặp
TEST_RATIO = 0.2      # tỉ lệ test
SEED       = 42       # seed tái lập
```

---

## 7. Hướng phát triển

- Thêm biến `model_year` / `origin`.
- Thử **regularization (Ridge)** để chống overfit khi tăng bậc.
- **Cross-validation** để ước lượng độ ổn định.
- So sánh với `scikit-learn` để kiểm chứng phần tự cài.

---

> Tài liệu chi tiết: [`README_EDA.md`](README_EDA.md) (phân tích) ·
> [`README_TRAIN.md`](README_TRAIN.md) (huấn luyện) ·
> [`THUYET_TRINH.md`](THUYET_TRINH.md) (kịch bản thuyết trình).
