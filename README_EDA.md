# EDA — Phân tích khám phá dữ liệu Auto MPG

## EDA là gì và để làm gì?

**EDA (Exploratory Data Analysis)** là bước "làm quen" với dữ liệu *trước khi*
xây mô hình: xem dữ liệu có gì, sạch hay bẩn, các biến quan hệ với nhau ra sao.
Mục tiêu cụ thể của tài liệu này là trả lời **một câu hỏi**:

> Quan hệ giữa các đặc trưng của xe và mức tiêu thụ nhiên liệu `mpg` là
> **đường thẳng** hay **đường cong**?

Nếu là đường cong → hồi quy tuyến tính (Linear Regression) sẽ thiếu, và
**Polynomial Regression (hồi quy đa thức)** mới phù hợp. Toàn bộ EDA dưới đây
được thiết kế để chứng minh điều đó bằng số liệu và hình ảnh.

- **Script:** `eda.py`
- **Dữ liệu:** `data/auto-mpg.data` (UCI Auto MPG, 398 dòng × 9 cột)
- **Output:** `Figures/EDA/*.png` + `Results/eda_summary.txt`
- **Chạy:** `python3 eda.py`

---

## 1. Bộ dữ liệu có gì?

Mỗi dòng là **một mẫu xe**; ta muốn dùng các thông số kỹ thuật để **dự đoán
`mpg`** (miles per gallon — xe đi được bao nhiêu dặm với 1 gallon xăng; càng cao
càng tiết kiệm).

| Thuộc tính | Kiểu | Ý nghĩa |
|---|---|---|
| `mpg` | liên tục | **Biến mục tiêu (target)** — thứ cần dự đoán |
| `weight` | liên tục | Trọng lượng xe |
| `horsepower` | liên tục | Công suất động cơ (mã lực) |
| `displacement` | liên tục | Dung tích xi-lanh |
| `acceleration` | liên tục | Khả năng tăng tốc |
| `cylinders` | rời rạc | Số xi-lanh (4, 6, 8…) |
| `model_year` | rời rạc | Năm sản xuất |
| `origin` | rời rạc | Vùng: 1=USA, 2=Europe, 3=Japan |
| `car_name` | chuỗi | Tên xe — chỉ để định danh, **không dùng** |

**Chất lượng dữ liệu:**
- Tổng cộng **398 dòng × 9 cột**.
- Chỉ cột `horsepower` bị **thiếu 6 ô** (ghi `'?'`, được đọc thành `NaN`). Số
  lượng nhỏ nên ta loại 6 dòng đó khi cần phân tích định lượng.

---

## 2. Quy trình 4 bước (mỗi bước trả lời 1 câu hỏi)

| Bước | Câu hỏi cần trả lời | Hình minh họa |
|---|---|---|
| **1. Tổng quan & chất lượng** | Dữ liệu sạch không? Có giá trị bất thường? | `step1_outliers_boxplot.png` |
| **2. Phân phối biến** | Mỗi biến trải ra như thế nào? | `step2_distributions.png`, `step2_discrete_count.png` |
| **3. Quan hệ với target** | Biến nào ảnh hưởng `mpg`? Thẳng hay cong? | `step3_feature_vs_target.png`, `step3_heatmap.png` |
| **4. Bằng chứng phi tuyến** | Đường thẳng có "bỏ sót" phần cong không? | `step4_residuals_linear.png` |

Dưới đây giải thích chi tiết từng bước và **cách đọc** kết quả.

---

## 3. Giải thích từng kết quả

### 3.1 — Tương quan với `mpg` (hệ số `r`)

Hệ số tương quan `r` đo **mức độ và chiều của quan hệ tuyến tính** giữa hai biến,
nằm trong khoảng `[-1, 1]`:
- `r` gần **−1**: một biến tăng thì biến kia **giảm** mạnh.
- `r` gần **+1**: cùng tăng cùng giảm mạnh.
- `r` gần **0**: gần như không liên quan (theo kiểu đường thẳng).

| Feature | r với `mpg` | Đọc thế nào |
|---|---|---|
| `weight` | **−0.83** | Xe càng **nặng** càng **tốn xăng** (mpg thấp) — liên hệ mạnh |
| `displacement` | **−0.80** | Dung tích càng lớn càng tốn xăng |
| `horsepower` | **−0.78** | Càng nhiều mã lực càng tốn xăng |
| `acceleration` | +0.42 | Liên hệ **yếu** |

→ Ba biến `weight`, `horsepower`, `displacement` là **ứng viên feature tốt
nhất**; `acceleration` yếu nên loại.

> **⚠️ Cảnh báo đa cộng tuyến (multicollinearity):** Bốn biến liên quan tới kích
> cỡ động cơ (`cylinders`, `displacement`, `horsepower`, `weight`) **tương quan
> với nhau rất cao** (|r| lên tới **0.93**). Nghĩa là chúng "nói cùng một câu
> chuyện". Dùng cả bốn cùng lúc gây dư thừa và làm hệ số mô hình kém ổn định →
> nên chọn **một vài biến đại diện** thay vì tất cả.

### 3.2 — "Độ cong còn sót" (cốt lõi của EDA này)

Đây là phép đo quan trọng nhất. Cách làm:
1. Khớp một **đường thẳng** (hồi quy bậc 1) cho `mpg` theo từng feature.
2. Tính **residual** = phần sai lệch `giá trị thật − giá trị đường thẳng dự đoán`.
3. Nếu đường thẳng đã "đúng", residual phải **rải ngẫu nhiên quanh 0**. Nhưng nếu
   residual **uốn thành hình chữ U** (âm ở hai đầu, dương ở giữa hoặc ngược lại)
   → quan hệ thật là **đường cong** mà đường thẳng đã bỏ sót.
4. "Độ cong còn sót" = phần trăm phương sai của residual mà **một đường cong bậc
   2** giải thích được. **Càng cao → quan hệ càng cong → càng cần đa thức.**

| Feature | Độ cong còn sót | Ý nghĩa |
|---|---|---|
| `horsepower` | **20.7%** | Rất cong — đường thẳng bỏ sót nhiều |
| `displacement` | **11.5%** | Cong rõ |
| `weight` | 7.3% | Cong vừa |
| `acceleration` | 1.8% | Gần như thẳng |

→ Xem hình `step4_residuals_linear.png`: đường đỏ (xu hướng residual) **uốn cong
rõ** ở `horsepower`/`displacement`, gần như nằm ngang ở `acceleration`.

### 3.3 — R² tăng bao nhiêu khi nâng bậc đa thức?

**R² (R bình phương)** cho biết mô hình giải thích được **bao nhiêu % biến thiên**
của `mpg` (0 = vô dụng, 1 = hoàn hảo). Ta khớp lần lượt đa thức **bậc 1 → 2 → 3**
cho từng feature và xem R² thay đổi:

| Feature | bậc 1 | bậc 2 | bậc 3 | Nhận xét |
|---|---|---|---|---|
| `weight` | 0.693 | **0.715** | 0.715 | Tăng ở bậc 2, bậc 3 đứng yên |
| `displacement` | 0.648 | **0.689** | 0.690 | Tăng ở bậc 2, bậc 3 đứng yên |
| `horsepower` | 0.606 | **0.688** | 0.688 | Tăng mạnh ở bậc 2 |
| `acceleration` | 0.179 | 0.194 | 0.196 | Thấp ở mọi bậc → biến yếu |

**Cách đọc:** R² **nhảy lên ở bậc 2** (đường cong khớp tốt hơn đường thẳng) nhưng
**không tăng thêm ở bậc 3** (cong hơn nữa cũng vô ích, chỉ tổ overfit). Đây chính
là dấu hiệu kinh điển nói rằng **bậc 2 là điểm dừng tối ưu**.

---

## 4. Kết luận

1. **Quan hệ cong, không thẳng:** `mpg ~ {weight, horsepower, displacement}` cong
   rõ rệt — residual của đường thẳng uốn thành **chữ U** (mục 3.2).
2. **Bậc 2 là tối ưu:** R² **tăng mạnh từ bậc 1 lên bậc 2**, rồi **chững ở bậc 3**
   (mục 3.3) → chọn **Polynomial bậc 2** là đủ cong mà không overfit.
3. **Loại biến yếu:** `acceleration` tương quan yếu và gần như thẳng → **không
   dùng**. Tránh dùng cả nhóm động cơ cùng lúc vì **đa cộng tuyến**.

> **⇒ Bộ dữ liệu Auto MPG rất phù hợp để minh họa Polynomial Regression**, với
> các feature tốt nhất là `weight`, `horsepower`, `displacement`. Kết luận này
> được dùng trực tiếp ở bước huấn luyện — xem `README_TRAIN.md`.

---

## 5. Bảng tra hình ảnh

| File | Cho thấy điều gì |
|---|---|
| `step1_outliers_boxplot.png` | Phân bố và giá trị ngoại lai của 4 biến liên tục |
| `step2_distributions.png` | Histogram của `mpg` và các feature (xem độ lệch) |
| `step2_discrete_count.png` | Số lượng theo `cylinders`, `model_year`, `origin` |
| `step3_feature_vs_target.png` | Scatter + so sánh **đường thẳng vs đường cong bậc 2** |
| `step3_heatmap.png` | Ma trận tương quan (gồm cảnh báo đa cộng tuyến) |
| `step4_residuals_linear.png` | Residual đường thẳng uốn **chữ U** → cần đa thức |
