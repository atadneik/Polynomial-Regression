# Polynomial Regression — CCPP (from scratch)

Project minh hoạ **Polynomial Regression viết từ đầu bằng NumPy**, áp dụng lên
dataset *Combined Cycle Power Plant* (CCPP) của UCI. Toàn bộ phần lõi (xây ma
trận đặc trưng, Normal Equation, Gradient Descent, hàm mất mát, scaler) đều
tự cài, không dùng `scikit-learn`. Pandas/Matplotlib/Seaborn chỉ dùng cho
load dữ liệu và trực quan hoá.

---

## 1. Mục tiêu

Dự đoán **PE — Công suất điện đầu ra (MW)** của một nhà máy nhiệt điện chu
trình hỗn hợp từ **AT — Nhiệt độ môi trường (°C)**.

- Quan hệ vật lý: nhiệt độ không khí càng cao thì turbine khí càng kém hiệu
  quả → công suất giảm. Đường cong khá mượt, ít nhiễu → dataset rất phù hợp
  để **minh hoạ Polynomial Regression** và bias-variance tradeoff.
- Bài toán thuộc loại **học có giám sát — hồi quy** (target là số thực).

---

## 2. Dataset — Combined Cycle Power Plant (CCPP)

### 2.1. Bối cảnh — Nhà máy điện chu trình hỗn hợp là gì?

**Combined Cycle Power Plant (CCPP)** — nhà máy nhiệt điện chu trình hỗn
hợp — là loại nhà máy phát điện sử dụng **hai chu trình nhiệt động kết hợp**
để tận dụng tối đa năng lượng từ nhiên liệu:

1. **Chu trình Brayton (turbine khí — Gas Turbine, GT):** đốt khí tự nhiên,
   khí cháy nóng làm quay turbine khí → tạo điện. Khí thải sau turbine vẫn
   còn rất nóng (~500–600°C).
2. **Chu trình Rankine (turbine hơi — Steam Turbine, ST):** dùng khí thải
   nóng đó để đun sôi nước → hơi nước cao áp làm quay turbine hơi → tạo
   thêm điện.

Nhờ tận dụng nhiệt 2 lần, hiệu suất CCPP đạt **55–60%** (so với chỉ ~33%
của nhà máy nhiệt điện thường). Tổng công suất điện = công suất GT + ST.

**Dataset này** ghi lại số liệu vận hành **trong 6 năm (2006–2011)** của
**một nhà máy CCPP thật**, khi nhà máy chạy **full-load** (tải tối đa).
Mục tiêu: dự đoán **tổng công suất điện đầu ra theo giờ** dựa trên 4 điều
kiện môi trường và vận hành.

### 2.2. Nguồn gốc & thông tin chính

| Thuộc tính | Giá trị |
|------------|---------|
| **Nguồn** | UCI Machine Learning Repository |
| **Tên gốc** | *Combined Cycle Power Plant Data Set* |
| **Người đóng góp** | Pınar Tüfekci & Heysem Kaya (Đại học Namık Kemal, Thổ Nhĩ Kỳ) |
| **Bài báo gốc** | Tüfekci, P. (2014), *Int. Journal of Electrical Power & Energy Systems*, vol. 60, pp. 126–140 |
| **Số mẫu** | 9568 (mỗi mẫu = 1 quan trắc theo giờ) |
| **Số cột** | 5 (4 đặc trưng + 1 target) |
| **Thiếu / NaN** | Không có |
| **Khoảng thời gian** | 6 năm (2006–2011) |
| **Loại bài toán** | Hồi quy (regression) |

### 2.3. Mô tả từng cột

| Cột | Ý nghĩa vật lý | Đơn vị | Khoảng giá trị | Vai trò trong CCPP |
|-----|----------------|--------|----------------|---------------------|
| **AT** | Ambient Temperature — nhiệt độ không khí môi trường | °C | 1.81 → 37.11 | Không khí nạp vào turbine khí. Càng nóng → không khí loãng → khối lượng nạp giảm → công suất GT giảm. |
| **V**  | Exhaust Vacuum — chân không ở bình ngưng | cm Hg | 25.36 → 81.56 | Đo độ chân không của bình ngưng (condenser) ở cuối chu trình hơi. V càng cao = chân không càng sâu = chênh áp qua turbine hơi càng lớn = ST sinh thêm điện. |
| **AP** | Ambient Pressure — áp suất khí quyển | mbar | 992.89 → 1033.30 | Áp suất môi trường cao → mật độ không khí nạp cao → GT chạy mạnh hơn. |
| **RH** | Relative Humidity — độ ẩm tương đối | % | 25.56 → 100.16 | Hơi nước trong không khí ảnh hưởng quá trình đốt và làm mát. Cao thì thường có lợi nhẹ. |
| **PE** | **Power Output — công suất điện đầu ra (target)** | MW | 420.26 → 495.76 | **Cần dự đoán.** Tổng công suất nhà máy theo giờ. |

> **Lưu ý quan trọng:** dataset chỉ ghi nhận khi nhà máy hoạt động **full
> load**, tức không có các giai đoạn khởi động / dừng / tải thấp. Vì thế
> quan hệ giữa biến môi trường và PE rất sạch, ít nhiễu — đó là lý do bài
> hồi quy đạt R² ≈ 0.92–0.95 dễ dàng.

### 2.4. Phân bố thực tế của dữ liệu

Từ `Results/eda_summary.txt`:

```
                AT         V          AP         RH         PE
count       9568.0    9568.0      9568.0     9568.0     9568.0
mean        19.65     54.31     1013.26      73.31     454.37
std          7.45     12.71        5.94      14.60      17.07
min          1.81     25.36      992.89      25.56     420.26
25%         13.51     41.74     1009.10      63.33     439.75
50%         20.35     52.08     1012.94      74.98     451.55
75%         25.72     66.54     1017.26      84.83     468.43
max         37.11     81.56     1033.30     100.16     495.76
```

- **AT (°C):** trải đều từ ~2 → 37°C — bao trùm cả mùa đông lẫn mùa hè ở
  Thổ Nhĩ Kỳ. Phân bố hai đỉnh nhẹ (theo mùa).
- **V (cm Hg):** phân bố **bimodal rõ** — có 2 cụm tương ứng 2 chế độ vận
  hành / 2 mùa khác nhau. Xem `Figures/EDA/histograms.png`.
- **AP (mbar):** gần phân bố chuẩn quanh 1013 mbar (áp suất khí quyển tiêu
  chuẩn).
- **RH (%):** lệch trái — độ ẩm thường cao (median 75%), hiếm khi thấp dưới 30%.
- **PE (MW):** **bimodal** giống V — 2 chế độ vận hành rõ ràng.

### 2.5. Mối quan hệ giữa các biến (từ EDA)

Hệ số tương quan Pearson với PE (|r| giảm dần):

```
AT : r = −0.9482   ← rất mạnh, tuyến tính âm  (1 biến này đã ~90% câu chuyện)
 V : r = −0.8698   ← mạnh, tuyến tính âm
AP : r = +0.5184   ← trung bình, tuyến tính dương
RH : r = +0.3898   ← yếu, tuyến tính dương
```

**Đa cộng tuyến:** AT và V có tương quan **r = +0.84** với nhau — về cơ bản
khi nhiệt độ tăng thì điều kiện vận hành chân không cũng thay đổi đồng bộ
(2 biến không độc lập). Nếu dùng cả 4 đặc trưng cho mô hình tuyến tính
thường, sẽ có hiện tượng đa cộng tuyến → trọng số không ổn định. Đây là lý
do trong các bài đầy đủ người ta dùng Ridge/Lasso.

### 2.6. Vì sao project này chỉ dùng **AT**?

- AT có r = **−0.95** với PE → một mình đã giải thích ~90% phương sai.
- Quan hệ AT→PE **mượt, gần tuyến tính, ít nhiễu** → là điều kiện vàng để
  minh hoạ **Polynomial Regression** và **bias-variance tradeoff** mà không
  bị che bởi các biến phụ.
- Bài toán 1 chiều giúp **vẽ đồ thị 2D trực quan**: trục x = AT, trục y =
  PE, đường khớp đa thức đè lên scatter — học sinh nhìn ra ngay bậc nào
  underfit, bậc nào vừa, bậc nào bị Runge.
- Tránh phải xử lý đa cộng tuyến (AT–V) — sẽ làm bài học rối, lệch chủ đề.

Project này tập trung vào **thuật toán Polynomial Regression từ đầu**,
không phải tối ưu R² cao nhất có thể. Nếu mục tiêu là production, người ta
sẽ dùng cả 4 đặc trưng + Ridge → R² ≈ 0.95+.

### 2.7. File trong repo

| Đường dẫn | Nội dung |
|-----------|----------|
| `data/CCPP/Folds5x2_pp.xlsx` | File gốc tải từ UCI (định dạng Excel). |
| `data/CCPP/Readme.txt` | Readme đi kèm từ UCI — mô tả nguồn gốc, các cột. |
| `data/ccpp.csv` | Phiên bản CSV đã chuyển đổi để code Python đọc nhanh bằng `pandas.read_csv`. **Đây là file thực tế được dùng trong `load_data()`** của cả `eda_ccpp.py` và `train_ccpp.py`. |

---

## 3. Cấu trúc thư mục

```
Polynomial Regression/
├── data/
│   ├── ccpp.csv                     # dataset đã chuyển CSV (dùng trong code)
│   └── CCPP/
│       ├── Folds5x2_pp.xlsx         # file gốc từ UCI
│       └── Readme.txt
├── polynomial_regression.py         # CORE — Polynomial Regression from scratch
├── eda_ccpp.py                      # script EDA — sinh figures + summary
├── train_ccpp.py                    # script train — sinh metric + figures
├── Figures/
│   ├── EDA/                         # ảnh EDA (5 file)
│   │   ├── histograms.png
│   │   ├── scatter_vs_PE.png
│   │   ├── corr_heatmap.png
│   │   ├── boxplots.png
│   │   └── pairplot.png
│   └── Train/                       # ảnh khi train (2 file)
│       ├── polyfit.png
│       └── cost_curve.png
└── Results/                         # kết quả dạng text/CSV
    ├── eda_summary.txt
    ├── ccpp_metrics.txt
    ├── ccpp_metrics.csv
    ├── ccpp_gd_cost_history.csv
    └── ccpp_weights.txt
```

---

## 4. Cách chạy

Yêu cầu: Python 3.10+, các package: `numpy`, `pandas`, `matplotlib`,
`seaborn`, `openpyxl` (chỉ cần nếu muốn đọc lại file xlsx gốc).

```bash
# (1) EDA — phân tích dữ liệu trước khi train
python3 eda_ccpp.py

# (2) Train — chạy polynomial regression, lưu metric + figures
python3 train_ccpp.py
```

Mỗi script chạy độc lập, ghi đè kết quả vào `Figures/` và `Results/`.

---

## 5. Giải thích chi tiết từng file code

### 5.1. `polynomial_regression.py` — CORE (≈ 80 dòng)

Đây là phần lõi, **không phụ thuộc dataset nào**. Triết lý cốt lõi:
> *Polynomial regression = linear regression trên đặc trưng đã được nâng luỹ
> thừa.* Phi tuyến theo x, **tuyến tính theo trọng số w** → vẫn dùng được
> Normal Equation / Gradient Descent y hệt linear regression.

| Hàm / Class | Vai trò |
|-------------|--------|
| `build_design_matrix(x, degree)` | Biến vector x (m,) thành ma trận Vandermonde `[1, x, x², …, x^d]` kích thước (m, d+1). Cột đầu = 1 ứng với bias w₀. |
| `normal_equation(X, y)` | Nghiệm đóng `w = (XᵀX)⁻¹ Xᵀy`, dùng `np.linalg.pinv` để ổn định khi `XᵀX` gần suy biến (chuyện rất hay xảy ra khi degree cao). |
| `predict(X, w)` | Đơn giản `X @ w`. |
| `mse(y_true, y_pred)` | Mean squared error. |
| `r2_score(y_true, y_pred)` | `1 − SS_res / SS_tot` — phần trăm phương sai mô hình giải thích được. |
| `gradient_descent(X, y, lr, n_iters)` | Tự cài: cập nhật `w ← w − lr · (2/m) · Xᵀ(Xw − y)`. Trả về `w` cuối + `cost_history` để vẽ đồ thị hội tụ. |
| `class StandardScaler1D` | Chuẩn hoá z = (x − mean) / std. **Tách rời `fit` (chỉ học trên train) và `transform` (áp lên val/test)** để tránh data leakage. |

> ⚠️ **Vì sao GD cần thêm scaling cột?** Sau khi x đã được StandardScaler đưa
> về z, các cột `x², x³, …, xⁿ` vẫn có thang đo rất khác nhau (x⁸ lớn hơn x
> hàng triệu lần). Nếu chạy GD thẳng trên đó, gradient sẽ rất lệch và phân
> kỳ (NaN). Vì vậy trong `train_ccpp.py:compare_methods` mình chuẩn hoá tiếp
> từng cột của ma trận đặc trưng trước khi đưa vào GD.

---

### 5.2. `eda_ccpp.py` — EDA (≈ 130 dòng)

Script phân tích dữ liệu **trước khi train**, dùng seaborn cho trực quan đẹp
hơn. Mỗi hàm sinh 1 figure:

| Hàm | Figure | Mục đích |
|-----|--------|---------|
| `plot_histograms` | `EDA/histograms.png` | Histogram + KDE cho 5 biến → thấy lệch, đa đỉnh, outlier. |
| `plot_scatter_vs_target` | `EDA/scatter_vs_PE.png` | Scatter từng feature vs PE, có line hồi quy tuyến tính → biết quan hệ tuyến tính/cong, ghi r ngay tiêu đề. |
| `plot_corr_heatmap` | `EDA/corr_heatmap.png` | Heatmap tương quan Pearson 5×5, annotated → phát hiện đa cộng tuyến. |
| `plot_boxplots` | `EDA/boxplots.png` | Boxplot **chuẩn hoá z-score** (5 biến chung 1 trục) → so outlier ngay trên cùng thang đo. |
| `plot_pairplot` | `EDA/pairplot.png` | Pairplot sample 2000 mẫu — toàn cảnh quan hệ từng cặp + KDE đường chéo. |
| `save_summary` | `Results/eda_summary.txt` | describe(), correlation với PE (sắp xếp \|r\| giảm dần), ma trận tương quan đầy đủ. |

---

### 5.3. `train_ccpp.py` — TRAIN (≈ 250 dòng)

Pipeline đầy đủ: load → split → scale → so sánh các bậc → chọn bậc → báo cáo
test → vẽ đồ thị → đối chiếu NE/GD.

#### Luồng `main()`

```
1. load_data()                           → x = AT, y = PE
2. train_val_test_split(x, y)            → 60% train / 20% val / 20% test
3. StandardScaler1D().fit(x_tr).transform(...) cho cả train/val/test
4. Vòng lặp các degree ∈ {1,2,3,4,5,8,12}:
      evaluate(degree, train, train)     → metric trên train
      evaluate(degree, train, val)       → metric trên val
      lưu (w, m_tr, m_val) vào dict results
   In bảng so sánh dùng VALIDATION R².
5. best_degree = argmax R²_val           ← chọn siêu tham số bằng VAL
6. evaluate(best_degree, train, test)    ← chấm 1 LẦN trên test (giữ kín)
   In bảng kết quả cuối cho train/val/test.
7. save_results(...)                     → Results/ccpp_metrics.{txt,csv}
8. plot_fits(...)                        → Figures/Train/polyfit.png
9. compare_methods(...)                  → NE vs GD, lưu cost history + w
```

#### Vì sao chia **train/val/test** (60/20/20)?

Lúc đầu code chỉ có train/test 80/20 và **chọn bậc tốt nhất dựa trên test**.
Đó là **data leakage**: tập test bị dùng để ra quyết định mô hình thì không
còn "chưa từng thấy" nữa, metric báo cáo sẽ lạc quan.

Đúng quy trình:
- **train** — học trọng số w cho mỗi bậc.
- **val** — chọn bậc tốt nhất.
- **test** — **chỉ dùng đúng 1 lần ở cuối** để ước lượng performance thực tế.

#### Hàm `compare_methods` — kiểm chứng lý thuyết

So sánh Normal Equation (nghiệm đóng) vs Gradient Descent (lặp). Vì hàm mất
mát MSE của linear regression là **parabol lồi có duy nhất một cực tiểu
toàn cục**, hai phương pháp phải hội tụ về cùng MSE → nếu khớp ⇒ implement
đúng. Code lưu:
- `Results/ccpp_gd_cost_history.csv` — cost qua từng vòng lặp (8000 dòng).
- `Results/ccpp_weights.txt` — w_NE và w_GD + MSE đối chiếu.
- `Figures/Train/cost_curve.png` — đường hội tụ log–log.

---

## 6. Kết quả EDA

`Results/eda_summary.txt` (tóm tắt):

```
Tương quan Pearson với PE (|r| giảm dần):
  AT : r = −0.9482   ← rất mạnh, tuyến tính âm
   V : r = −0.8698   ← mạnh
  AP : r = +0.5184   ← trung bình
  RH : r = +0.3898   ← yếu

Ma trận tương quan đầy đủ — chú ý cặp AT–V: r = +0.84 (đa cộng tuyến cao).
```

**Quan sát chính từ figures:**
- `histograms.png` — V và PE có phân bố **2 đỉnh** (bimodal) → nhà máy hoạt
  động ở 2 chế độ vận hành khác nhau. AT và AP gần phân bố chuẩn.
- `scatter_vs_PE.png` — AT vs PE **gần như tuyến tính âm**, V vs PE cũng
  tuyến tính nhưng có "vệt" rõ ràng do bimodal; AP, RH chỉ tương quan yếu.
- `corr_heatmap.png` — cặp **AT ↔ V có r = 0.84** → nếu sau này dùng mô
  hình đa biến phải lưu ý đa cộng tuyến (có thể bỏ V hoặc dùng
  Ridge/Lasso).
- `boxplots.png` — AT, V, PE không có outlier; AP, RH có vài outlier nhưng
  hợp lý về mặt vật lý (áp suất / độ ẩm cực trị).
- `pairplot.png` — bức tranh toàn cảnh; nhìn được mọi cặp quan hệ trong
  cùng một hình.

→ **Kết luận EDA:** dùng AT làm feature đơn cho Polynomial Regression là
hợp lý — đã cover được phần lớn signal, quan hệ đủ mượt để minh hoạ bài học.

---

## 7. Kết quả Train

### 7.1. So sánh các bậc trên VALIDATION

`Results/ccpp_metrics.txt`:

| degree | R² train | R² val | RMSE val | MSE val |
|-------:|---------:|-------:|---------:|--------:|
|  1     |  0.8985  | 0.8939 | 5.578    | 31.111  |
|  2     |  0.9063  | 0.9024 | 5.349    | 28.616  |
|  3     |  0.9110  | 0.9076 | 5.203    | 27.071  |
|  4     |  0.9110  | 0.9077 | 5.202    | 27.063  |
|  5     |  0.9110  | 0.9077 | 5.202    | 27.058  |
|  8     |  0.9111  | 0.9078 | 5.200    | 27.039  |
| 12     |  0.9112  | 0.9077 | 5.201    | 27.054  |

**Diễn giải:**
- Từ bậc 1 → 3, R² val tăng đáng kể (0.894 → 0.908) — đường thẳng underfit,
  cong thêm một chút giúp khớp tốt hơn.
- Từ bậc 3 trở đi, R² gần như **bão hoà** (chênh số thập phân thứ 4) → quan
  hệ AT–PE bản chất chỉ hơi cong, bậc cao không lợi gì thêm.
- Bậc 12 vẫn không overfit nghiêm trọng (R² val vẫn ổn) — nhờ dataset có
  9568 mẫu, đủ nhiều để bậc 12 không "học vẹt".
- Theo R² val tuyệt đối thì argmax = **bậc 8**, nhưng thực dụng **bậc 3**
  là lựa chọn tốt hơn (mô hình gọn gấp đôi, chất lượng tương đương).

### 7.2. Kết quả cuối trên TEST (bậc đã chọn = 8)

```
 split |    R²    |   RMSE  |   MSE
 train |  0.9111  |  5.081  | 25.813
   val |  0.9078  |  5.200  | 27.039
  test |  0.9188  |  4.860  | 23.623
```

R² ≈ 0.92 trên test → mô hình giải thích được 92% phương sai PE chỉ từ AT.
Test cao hơn train/val một chút là do may rủi khi chia ngẫu nhiên, không
phải sai sót.

### 7.3. Đồ thị `Figures/Train/polyfit.png`

Hiển thị 4 đường khớp (degree = 1, 2, 5, 12) trên scatter dữ liệu thật:
- Degree 1 (đường thẳng) — bias rõ, underfit nhẹ ở hai biên.
- Degree 2 và 5 — gần như trùng nhau, bám sát mây dữ liệu → "vùng đẹp".
- Degree 12 — bám sát giữa, **vọt xuống ở biên phải** (hiện tượng Runge
  điển hình của polynomial bậc cao). Trục y đã được giới hạn theo khoảng PE
  thật để bậc 12 không kéo méo cả đồ thị.

### 7.4. Đồ thị `Figures/Train/cost_curve.png` (log–log)

Cost của Gradient Descent qua 8000 vòng:
- Iter 1: cost ≈ 2.07 × 10⁵.
- Iter ~30: cost giảm xuống ~30 (giảm 4 bậc độ lớn).
- Iter 50+: gần như chạm đáy ≈ 25.82.

Trục log–log mới thấy được "S-curve" hội tụ — pha học sốc rồi pha tinh
chỉnh. Đây là chữ ký kinh điển của Gradient Descent trên hàm lồi.

### 7.5. NE vs GD — `Results/ccpp_weights.txt`

```
MSE NE = 25.8134      ← Normal Equation (đặc trưng gốc)
MSE GD = 25.8175      ← Gradient Descent (đặc trưng scale theo cột)
```

Hai phương pháp ra MSE gần như trùng (chênh 0.004) → cả hai đều tìm được
cực tiểu toàn cục, xác nhận implementation đúng. Vector w khác nhau **không
phải lỗi** — vì GD chạy trên đặc trưng đã scale theo cột nên ý nghĩa của
từng tham số khác với NE; điều quan trọng là dự đoán cuối cùng tương đương.

---

## 8. Bài học rút ra

1. **Polynomial Regression = Linear Regression trên đặc trưng nâng luỹ
   thừa.** Phi tuyến theo x, tuyến tính theo w → mọi công cụ của linear
   regression dùng được.
2. **Train / Val / Test chuẩn**: dùng val để chọn siêu tham số, test chỉ
   mở 1 lần ở cuối.
3. **Bias–variance**: degree quá thấp → underfit (đường thẳng kém); degree
   quá cao → overfit hoặc vọt biên (Runge). Với dataset mượt + nhiều mẫu,
   "bậc đẹp" thường ở khoảng 2–3.
4. **Feature scaling thoát chết cho GD**: phải scale từng cột đa thức,
   không thì gradient phân kỳ ngay.
5. **NE vs GD**: trên bài toán lồi, hai cách phải hội tụ cùng cực tiểu —
   đó là cách hay để kiểm chứng implementation từ đầu.

---

## 9. Tham khảo

- UCI Machine Learning Repository — *Combined Cycle Power Plant Data Set*.
- Tüfekci, P. (2014). *Prediction of full load electrical power output of
  a base load operated combined cycle power plant using machine learning
  methods.* International Journal of Electrical Power & Energy Systems.
- pickus91/Polynomial-Regression-From-Scratch (GitHub) — nguồn cảm hứng
  cho phần code lõi.
# Polynomial-Regression
