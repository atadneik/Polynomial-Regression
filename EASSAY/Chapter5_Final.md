# CHƯƠNG 5. THỰC NGHIỆM THUẬT TOÁN SUPERVISED LEARNING

Chương này trình bày toàn bộ quá trình triển khai thực nghiệm thuật toán Polynomial Regression trên bộ dữ liệu Auto MPG thông qua một quy trình chuẩn hóa gồm 14 bước. Mỗi bước được thực hiện có chủ đích, dựa trên kết quả và nhận định từ bước trước, nhằm đảm bảo tính minh bạch, khả năng tái lập và chất lượng mô hình cuối cùng.

---

## 5.1 Giới thiệu Auto MPG Dataset

### 5.1.1 Giới thiệu bài toán

Trong chương này, đề tài tiến hành thực nghiệm thuật toán Polynomial Regression trên bộ dữ liệu Auto MPG Dataset nhằm dự đoán mức tiêu thụ nhiên liệu của xe hơi. Mục tiêu của bài toán là xây dựng một mô hình có khả năng dự đoán giá trị MPG – Miles Per Gallon, tức là số dặm xe có thể đi được với một gallon nhiên liệu.

MPG là một chỉ số quan trọng trong ngành ô tô. Giá trị MPG càng cao thì xe càng tiết kiệm nhiên liệu, ngược lại MPG thấp thể hiện xe tiêu hao nhiên liệu nhiều hơn. Việc dự đoán MPG có ý nghĩa thực tiễn trong nhiều trường hợp như đánh giá hiệu suất xe, hỗ trợ thiết kế động cơ, phân tích xu hướng tiết kiệm nhiên liệu và giúp người dùng lựa chọn phương tiện phù hợp.

Bài toán này thuộc nhóm Supervised Learning – Regression vì dữ liệu đầu vào có các đặc trưng kỹ thuật của xe, còn đầu ra cần dự đoán là một giá trị liên tục.

**Luồng bài toán tổng quát:**
```text
Thông số kỹ thuật xe
(cylinders, displacement, horsepower, weight, acceleration, model_year, origin...) 
│
Tiền xử lý dữ liệu và kỹ thuật đặc trưng
│
▼
Polynomial Regression + Ridge Regression 
│
▼
Dự đoán MPG
```

### 5.1.2 Giới thiệu tập dữ liệu Auto MPG

Bộ dữ liệu được sử dụng là Auto MPG Dataset, được thu thập từ kho dữ liệu UCI Machine Learning Repository. Đây là một bộ dữ liệu kinh điển thường được sử dụng trong các bài toán hồi quy của Machine Learning.

Bộ dữ liệu ban đầu gồm 398 mẫu xe và 9 thuộc tính, bao gồm thông tin về dung tích xi-lanh, công suất động cơ, trọng lượng xe, khả năng tăng tốc, năm sản xuất, xuất xứ xe và chỉ số MPG.

**Bảng 5.1: Thông tin tổng quan về bộ dữ liệu Auto MPG**

| Đặc điểm | Thông tin chi tiết |
|---|---|
| **Tên bộ dữ liệu** | Auto MPG Dataset |
| **Nguồn dữ liệu** | UCI Machine Learning Repository |
| **Loại bài toán** | Regression (Hồi quy) |
| **Số lượng mẫu (Rows)** | 398 |
| **Số lượng đặc trưng (Columns)** | 9 (bao gồm 1 biến mục tiêu) |
| **Biến mục tiêu (Target)** | `mpg` (Mức tiêu thụ nhiên liệu - Miles Per Gallon) |
| **Tình trạng giá trị khuyết (Missing)** | 6 giá trị khuyết ở cột `horsepower` |

**Bảng 5.2: Danh sách và mô tả các thuộc tính trong dữ liệu**

| STT | Tên thuộc tính | Phân loại | Kiểu dữ liệu | Ý nghĩa |
|---|---|---|---|---|
| 1 | `mpg` | Continuous | Float | Mức tiêu thụ nhiên liệu (Miles Per Gallon) - **Biến mục tiêu** |
| 2 | `cylinders` | Discrete | Integer | Số lượng xi-lanh của động cơ |
| 3 | `displacement` | Continuous | Float | Dung tích xi-lanh động cơ |
| 4 | `horsepower` | Continuous | Float | Công suất động cơ (Mã lực) |
| 5 | `weight` | Continuous | Float | Trọng lượng của xe (lbs) |
| 6 | `acceleration` | Continuous | Float | Thời gian tăng tốc từ 0-60 mph (giây) |
| 7 | `model_year` | Discrete | Integer | Năm sản xuất (Ví dụ: 70 = 1970) |
| 8 | `origin` | Categorical | Integer | Xuất xứ (1 = Mỹ, 2 = Châu Âu, 3 = Châu Á) |
| 9 | `car_name` | Nominal | String | Tên dòng xe (Chuỗi định danh duy nhất) |

---

## 5.2 Chuẩn Bị và Tiền Xử Lý Dữ Liệu

Chất lượng dữ liệu quyết định phần lớn hiệu năng của bất kỳ mô hình học máy nào. Trước khi tiến hành xây dựng mô hình, cần hiểu rõ bản chất của dữ liệu, làm sạch các bất thường, bổ sung thông tin qua kỹ thuật đặc trưng, và phân chia hợp lý để đảm bảo việc đánh giá là công bằng và không bị rò rỉ thông tin (Data Leakage).

### 5.2.1 Bước 1 — Khám Phá Dữ Liệu (EDA)

Mục tiêu của bước Exploratory Data Analysis (EDA) là xây dựng một bức tranh toàn diện về chất lượng dữ liệu, phân phối các biến và mối quan hệ tiềm ẩn giữa chúng — từ đó đưa ra định hướng kỹ thuật cho các bước tiếp theo.

**Phân phối của các biến số:**

![Phân phối các biến số](../Figures/EDA/distributions.png)

*Hình 5.2: Biểu đồ phân phối các biến trong tập dữ liệu Auto MPG.*

Nhìn vào biểu đồ phân phối, biến mục tiêu `mpg` có phân phối lệch phải (right-skewed) nhẹ, tập trung trong khoảng 15–30 mpg. Đáng chú ý hơn, các biến `displacement` và `weight` đều có dạng phân phối hai đỉnh (bimodal), phản ánh thực tế rằng thị trường xe hơi trong giai đoạn này phân thành hai phân khúc rõ rệt: xe nhỏ, nhẹ (thường là xe Nhật/Âu sau khủng hoảng dầu mỏ 1973) và xe lớn, nặng (truyền thống Mỹ). Sự phân tầng này sẽ là thông tin quan trọng khi thiết kế đặc trưng.

**Phát hiện và đánh giá ngoại lệ (Outliers):**

![Boxplot ngoại lệ](../Figures/EDA/boxplots.png)

*Hình 5.3: Boxplot phát hiện ngoại lệ trong các biến số.*

Phân tích boxplot cho thấy `horsepower` và `acceleration` có một số giá trị cực đại nằm ngoài vùng râu (whisker). Tuy nhiên, sau khi kiểm tra kỹ lưỡng theo ngữ cảnh thực tế (domain knowledge), những giá trị này tương ứng với các dòng xe hiệu năng cao (muscle cars) hoặc xe đặc biệt tiết kiệm — đây là các quan sát hoàn toàn hợp lệ về mặt vật lý, không phải lỗi đo lường. Do đó, quyết định được đưa ra là **giữ nguyên toàn bộ ngoại lệ** để mô hình học được phổ rộng nhất có thể của thực tế.

**Ma trận tương quan (Correlation Matrix):**

![Ma trận tương quan](../Figures/EDA/correlation.png)

*Hình 5.4: Ma trận tương quan giữa các biến.*

Ma trận tương quan tiết lộ hai vấn đề cốt lõi cần giải quyết. Thứ nhất, hiện tượng **đa cộng tuyến (Multicollinearity)** cực mạnh giữa `cylinders`, `displacement`, `weight` và `horsepower`, với hệ số tương quan vượt 0.89. Điều này đặt ra nhu cầu về regularization để mô hình không bị mất ổn định khi các đặc trưng cộng tuyến. Thứ hai, tất cả các biến trên đều tương quan nghịch mạnh với `mpg`, xác nhận nguyên lý vật lý: xe càng nặng, càng lớn động cơ thì càng tiêu thụ nhiều nhiên liệu.

**Tương quan phi tuyến giữa đặc trưng và biến mục tiêu:**

![Tương quan đặc trưng với MPG](../Figures/EDA/features_vs_target.png)

*Hình 5.5: Mối quan hệ giữa các đặc trưng chính và biến mục tiêu `mpg`.*

Đây là phát hiện quan trọng nhất của bước EDA: mối quan hệ giữa `displacement`, `horsepower`, `weight` với `mpg` **không phải đường thẳng mà có dạng cong lồi rõ rệt**. Một mô hình tuyến tính đơn thuần sẽ không thể nắm bắt được hình dạng này, từ đó dẫn đến Underfitting có hệ thống. Phát hiện này là căn cứ khoa học trực tiếp cho quyết định sử dụng **Polynomial Regression** trong nghiên cứu này.

> **Quyết định từ Bước 1:** Dữ liệu có 6 giá trị khuyết trong cột `horsepower` cần được xử lý. Tồn tại đa cộng tuyến cao → cần regularization. Quan hệ phi tuyến tính với `mpg` → xác nhận Polynomial Regression là lựa chọn phù hợp. Tiến hành **Bước 2 — Data Cleaning**.

---

### 5.2.2 Bước 2 — Làm Sạch Dữ Liệu (Data Cleaning)

Nguyên tắc nền tảng trong học máy là "Garbage In, Garbage Out" — dữ liệu nhiễu sẽ tạo ra mô hình nhiễu bất kể thuật toán sử dụng tinh vi đến đâu. Bước này tập trung vào hai thao tác chính: xử lý giá trị khuyết và loại bỏ đặc trưng không có giá trị dự đoán.

Đối với 6 giá trị khuyết trong `horsepower`, phương án thay thế bằng **trung vị (median)** được lựa chọn thay vì trung bình (mean). Lý do là trung vị bền vững hơn trước sự kéo lệch của các giá trị cực đại — với một số xe thể thao có horsepower cao bất thường, trung bình bị đẩy lên quá mức, trong khi trung vị (93.5 HP) phản ánh chính xác hơn "xe điển hình" trong tập dữ liệu.

```python
# Thay thế missing value ở horsepower bằng Median để chống nhiễu từ outlier
median_hp = df['horsepower'].median()  # = 93.5
df['horsepower'] = df['horsepower'].fillna(median_hp)

# Xóa cột định danh car_name — không có ý nghĩa dự đoán
df.drop(columns=['car_name'], inplace=True)
```

Cột `car_name` được loại bỏ bởi vì tên xe là định danh duy nhất cho mỗi dòng — nếu giữ nguyên, mô hình sẽ cố gắng "học thuộc" từng cái tên thay vì học các đặc trưng vật lý thực sự, dẫn đến mất hoàn toàn khả năng tổng quát hóa. Như đã phân tích ở Bước 1, các ngoại lệ được giữ nguyên vì chúng phản ánh tính đa dạng thực tế của thị trường xe hơi.

**Bảng 5.3: Kết quả làm sạch dữ liệu**

| Bước xử lý | Trạng thái trước xử lý | Hành động | Trạng thái sau xử lý |
|---|---|---|---|
| **Xử lý Missing Value** | Cột `horsepower` có 6 giá trị khuyết | Điền bằng giá trị trung vị (Median = 93.5) | Không còn giá trị khuyết |
| **Xử lý Đặc trưng nhiễu** | Tồn tại cột định danh `car_name` | Xóa bỏ cột `car_name` | Dữ liệu chỉ còn 8 đặc trưng (đã tính mpg) |
| **Tổng kết Số lượng mẫu** | 398 dòng | Giữ nguyên các ngoại lệ hợp lệ | **398 dòng** |

Sau bước làm sạch, tập dữ liệu có **398 mẫu** và **8 đặc trưng** (7 đặc trưng + 1 biến mục tiêu), không còn giá trị khuyết. Dữ liệu đã đủ điều kiện để tiến vào giai đoạn kỹ thuật đặc trưng.

> **Quyết định từ Bước 2:** Dữ liệu sạch, không còn khuyết thiếu. Tiến hành **Bước 3 — Feature Engineering** để bổ sung thông tin từ domain knowledge.

---

### 5.2.3 Bước 3 — Kỹ Thuật Đặc Trưng (Feature Engineering)

Feature Engineering là bước mà kiến thức ngành (domain knowledge) được chuyển hóa thành toán học. Mục tiêu là tạo ra một không gian đặc trưng (feature space) phong phú và có ý nghĩa hơn, giúp mô hình "nhìn thấy" những mẫu dữ liệu mà nó không thể nhận ra nếu chỉ dùng đặc trưng thô.

**Mã hóa biến phân loại (Categorical Encoding):**

Biến `origin` nhận giá trị 1 (Mỹ), 2 (Châu Âu) hoặc 3 (Châu Á) là biến phân loại danh nghĩa (nominal), không có thứ bậc. Nếu giữ nguyên dạng số, mô hình sẽ hiểu nhầm rằng "Châu Á > Châu Âu > Mỹ" theo nghĩa toán học — một diễn giải hoàn toàn sai về mặt ngữ nghĩa. One-Hot Encoding phá bỏ định kiến tuyến tính này bằng cách tạo 3 vector nhị phân độc lập.

**Tạo đặc trưng mới từ tri thức ngành (Domain Feature Creation):**

Dựa trên nguyên lý cơ học xe hơi, hai đặc trưng tổ hợp (interaction feature) được tạo ra:

```python
# One-Hot Encoding cho biến phân loại origin
X = pd.get_dummies(X, columns=['origin'])

# Feature Creation: Tỉ lệ Trọng lượng / Mã lực
# (Xe có tỉ số này càng cao → động cơ phải kéo tải nặng hơn → tốn xăng hơn)
X['weight_per_hp'] = X['weight'] / X['horsepower']

# Feature Creation: Dung tích xi-lanh trung bình
X['displacement_per_cylinder'] = X['displacement'] / X['cylinders']
```

![Đặc trưng mới so với MPG](../Figures/FE/new_features.png)

*Hình 5.6: Tương quan của đặc trưng `weight_per_hp` mới tạo với biến mục tiêu `mpg`.*

Nhìn vào biểu đồ, `weight_per_hp` thể hiện tương quan thuận chiều rất rõ ràng và gần như tuyến tính với `mpg`. Điều này không phải ngẫu nhiên — đặc trưng này gom nhóm thông tin đa cộng tuyến từ `weight` và `horsepower` thành một lăng kính duy nhất có ý nghĩa vật lý: xe có cùng mã lực nhưng nhẹ hơn sẽ tiêu thụ ít nhiên liệu hơn. Đây sẽ là một trong những đặc trưng mạnh nhất của mô hình.

**Lý do không thực hiện Scaling và Polynomial tại bước này:**

Một nguyên tắc quan trọng cần nhấn mạnh: việc chuẩn hóa dữ liệu (StandardScaler) và tạo đặc trưng đa thức (PolynomialFeatures) được **chủ động dời sang bên trong Pipeline** ở bước modeling. Nếu thực hiện tại đây, các tham số của Scaler (mean, variance) sẽ được tính trên toàn bộ dữ liệu — bao gồm cả tập Test — khiến thông tin từ tập Test "rò rỉ" vào quá trình huấn luyện, tạo ra ảo tưởng về hiệu năng tốt hơn thực tế (Data Leakage).

Sau bước này, Feature Set bao gồm **398 mẫu** và **11 đặc trưng**.

> **Quyết định từ Bước 3:** Feature set đã đầy đủ và có ý nghĩa. Tiến hành **Bước 4 — Data Split** để phân chia dữ liệu theo cách chống Data Leakage.

---

### 5.2.4 Bước 4 — Phân Chia Dữ Liệu (Data Split)

Việc đánh giá một mô hình cần tuân thủ nguyên tắc công bằng: mô hình không được phép "nhìn thấy" dữ liệu đánh giá trong quá trình học. Điều này đòi hỏi phải phân chia dữ liệu thành ba tập độc lập với vai trò rõ ràng.

Chiến lược phân chia được áp dụng là **70/15/15** (Train / Validation / Test):

```python
from sklearn.model_selection import train_test_split

# Chia 70% Train, 30% Temp — random_state=42 đảm bảo tính tái lập
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# Chia tiếp Temp thành 15% Validation và 15% Test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)
```

Kết quả thu được: **278 mẫu Train** (dùng để huấn luyện), **60 mẫu Validation** (dùng để tinh chỉnh siêu tham số), và **60 mẫu Test** (được "cất kín", chỉ mở ra ở bước đánh giá cuối cùng). Việc sử dụng `random_state=42` đảm bảo bất kỳ ai chạy lại code đều nhận được đúng cùng kết quả đó (reproducibility).

![So sánh phân phối 3 tập](../Figures/Split/distribution_comparison.png)

*Hình 5.7: Density plot so sánh phân phối biến mục tiêu `mpg` trên 3 tập Train, Validation và Test.*

Ba đường cong mật độ xác suất có hình dáng rất tương đồng nhau — đỉnh (peak) cùng nằm trong khoảng 15–20 mpg, phần đuôi phải (right tail) có độ dài và hình dạng gần như trùng khớp. Điều này chứng minh rằng phép chia ngẫu nhiên đã bảo toàn được cấu trúc phân phối của tập gốc. Mô hình học từ tập Train sẽ không gặp phải sự khác biệt phân phối (distribution shift) khi đối mặt với tập Validation hay Test.

> **Quyết định từ Bước 4:** Ba tập dữ liệu đạt tiêu chuẩn công bằng và an toàn. Tiến sang giai đoạn xây dựng mô hình với **Bước 5 — Baseline Model**.

---

## 5.3 Xây Dựng và Lựa Chọn Mô Hình

Với dữ liệu đã được chuẩn bị kỹ lưỡng, giai đoạn này bắt đầu từ việc xây dựng một mốc so sánh tối thiểu (baseline), sau đó thử nghiệm nhiều kiến trúc mô hình, tinh chỉnh siêu tham số và kết thúc bằng việc huấn luyện mô hình cuối cùng trên toàn bộ dữ liệu có sẵn.

### 5.3.1 Bước 5 — Mô Hình Cơ Sở (Baseline Model)

Trước khi đầu tư vào các kỹ thuật phức tạp, điều cần thiết là xác lập một **mốc so sánh (benchmark)**: đây là hiệu năng tối thiểu mà bất kỳ mô hình "thông minh" hơn nào cũng bắt buộc phải vượt qua. Nếu mô hình phức tạp không đánh bại được baseline, đó là dấu hiệu cho thấy sự phức tạp không mang lại giá trị thực.

Mô hình baseline được chọn là **Linear Regression** kết hợp với StandardScaler, được gói trong một `sklearn.pipeline.Pipeline` ngay từ đầu để đảm bảo Scaler chỉ được fit trên tập Train — không bao giờ "nhìn" vào tập Validation hay Test.

| Tập dữ liệu | R² | RMSE (mpg) | MAE (mpg) |
|---|---|---|---|
| Train | 0.8454 | 3.1068 | 2.3109 |
| Validation | 0.8481 | 2.7304 | 2.1784 |

Với R² ≈ 0.85, mô hình tuyến tính đơn thuần đã giải thích được 85% sự biến thiên của mức tiêu thụ nhiên liệu — một kết quả không tệ. Tuy nhiên, phân tích đồ thị phần dư (Residuals) tiết lộ một vấn đề cơ bản:

![Hiệu năng Baseline](../Figures/Baseline/baseline_performance.png)

*Hình 5.8: Biểu đồ Actual vs Predicted (trái) và Residuals (phải) của mô hình Baseline.*

Nhìn vào biểu đồ Residuals (phần bên phải), sai số **không phân bố ngẫu nhiên** quanh trục 0 như lý thuyết yêu cầu. Thay vào đó, chúng tạo thành một đường cong hình chữ U rõ rệt: ở vùng dự đoán thấp (MPG nhỏ) và vùng dự đoán cao (MPG lớn), sai số đều có xu hướng lệch cùng chiều. Hiện tượng này — trong thống kê gọi là **heteroscedastic residuals với systematic pattern** — là bằng chứng không thể chối cãi rằng mô hình tuyến tính quá đơn giản để nắm bắt hình dạng cong của dữ liệu (Underfitting). Biểu đồ này trực tiếp xác nhận kết luận từ EDA ở Bước 1: dữ liệu mang tính phi tuyến tính.

> **Quyết định từ Bước 5:** Baseline R² = 0.8481, RMSE = 2.7304 là mốc tối thiểu. Biểu đồ Residuals chứng minh sự cần thiết của mô hình phi tuyến. Chuyển sang **Bước 6 — Model Selection** để thử nghiệm Polynomial Regression.

---

### 5.3.2 Bước 6 — Lựa Chọn Mô Hình (Model Selection)

Thay vì đầu tư ngay vào một kiến trúc duy nhất, chiến lược đúng đắn là **đánh giá nhanh nhiều ứng viên** (candidate models) để xác định hướng đi tiềm năng nhất trước khi tinh chỉnh chuyên sâu. Năm ứng viên được thử nghiệm, tất cả đều được gói trong Pipeline để chống Data Leakage:

1. **Baseline** — Linear Regression (degree=1)
2. **Poly (Deg 2) + Linear** — Polynomial bậc 2 + OLS
3. **Poly (Deg 3) + Linear** — Polynomial bậc 3 + OLS
4. **Poly (Deg 2) + Ridge** — Polynomial bậc 2 + Ridge (L2)
5. **Poly (Deg 2) + Lasso** — Polynomial bậc 2 + Lasso (L1)

| Mô hình | Train R² | Val R² | Train RMSE | Val RMSE |
|---|---|---|---|---|
| Baseline (Linear) | 0.8454 | 0.8481 | 3.1068 | 2.7304 |
| Poly (Deg 2) + Linear | 0.9135 | 0.8638 | 2.3247 | 2.5851 |
| **Poly (Deg 3) + Linear** | 0.9744 | **−93.5356** | 1.2642 | **68.1064** |
| **Poly (Deg 2) + Ridge** | 0.8820 | **0.8696** | 2.7143 | **2.5295** |
| Poly (Deg 2) + Lasso | 0.8498 | 0.8276 | 3.0628 | 2.9085 |

![So sánh các mô hình](../Figures/Selection/model_comparison.png)

*Hình 5.9: Biểu đồ so sánh Train RMSE và Validation RMSE của 5 ứng viên.*

Kết quả thực nghiệm đem lại ba nhận định cốt lõi. **Thứ nhất**, ngay khi nâng bậc đa thức lên 2, R² trên tập Validation nhảy vọt từ 0.8481 lên 0.8638 — xác nhận giả thuyết phi tuyến tính từ EDA là đúng đắn. **Thứ hai**, mô hình Polynomial bậc 3 là một thảm họa điển hình của Overfitting: Train RMSE cực thấp (1.26 — mô hình gần như học thuộc lòng dữ liệu) nhưng Val RMSE bùng nổ lên đến **68.1** — gấp hơn 25 lần mô hình baseline. Khi số lượng đặc trưng đa thức tăng theo cấp số nhân với bậc, mô hình bắt đầu học cả nhiễu đo lường, mất hoàn toàn khả năng tổng quát hóa. **Thứ ba**, Regularization (Ridge) thể hiện vai trò rõ rệt: so với Poly (Deg 2) + Linear, mô hình Ridge đạt Val RMSE **thấp hơn** (2.5295 vs 2.5851) trong khi Train RMSE lại cao hơn — đây là dấu hiệu lành mạnh, cho thấy Ridge đang giảm bớt Overfitting bằng cách "kìm kẹp" các trọng số.

> **Quyết định từ Bước 6:** Loại bỏ hoàn toàn bậc 3 vì Overfitting quá nặng. **Poly (Deg 2) + Ridge** được chọn là ứng viên tốt nhất để tinh chỉnh. Chuyển sang **Bước 7 — Hyperparameter Tuning**.

---

### 5.3.3 Bước 7 — Tinh Chỉnh Siêu Tham Số (Hyperparameter Tuning)

Với cấu trúc mô hình đã được chốt (Polynomial bậc 2 + Ridge), bước tiếp theo là tìm ra giá trị tối ưu cho **siêu tham số alpha** — hệ số phạt L2 của Ridge Regression. Alpha kiểm soát mức độ "kìm kẹp" các trọng số: alpha nhỏ thì gần với OLS thông thường (dễ Overfitting), alpha lớn thì ép mạnh về 0 (dễ Underfitting).

Thay vì quét tuyến tính, không gian tìm kiếm được thiết lập trên **thang đo logarit** — vì alpha có thể tối ưu ở bất kỳ độ lớn nào từ 0.001 đến 10,000, và thang log đảm bảo mỗi bước nhảy đại diện cho cùng một tỉ lệ thay đổi tương đối:

```python
import numpy as np

# Quét 100 giá trị alpha đều nhau trên thang logarit từ 10^-3 đến 10^4
alphas = np.logspace(-3, 4, 100)

best_alpha, best_val_rmse = None, float('inf')
for alpha in alphas:
    pipeline.set_params(regressor__alpha=alpha)
    pipeline.fit(X_train, y_train)
    val_rmse = np.sqrt(mean_squared_error(y_val, pipeline.predict(X_val)))
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_alpha = alpha
```

![Validation Curve](../Figures/Tuning/validation_curve.png)

*Hình 5.10: Validation Curve — biểu đồ RMSE theo giá trị alpha trên thang logarit.*

Biểu đồ Validation Curve là một minh họa sống động của lý thuyết Bias-Variance Tradeoff. Đường cong được chia thành ba vùng rõ ràng. **Vùng trái (alpha < 1):** Train RMSE rất thấp nhưng Val RMSE cao — khoảng cách lớn giữa hai đường là dấu hiệu Overfitting nhẹ, khi mô hình quá tự do để khớp chi tiết của tập Train. **Vùng phải (alpha > 100):** Cả Train RMSE và Val RMSE cùng tăng vọt — Underfitting xảy ra khi alpha quá lớn ép các trọng số gần về 0, làm mô hình mất sức biểu đạt. **Vùng "Sweet Spot" (alpha ≈ 1–10):** Đây là điểm cân bằng lý tưởng, nơi Val RMSE đạt đáy thấp nhất. Quá trình tối ưu hóa xác định được **Best Alpha = 0.2984**, với hiệu năng tại điểm tối ưu: Val R² = 0.8720, Val RMSE = 2.5065.

> **Quyết định từ Bước 7:** Chốt cấu hình cuối cùng: **Polynomial (degree=2) + Ridge(alpha=0.2984)**. Chuyển sang **Bước 8 — Train Final Model**.

---

### 5.3.4 Bước 8 — Huấn Luyện Mô Hình Cuối Cùng (Final Model Training)

Một khi cấu trúc mô hình và siêu tham số đã được chốt, tập Validation không còn vai trò tinh chỉnh nữa. Lúc này, chiến lược **gộp tập Validation vào tập Train** (X_train_full = X_train ∪ X_val) được áp dụng để mô hình học từ nhiều dữ liệu hơn trước khi đối mặt với tập Test.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
import numpy as np

# Gộp Train + Validation thành tập huấn luyện cuối cùng (338 mẫu)
X_train_full = np.vstack([X_train, X_val])
y_train_full = np.concatenate([y_train, y_val])

# Xây dựng Pipeline hoàn chỉnh: Poly → Scale → Ridge
final_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('regressor', Ridge(alpha=0.2984))
])

# Huấn luyện trên toàn bộ dữ liệu có sẵn (trừ Test)
final_pipeline.fit(X_train_full, y_train_full)
```

Việc gộp dữ liệu (từ 278 lên 338 mẫu, tăng 21.6%) giúp củng cố độ vững vàng (robustness) của mô hình mà không vi phạm nguyên tắc chống leakage — vì tập Test vẫn được "cất kín" hoàn toàn. Một điểm đáng chú ý về hiệu quả tính toán: toàn bộ quá trình fit Pipeline (bao gồm sinh đặc trưng đa thức, chuẩn hóa và hồi quy Ridge) chỉ mất **16.61 mili-giây**. Ridge Regression sử dụng nghiệm dạng đóng (closed-form solution) thông qua Cholesky decomposition — không cần lặp hội tụ như Gradient Descent — khiến nó cực kỳ hiệu quả ngay cả khi số lượng đặc trưng sau Polynomial (66 đặc trưng) tương đối lớn.

Mô hình đã được huấn luyện và xuất dưới định dạng Joblib tại `models/final_polynomial_ridge_model.joblib`, sẵn sàng cho quá trình đánh giá.

---

## 5.4 Kiểm Định và Đánh Giá

Giai đoạn này trả lời câu hỏi cốt lõi: mô hình có thực sự học được tri thức từ dữ liệu, hay chỉ đơn giản là học thuộc lòng tập Train? Quá trình đánh giá được thực hiện theo nhiều góc độ bổ trợ lẫn nhau để đảm bảo kết luận toàn diện và đáng tin cậy.

### 5.4.1 Bước 9 — Đánh Giá Trên Tập Test (Evaluation Metrics)

Đây là thời điểm "mở niêm phong" tập Test — 60 mẫu dữ liệu chưa từng được mô hình nhìn thấy trong suốt toàn bộ quá trình xây dựng. Đây là bài kiểm tra trung thực nhất về khả năng tổng quát hóa.

| Chỉ số | Giá trị | So sánh với Baseline |
|---|---|---|
| R² Score | **0.9397** | +11.2% (từ 0.8454) |
| RMSE (mpg) | **1.9906** | −35.9% (từ 3.1068) |
| MAE (mpg) | **1.4564** | − |
| MSE | **3.9626** | − |

![Đánh giá mô hình trên Test Set](../Figures/Evaluation/evaluation_plots.png)

*Hình 5.11: Actual vs Predicted (trái) và phân phối sai số (phải) trên tập Test.*

So sánh hai biểu đồ này với biểu đồ baseline ở Hình 5.8 cho thấy sự cải thiện rõ rệt. Trong biểu đồ Actual vs Predicted, các điểm dữ liệu bám sát đường đứt nét y = x một cách đồng đều trên toàn dải — đường cong hình chữ U đặc trưng của Underfitting đã hoàn toàn biến mất. Quan trọng hơn, biểu đồ phân phối sai số (Residuals) bên phải có hình dạng chuông đối xứng cân xứng quanh giá trị 0. Sai số phân bố chuẩn (Normal) quanh mốc 0 — hay còn gọi là **Unbiased Residuals** — là chỉ số vàng xác nhận rằng mô hình không còn bỏ sót bất kỳ mẫu hệ thống nào trong dữ liệu. Những sai số còn lại hoàn toàn là Irreducible Error — nhiễu ngẫu nhiên mà không thuật toán nào có thể loại bỏ.

> **Quyết định từ Bước 9:** Hiệu năng vượt kỳ vọng với R² = 0.9397. Tuy nhiên, cần xác nhận rằng kết quả này không phải do may mắn từ tập Test dễ. Chuyển sang **Bước 10 — Cross Validation**.

---

### 5.4.2 Bước 10 — Kiểm Định Chéo K-Fold (Cross Validation)

Một kết quả R² = 0.9397 trên tập Test 60 mẫu có thể phần nào do yếu tố ngẫu nhiên: tập Test có thể vô tình chứa các mẫu dễ dự đoán hơn mức trung bình. Kỹ thuật **K-Fold Cross Validation** giải quyết vấn đề này bằng cách đánh giá mô hình trên 10 tập con (fold) khác nhau, mỗi lần dùng 9/10 dữ liệu để train và 1/10 để test, rồi lấy trung bình.

```python
from sklearn.model_selection import cross_validate, KFold

# 10-Fold CV — mỗi fold là một "chiến trường" độc lập
kf = KFold(n_splits=10, shuffle=True, random_state=42)
cv_results = cross_validate(
    final_pipeline, X, y,
    cv=kf,
    scoring=['r2', 'neg_root_mean_squared_error']
)

mean_r2  = np.mean(cv_results['test_r2'])           # = 0.8728
std_r2   = np.std(cv_results['test_r2'])            # = 0.0311
mean_rmse = -np.mean(cv_results['test_neg_root_mean_squared_error'])  # = 2.7432
```

![Kết quả Cross Validation](../Figures/CV/cv_scores.png)

*Hình 5.12: Boxplot R² và RMSE từ 10 lần Cross Validation.*

**Mean CV R² = 0.8728 ± 0.0311** và **Mean CV RMSE = 2.7432 ± 0.5153**. Nhìn vào Boxplot, hộp chữ nhật (interquartile range) rất hẹp trong vùng 0.85–0.90, đuôi hộp thấp nhất dừng ở khoảng 0.80 — không hề có fold nào cho kết quả âm hay đột ngột tụt xuống mức thấp bất thường (dấu hiệu Overfitting nghiêm trọng). Độ lệch chuẩn 0.0311 là một con số rất nhỏ, chứng tỏ mô hình **ổn định cao trước sự thay đổi của dữ liệu train**.

Sự khác biệt giữa R² = 0.9397 (Test Set, Bước 9) và R² = 0.8728 (CV Mean) được giải thích thỏa đáng: có 1–2 fold vọt lên trên 0.90, cho thấy tập Test ở Bước 9 thực sự "dễ dự đoán" hơn trung bình một chút. Con số **87.28%** chính là kỳ vọng hiệu năng thực tế nhất khi triển khai mô hình trên dữ liệu mới.

> **Quyết định từ Bước 10:** Mô hình vượt qua bài Stress Test. Kỳ vọng thực tế là R² ≈ 87%. Chuyển sang **Bước 11 — Experiment Management** để lưu vết và lý giải giới hạn này.

---

### 5.4.3 Bước 11 — Quản Lý Thí Nghiệm (Experiment Management)

Tính tái lập (Reproducibility) là nền tảng của nghiên cứu khoa học. Mọi thí nghiệm — dù cho kết quả xuất sắc đến đâu — đều vô nghĩa nếu không thể tái tạo lại chính xác. Bước này ghi lại đầy đủ "công thức" đã tạo ra kết quả hiện tại:

```json
{
  "experiment_id": "poly_ridge_v1",
  "timestamp": "2026-07-10",
  "random_seed": 42,
  "model": "Ridge",
  "degree": 2,
  "alpha": 0.2984,
  "features_used": [
    "cylinders", "displacement", "horsepower", "weight",
    "acceleration", "model_year", "origin_1", "origin_2", "origin_3",
    "weight_per_hp", "displacement_per_cylinder"
  ],
  "test_r2": 0.9397,
  "cv_r2_mean": 0.8728,
  "cv_r2_std": 0.0311
}
```

Quan trọng hơn, bước này trả lời câu hỏi then chốt: **tại sao R² kỳ vọng dừng ở 87% mà không thể cao hơn?** Phân tích cho thấy có ba nguyên nhân gốc rễ không thể giải quyết bằng kỹ thuật:

**Thứ nhất, Omitted Variable Bias (Thiên lệch do thiếu biến):** Mức tiêu thụ nhiên liệu trong thực tế còn phụ thuộc vào hệ số cản gió (aerodynamics), loại hộp số (manual vs automatic), tỉ số truyền (gear ratios) và loại lốp xe — những dữ liệu hoàn toàn vắng mặt trong tập Auto MPG. Không có mô hình nào có thể dự đoán những gì nó không có quyền "nhìn thấy". **Thứ hai, Nhiễu đo lường lịch sử:** Dữ liệu được thu thập trong thập niên 1970–1980, khi tiêu chuẩn đo lường mã lực (SAE gross vs SAE net) chưa đồng nhất — sai số vật lý hệ thống này đã "ám" vào dữ liệu và không thể loại bỏ. **Thứ ba, Giới hạn của thuật toán toàn cục:** Polynomial Regression áp dụng một đường cong duy nhất cho toàn bộ không gian dữ liệu, không thể nắm bắt những thay đổi cục bộ đột ngột (ví dụ: công nghệ động cơ thay đổi theo từng năm do khủng hoảng dầu mỏ).

Kết luận: **87% là trần vật lý của bài toán này với tập dữ liệu hiện có** — không phải điểm yếu của thuật toán. Cố ép lên 95% bằng cách tăng bậc đa thức sẽ chỉ dẫn đến Overfitting như đã chứng minh ở Bước 6.

> **Quyết định từ Bước 11:** Thí nghiệm được lưu vết đầy đủ tại `Results/experiment_log.json`. Chuyển sang **Bước 12** để kiểm định thống kê xem sự cải tiến có ý nghĩa thật sự không.

---

### 5.4.4 Bước 12 — Kiểm Định Thống Kê (Statistical Validation)

Một câu hỏi quan trọng cần trả lời: sự cải tiến từ R² = 0.8438 (Linear Baseline) lên 0.8728 (Polynomial Ridge) có thực sự có **ý nghĩa thống kê** không, hay chỉ là biến động ngẫu nhiên trong dữ liệu? Phương pháp **Paired T-Test** (kiểm định T bắt cặp) được áp dụng, trong đó hai mô hình được đánh giá trên **đúng 10 fold giống hệt nhau** — đảm bảo tính so sánh công bằng.

```python
from scipy import stats

# Chạy CV cho cả 2 mô hình trên cùng 10 fold
scores_poly   = cross_val_score(final_pipeline, X, y, cv=kf, scoring='r2')
scores_linear = cross_val_score(baseline_pipeline, X, y, cv=kf, scoring='r2')

# Paired T-Test: kiểm định H0 "hai mô hình có hiệu năng bằng nhau"
t_stat, p_value = stats.ttest_rel(scores_poly, scores_linear)
# Kết quả: p_value = 1.626785e-03
```

![So sánh bắt cặp 10 Fold](../Figures/Stats/paired_comparison.png)

*Hình 5.13: Pointplot so sánh R² của Linear và Polynomial trên từng Fold.*

**P-value = 0.00163**, nhỏ hơn ngưỡng 0.05 hàng chục lần. Theo lý thuyết kiểm định giả thuyết, điều này có nghĩa: xác suất để hai mô hình thực sự có sức mạnh ngang nhau nhưng Polynomial vô tình đạt điểm cao hơn do may mắn chỉ là 0.163% — thực tế là không đáng kể. Nhìn vào biểu đồ Pointplot, đường Polynomial (màu xanh) **luôn nằm trên** đường Linear (màu cam) trong tất cả 10 fold mà không có bất kỳ ngoại lệ nào — kể cả ở những fold dữ liệu khó nhất.

> **Quyết định từ Bước 12:** Sự vượt trội của Polynomial là **có ý nghĩa thống kê (Statistically Significant)**. Chuyển sang **Bước 13 — Error Analysis** để tìm hiểu mô hình thất bại ở đâu.

---

### 5.4.5 Bước 13 — Phân Tích Lỗi (Error Analysis)

Một mô hình xuất sắc về mặt tổng thể vẫn có thể có những điểm mù cụ thể. Phân tích lỗi (Error Analysis) không nhìn vào số liệu trung bình mà tập trung vào những trường hợp mà mô hình **sai nhiều nhất** — đây là nơi hé lộ những giới hạn thực sự của hệ thống.

Bảng dưới đây trình bày 5 dòng xe mà mô hình dự đoán sai lớn nhất trên tập Test:

| Thực Tế (MPG) | Dự Đoán (MPG) | Sai Số | Trọng Lượng (lbs) | Mã Lực | Năm SX |
|---|---|---|---|---|---|
| 32.0 | 37.9 | **5.9** | 1,965 | 67 | 82 |
| 20.0 | 25.4 | **5.4** | 2,279 | 88 | 73 |
| 23.7 | 28.0 | **4.3** | 2,420 | 100 | 80 |

![Phân tích lỗi](../Figures/Errors/error_analysis.png)

*Hình 5.14: Phân bố sai số theo MPG thực tế (trái) và theo trọng lượng & xuất xứ (phải).*

Hai biểu đồ tiết lộ một pattern nhất quán: sai số lớn tập trung ở **xe MPG cao (> 35 mpg)** — những xe siêu tiết kiệm nhiên liệu — và đặc biệt là **xe nhẹ (< 2,500 lbs) có xuất xứ Nhật/Châu Âu**. Mô hình nhất quán dự đoán thấp hơn thực tế (Under-predict) cho nhóm này.

Giải thích cho hiện tượng này rất rõ ràng: các xe Nhật/Âu phân khúc tiết kiệm thế hệ 1980 đã tích hợp các công nghệ chuyên biệt (hộp số đặc biệt, tỉ số truyền tối ưu, hệ số khí động học cao) giúp chúng tiêu thụ ít xăng hơn đáng kể so với những gì các thông số cơ bản trong tập dữ liệu có thể giải thích. Vì những đặc trưng này không tồn tại trong dữ liệu, không có kỹ thuật Data Cleaning hay Feature Engineering nào có thể tạo ra thông tin từ chỗ không có.

> **Quyết định từ Bước 13:** Lỗi xuất phát từ **giới hạn dữ liệu**, không phải giới hạn thuật toán. Không cần quay lại Bước 1 hay 3. Chấp nhận ranh giới này của mô hình và tiến sang **Bước 14 — Model Interpretability**.

---

## 5.5 Giải Thích Mô Hình (Model Interpretability — Bước 14)

Một mô hình có hiệu năng tốt nhưng không giải thích được đôi khi còn nguy hiểm hơn một mô hình kém — vì không ai biết khi nào nó sẽ thất bại và tại sao. Bước này "mở hộp đen" của Pipeline, trả lời câu hỏi: **mô hình đã học được gì, và những gì nó học có thuận với lý lẽ thực tế không?**

Thách thức kỹ thuật ở đây là `PolynomialFeatures` đã biến đổi 11 đặc trưng gốc thành **66 đặc trưng đa thức** (bậc 1, bậc 2 và tương tác chéo). Cần dịch ngược các hệ số của Ridge về lại không gian đặc trưng có tên gọi để hiểu được ngữ nghĩa:

```python
# Trích xuất các bước từ Pipeline đã train
poly  = final_pipeline.named_steps['poly']
ridge = final_pipeline.named_steps['regressor']

# Lấy tên 66 đặc trưng đa thức và hệ số tương ứng
poly_feature_names = poly.get_feature_names_out(original_features)
coefficients       = ridge.coef_

# Ghép tên và hệ số, sắp xếp theo ảnh hưởng
feature_importance = pd.Series(coefficients, index=poly_feature_names)
feature_importance_sorted = feature_importance.reindex(
    feature_importance.abs().sort_values(ascending=False).index
)
```

![Feature Importance](../Figures/Interpretability/feature_importance.png)

*Hình 5.15: Biểu đồ Feature Importance — hệ số Ridge của 66 đặc trưng đa thức, sắp xếp theo độ lớn.*

Biểu đồ Feature Importance tiết lộ ba insight quan trọng. **Insight thứ nhất:** Đặc trưng tự tạo `weight_per_hp` và bình phương của nó (`weight_per_hp²`) đứng ở **vị trí hàng đầu** về ảnh hưởng dương (thanh màu xanh dài nhất). Chiến thắng này của Feature Engineering xác nhận rằng domain knowledge có thể tạo ra đặc trưng mạnh hơn cả các đặc trưng gốc của tập dữ liệu — `weight` và `horsepower` riêng lẻ đều không đứng đầu, nhưng tỉ số của chúng thì có.

**Insight thứ hai:** Các đặc trưng chứa `weight` — đặc biệt là tương tác `weight × model_year` và `weight` độc lập — có **hệ số âm lớn nhất** (thanh màu đỏ dài nhất, chỉ sang trái). Cứ mỗi đơn vị tăng trong trọng lượng, mô hình trừ thẳng tay điểm MPG — hoàn toàn nhất quán với vật lý học cơ bản: trọng lượng là kẻ thù số một của hiệu quả nhiên liệu.

**Insight thứ ba:** Các đặc trưng interaction (kết hợp hai biến, ví dụ `weight × displacement`) chiếm ưu thế hơn hẳn so với các đặc trưng đơn lẻ trong Top 20. Đây là lý do quan trọng giải thích tại sao Linear Regression thất bại ở Bước 5: thế giới thực không hoạt động theo từng biến độc lập, mà mọi thông số của xe hơi đều tương tác chéo với nhau — và chỉ Polynomial Regression mới có khả năng nắm bắt các tương tác này.

Logic của mô hình hoàn toàn khớp với Vật lý và Động lực học xe hơi. Mô hình không hề "học vẹt" — nó đã thực sự hiểu được cấu trúc nhân quả của dữ liệu.

---

## 5.6 Tổng Kết Chương

Chương này đã trình bày toàn bộ quá trình triển khai Polynomial Regression theo pipeline 14 bước nghiêm ngặt, từ dữ liệu thô đến mô hình sẵn sàng triển khai. Bảng dưới đây tổng hợp hiệu năng so sánh giữa mô hình baseline và mô hình tốt nhất:

| Tiêu Chí Đánh Giá | Baseline (Linear Reg.) | **Mô Hình Tốt Nhất (Poly Deg 2 + Ridge)** | Cải Thiện |
|---|---|---|---|
| Test R² | 0.8454 | **0.9397** | **+11.2%** |
| Test RMSE (mpg) | 3.1068 | **1.9906** | **−35.9%** |
| CV R² (mean ± std) | 0.8438 | **0.8728 ± 0.0311** | **+3.4%** |
| CV RMSE (mean) | ~2.73 | **2.7432 ± 0.5153** | Ổn định hơn |
| P-value (Paired T-Test) | — | **0.00163** | Significant |
| Thời gian huấn luyện | — | **16.61 ms** | Rất hiệu quả |

Từ 14 bước triển khai, năm bài học cốt lõi được rút ra:
1. **Pipeline là công cụ bắt buộc** — không phải tùy chọn — để ngăn chặn Data Leakage; Scaler và PolynomialFeatures phải nằm bên trong Pipeline, không được fit độc lập trên dữ liệu test. 
2. **Polynomial bậc 3 là cái bẫy kinh điển của Overfitting** — Val RMSE = 68.1 là lời cảnh báo rõ ràng rằng việc tăng độ phức tạp không kiểm soát sẽ phá hủy khả năng tổng quát hóa. 
3. **Ridge Regularization không chỉ là kỹ thuật bổ trợ** mà là điều kiện tiên quyết khi làm việc với không gian đặc trưng đa thức bậc cao — nơi đa cộng tuyến giữa các đặc trưng là không thể tránh khỏi. 
4. **Feature Engineering từ domain knowledge (`weight_per_hp`) tạo ra đột phá** — vượt qua tất cả đặc trưng gốc, khẳng định rằng hiểu biết về bài toán quan trọng không kém thuật toán. 
5. **Giới hạn của mô hình (R² ≈ 87%) là trần vật lý do thiếu dữ liệu** về công nghệ xe, không phải điểm yếu của thuật toán — việc nhận biết giới hạn này là biểu hiện của tư duy khoa học trưởng thành.

---

*Toàn bộ mã nguồn triển khai 14 bước được lưu trong các file: `eda.py`, `feature_engineering.py`, `train.py`, `baseline.py`, `model_selection.py`, `tuning.py`, `train_model.py`, `evaluation.py`, `cross_validation.py`, `experiment_management.py`, `statistical_validation.py`, `error_analysis.py`, `model_interpretability.py`. Kết quả thực nghiệm và log thí nghiệm được lưu tại thư mục `Results/`.*
