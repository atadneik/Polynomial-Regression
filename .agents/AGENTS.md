# ML Pipeline Rules — Polynomial Regression Project

## Nguyên tắc cốt lõi

> **Controlled Experiment**: Chỉ thay đổi **1 biến** ở mỗi lần thử nghiệm.
> Không bao giờ bỏ qua bước. Không bao giờ chuyển bước khi artifact chưa hoàn thành.

---

## Quy trình bắt buộc — 14 Bước Pipeline

Agent **PHẢI** tuân thủ đúng thứ tự 14 bước dưới đây. Mỗi bước phải hoàn tất trước khi chuyển sang bước tiếp theo.

### Cấu trúc mỗi Bước

Mỗi bước khi thực hiện **BẮT BUỘC** phải tuân thủ nghiêm ngặt 7 thành phần (Vibe Coding Checklist):

1. **Goal** — Mục tiêu cụ thể của bước
2. **Input** — Dữ liệu/artifact đầu vào (từ bước trước)
3. **Tasks** — Danh sách công việc cần thực hiện chi tiết
4. **Output** — Sản phẩm đầu ra cụ thể
5. **Insight** — Nhận định rút ra được sau khi phân tích
6. **Decision** — Quyết định hành động tiếp theo (tiếp tục / quay lại / điều chỉnh)
7. **Artifact** — File code, biểu đồ, file kết quả, và BẮT BUỘC xuất báo cáo vào file `BuocX.md` (ví dụ Bước 1 là `Buoc1.md`).

---

### Bước 1 — EDA
- **Goal**: Phân tích và hiểu đặc điểm của bộ dữ liệu gốc.
- **Input**: Raw dataset (`data/...`).
- **Tasks**: Phân tích phân phối (Distribution), phát hiện giá trị thiếu (Missing values), nhận diện ngoại lệ (Outliers), phân tích tương quan (Correlation matrix), kiểm tra Data Leakage tiềm ẩn. Trực quan hóa bằng biểu đồ.
- **Output**: Data Profile (Thống kê mô tả toàn diện).
- **Insight**: Hiểu chất lượng dữ liệu hiện tại, phân phối của target variable và đưa ra giả thuyết ban đầu.
- **Decision**: Xác định các kỹ thuật làm sạch dữ liệu cần áp dụng ở Bước 2.
- **Artifact**: Cập nhật file code `eda.py`, lưu biểu đồ vào thư mục `Figures/`, viết báo cáo vào `README_EDA.md`.

### Bước 2 — Data Cleaning
- **Goal**: Làm sạch và chuẩn hóa dữ liệu để đảm bảo độ tin cậy ("Garbage In, Garbage Out").
- **Input**: Raw dataset + Insight từ Bước 1.
- **Tasks**: Điền hoặc xóa Missing values, xử lý Duplicate, xử lý Outlier (capping/removal), chuẩn hóa kiểu dữ liệu (Datatype casting).
- **Output**: Clean Dataset.
- **Insight**: Dữ liệu đã đủ sạch và loại bỏ nhiễu để tiến hành mô hình hóa chưa?
- **Decision**: Chuyển sang Feature Engineering nếu dữ liệu đã đảm bảo chất lượng.
- **Artifact**: File dữ liệu sạch đã lưu, code cleaning trong `eda.py`.

### Bước 3 — Feature Engineering
- **Goal**: Tạo, chọn lọc và biến đổi đặc trưng để giúp mô hình học tốt hơn.
- **Input**: Clean Dataset.
- **Tasks**: Encoding (cho categorical), Scaling/Normalization, tạo Polynomial features (rất quan trọng cho project này), Feature creation (từ domain knowledge), Feature selection.
- **Output**: Feature Set (Ma trận X, vector y chuẩn).
- **Insight**: Feature nào mang lại nhiều thông tin nhất cho target? Feature nào gây nhiễu?
- **Decision**: Chốt danh sách feature tốt nhất để đưa vào huấn luyện.
- **Artifact**: Code feature engineering rõ ràng, logic được document lại.

### Bước 4 — Data Split
- **Goal**: Phân chia tập dữ liệu để huấn luyện và đánh giá công bằng.
- **Input**: Feature Set.
- **Tasks**: Phân chia tập Train / Validation / Test. Đảm bảo phân phối (Stratified nếu cần) và ngăn chặn Data Leakage (chỉ fit Scaler trên Train).
- **Output**: Dataset Split (Train/Val/Test).
- **Insight**: Việc phân chia có giữ nguyên cấu trúc phân phối của tập dữ liệu gốc không?
- **Decision**: Xác nhận các tập dữ liệu để bắt đầu modeling.
- **Artifact**: Code chia dữ liệu an toàn trong `train.py`.

### Bước 5 — Baseline Model
- **Goal**: Xây dựng mốc so sánh (benchmark) tối thiểu.
- **Input**: Dataset Split (Train/Val).
- **Tasks**: Khởi tạo và huấn luyện mô hình đơn giản nhất có thể (VD: Linear Regression không có polynomial).
- **Output**: Baseline Metrics.
- **Insight**: Một mô hình cực kỳ đơn giản đạt được hiệu suất bao nhiêu?
- **Decision**: Lấy kết quả này làm tiêu chuẩn tối thiểu. Mọi mô hình phức tạp hơn phải đánh bại baseline này.
- **Artifact**: Ghi nhận kết quả baseline vào `Results/` hoặc `README_TRAIN.md`.

### Bước 6 — Model Selection
- **Goal**: Đánh giá nhanh và chọn ra thuật toán/mô hình tiềm năng.
- **Input**: Dataset Split.
- **Tasks**: Thử nghiệm và đánh giá nhanh các thuật toán (VD: Polynomial Regression bậc 2, bậc 3, Ridge, Lasso...).
- **Output**: Candidate Models.
- **Insight**: Họ thuật toán hoặc bậc đa thức nào đang tỏ ra phù hợp nhất với tính chất của bài toán?
- **Decision**: Lựa chọn 1-2 mô hình tiềm năng nhất để đem đi tinh chỉnh chi tiết.
- **Artifact**: Bảng so sánh nhanh các mô hình ban đầu.

### Bước 7 — Hyperparameter Tuning
- **Goal**: Tìm cấu hình tham số tối ưu nhất cho candidate model.
- **Input**: Candidate Models, Validation Set.
- **Tasks**: Thực hiện Grid Search, Random Search trên Validation set (VD: tune alpha cho Ridge/Lasso, độ sâu cho cây, hoặc số bậc).
- **Output**: Best Hyperparameters.
- **Insight**: Tham số nào đang kiểm soát Overfitting/Underfitting tốt nhất?
- **Decision**: Chốt bộ tham số tối ưu để huấn luyện mô hình cuối cùng.
- **Artifact**: Log quá trình tuning, file lưu best parameters.

### Bước 8 — Train Model
- **Goal**: Huấn luyện mô hình cuối cùng.
- **Input**: Training set, Best Hyperparameters.
- **Tasks**: Huấn luyện (fit) mô hình với cấu hình tốt nhất.
- **Output**: Trained Model.
- **Insight**: Thời gian huấn luyện và sự hội tụ của mô hình.
- **Decision**: Mô hình đã sẵn sàng để chuyển sang bước đánh giá chưa?
- **Artifact**: Code train chi tiết trong `train.py`, file model nếu cần export.

### Bước 9 — Evaluation Metrics
- **Goal**: Đo lường hiệu năng của mô hình một cách toàn diện.
- **Input**: Trained Model, Test Set.
- **Tasks**: Dự đoán trên Test set và tính các metric phù hợp (MSE, RMSE, MAE, R²). Vẽ biểu đồ Dự đoán vs Thực tế.
- **Output**: Metrics Report.
- **Insight**: Điểm mạnh và điểm yếu cụ thể của mô hình (Underfitting hay Overfitting).
- **Decision**: Hiệu suất có đạt yêu cầu bài toán không?
- **Artifact**: Biểu đồ đánh giá trong `Figures/`, cập nhật `README_TRAIN.md`.

### Bước 10 — Cross Validation
- **Goal**: Kiểm tra tính ổn định và khả năng tổng quát hóa (Generalization).
- **Input**: Feature Set, Trained Model architecture.
- **Tasks**: Thực hiện K-Fold Cross Validation. Tính Mean và Standard Deviation của các metric.
- **Output**: Mean ± Std của Metrics.
- **Insight**: Mô hình có ổn định khi tập huấn luyện thay đổi không? Độ lệch chuẩn lớn báo hiệu điều gì?
- **Decision**: Đảm bảo mô hình đủ vững vàng trước dữ liệu chưa từng thấy.
- **Artifact**: Báo cáo kết quả Cross Validation.

### Bước 11 — Experiment Management
- **Goal**: Theo dõi, quản lý và đảm bảo tính tái lập (Reproducibility) của thí nghiệm.
- **Input**: Quá trình từ Bước 1 đến 10.
- **Tasks**: Lưu vết đầy đủ: version dữ liệu, các feature sử dụng, hyperparameter, random seed, và kết quả evaluation.
- **Output**: Experiment History.
- **Insight**: Thử nghiệm nào đột phá nhất? Tính chất nào dẫn đến sự đột phá đó?
- **Decision**: Lựa chọn phiên bản tốt nhất làm kết quả chính thức.
- **Artifact**: Log file lưu trong thư mục `Results/`.

### Bước 12 — Statistical Validation
- **Goal**: Kiểm chứng mức độ tin cậy bằng thống kê.
- **Input**: Kết quả của Baseline và Best Model.
- **Tasks**: Xác thực ý nghĩa thống kê của các cải tiến (Ví dụ: Sự chênh lệch RMSE giữa Polynomial bậc 2 và Linear Regression có thực sự mang ý nghĩa không, hay do ngẫu nhiên?).
- **Output**: Statistical Report.
- **Insight**: Việc làm phức tạp mô hình có thực sự đem lại giá trị đáng tin cậy không?
- **Decision**: Chấp nhận mô hình phức tạp hay quay về mô hình đơn giản.
- **Artifact**: Báo cáo kiểm định thống kê đơn giản.

### Bước 13 — Error Analysis
- **Goal**: Phân tích lỗi để thấu hiểu ranh giới của mô hình.
- **Input**: Predictions, Ground Truth (trên Test/Validation).
- **Tasks**: Tính Residuals. Phân tích sâu các điểm dữ liệu mà mô hình dự đoán sai nhiều nhất.
- **Output**: Error Categories.
- **Insight**: Mô hình thường sai ở những khoảng dữ liệu nào? Có pattern nào không? Lỗi do dữ liệu nhiễu hay do mô hình yếu?
- **Decision**: Quyết định xem có cần quay lại Bước Feature Engineering hoặc Data Cleaning để vá lỗi không.
- **Artifact**: Biểu đồ phân tích Residuals, báo cáo Error Analysis.

### Bước 14 — Model Interpretability
- **Goal**: Mở hộp đen, giải thích cách mô hình đưa ra dự đoán.
- **Input**: Trained Model, Feature Set.
- **Tasks**: Đánh giá mức độ quan trọng của các đặc trưng (Feature Importance), dùng SHAP hoặc LIME nếu mô hình phức tạp.
- **Output**: Explainability Report + Best Model.
- **Insight**: Đặc trưng nào đang chi phối dự đoán? Nó có hợp lý về mặt nghiệp vụ/thực tế không?
- **Decision**: Hoàn toàn tin tưởng vào mô hình và chốt pipeline để sẵn sàng sử dụng.
- **Artifact**: Biểu đồ Feature Importance, tổng kết dự án.

---

## Quy tắc thực hiện

### Controlled Experiment
- **BẮT BUỘC**: Chỉ thay đổi 1 biến mỗi experiment (Ví dụ: Thử bậc đa thức khác thì giữ nguyên Scaler).
- Log đầy đủ: dataset version, feature set, model config, metrics
- Mọi thay đổi phải có so sánh trước/sau rõ ràng.

### Gate Checking (Điều kiện chuyển bước)
- Không chuyển sang bước tiếp theo nếu artifact của bước hiện tại chưa hoàn thành hoặc insight chưa được làm rõ.
- Mỗi bước phải có Decision rõ ràng: tiếp tục / quay lại / điều chỉnh.
- Nếu kết quả không đạt → quay lại bước trước để xử lý nguyên nhân gốc rễ, KHÔNG nhảy cóc.

### Experiment Logging
- Mọi experiment phải được log với:
  - Timestamp
  - Thay đổi gì (1 biến duy nhất)
  - Metrics trước/sau
  - Quyết định (giữ / bỏ / thử tiếp)
