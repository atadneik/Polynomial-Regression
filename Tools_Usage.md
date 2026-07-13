# CÁC CÔNG CỤ VÀ THƯ VIỆN SỬ DỤNG TRONG DỰ ÁN

## 1. Giới thiệu chung
Trong phạm vi của dự án nghiên cứu và triển khai mô hình học máy bằng phương pháp Hồi quy Đa thức (Polynomial Regression), việc lựa chọn và ứng dụng các công cụ, thư viện phần mềm đóng vai trò thiết yếu. Quá trình này không chỉ đảm bảo tính chính xác trong việc tính toán mà còn tối ưu hóa hiệu năng thực thi của mô hình. Toàn bộ mã nguồn của dự án được phát triển dựa trên nền tảng ngôn ngữ lập trình Python, kết hợp với các thư viện mã nguồn mở hàng đầu trong lĩnh vực Khoa học Dữ liệu (Data Science) và Học máy (Machine Learning).

## 2. Tổng hợp công cụ và thư viện
Nhằm minh bạch hóa quá trình thực nghiệm, Bảng 1 dưới đây trình bày chi tiết danh sách các công cụ, phân loại, chức năng cụ thể và các thành phần cốt lõi được ứng dụng trong dự án.

**Bảng 1: Tổng hợp các công cụ và thư viện sử dụng trong dự án**

| Công cụ / Thư viện | Phân loại | Chức năng trong dự án | Các Module / Lớp thường dùng |
| :--- | :--- | :--- | :--- |
| **Python** | Ngôn ngữ lập trình | Đóng vai trò là ngôn ngữ nền tảng để phát triển mã nguồn, kịch bản xử lý dữ liệu và xây dựng toàn bộ hệ thống pipeline của mô hình. | Lõi ngôn ngữ Python |
| **Pandas** | Xử lý và thao tác dữ liệu | Hỗ trợ nạp dữ liệu từ các tệp thô, thực hiện các nghiệp vụ làm sạch dữ liệu (xử lý giá trị khuyết thiếu, loại bỏ ngoại lệ), thao tác và trích xuất đặc trưng trên dữ liệu dạng bảng. | `pd.read_csv`, `pd.DataFrame`, `pd.Series` |
| **NumPy** | Tính toán khoa học | Hỗ trợ các phép toán đại số tuyến tính, biến đổi ma trận và xử lý mảng đa chiều với tốc độ và hiệu năng cao. | `np.array`, `np.sqrt`, `np.log` |
| **Matplotlib** | Trực quan hóa dữ liệu | Đảm nhiệm việc xây dựng các biểu đồ tĩnh cơ bản (biểu đồ phân tán, biểu đồ tần suất), phục vụ quá trình xuất ảnh báo cáo đánh giá. | `matplotlib.pyplot` |
| **Seaborn** | Trực quan hóa thống kê | Mở rộng khả năng trực quan hóa với các biểu đồ phức tạp và trực quan hơn như ma trận tương quan (Heatmap) hay biểu đồ hộp (Boxplot). | `sns.heatmap`, `sns.boxplot`, `sns.histplot` |
| **Scikit-Learn (sklearn)**| Học máy (Machine Learning) | Thư viện cốt lõi cung cấp các thuật toán học máy, các kỹ thuật tiền xử lý dữ liệu và công cụ đánh giá hiệu năng mô hình. | *(Chi tiết bên dưới)* |
| - *sklearn.linear_model* | Mô hình hóa | Khởi tạo các thuật toán Hồi quy tuyến tính, Hồi quy Ridge, và Hồi quy Lasso. | `LinearRegression`, `Ridge`, `Lasso` |
| - *sklearn.preprocessing* | Tiền xử lý dữ liệu | Thực hiện chuẩn hóa dữ liệu (Z-score normalization) và tự động tạo các đặc trưng đa thức bậc cao. | `StandardScaler`, `PolynomialFeatures` |
| - *sklearn.pipeline* | Tự động hóa quy trình | Đóng gói và liên kết các bước tiền xử lý và huấn luyện thành một chu trình (pipeline) thống nhất, giúp hạn chế hiện tượng rò rỉ dữ liệu (data leakage). | `Pipeline` |
| - *sklearn.model_selection*| Lựa chọn và xác thực | Phân chia tập dữ liệu (Train/Val/Test), tìm kiếm siêu tham số và thực hiện Kiểm chứng chéo K-Fold (K-Fold Cross Validation). | `train_test_split`, `KFold`, `cross_val_score` |
| - *sklearn.metrics* | Đánh giá mô hình | Cung cấp các độ đo sai số chuẩn mực để kiểm định độ chính xác của mô hình dự đoán. | `mean_squared_error`, `mean_absolute_error`, `r2_score` |
| **SciPy** | Thống kê và Toán học | Hỗ trợ tính toán phân phối xác suất và tiến hành các phép kiểm định thống kê (như T-test) nhằm đánh giá ý nghĩa của các cải tiến trên mô hình. | `scipy.stats` |
| **Joblib** | Lưu trữ mô hình (Serialization)| Chuyển đổi và lưu trữ các pipeline, mô hình tốt nhất thành tệp tin nhị phân phục vụ cho việc tái sử dụng và dự đoán sau này. | `joblib.dump`, `joblib.load` |
| **OS / JSON / Time** | Tiện ích hệ thống | Xử lý các tác vụ liên quan đến hệ điều hành, cấu trúc thư mục, ghi nhận lịch sử thực nghiệm dạng JSON và đo lường thời gian huấn luyện. | `os.makedirs`, `json.dump`, `time.time` |

## 3. Tổng kết
Việc tích hợp và vận dụng linh hoạt các thư viện trên đã góp phần chuẩn hóa quy trình phát triển và kiểm định mô hình, từ giai đoạn phân tích dữ liệu khám phá (EDA) cho đến giai đoạn giải thích mô hình (Model Interpretability). Tất cả các thư viện này đều được quản lý thông qua hệ thống quản lý gói `pip` và được đóng gói chặt chẽ trong một môi trường ảo (Virtual Environment), đảm bảo tính độc lập và khả năng tái lập (reproducibility) cao cho dự án nghiên cứu.
