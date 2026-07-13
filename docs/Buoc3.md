# Feature Engineering Report (Phase 3)

## 1. Goal
Tạo, chọn lọc và biến đổi đặc trưng để biểu diễn dữ liệu tốt hơn (Feature Space), giúp mô hình dễ học hơn.

## 2. Input
- Dataset sạch: `clean_auto_mpg.csv`

## 3. Tasks Performed & Visual Insights

### Biến đổi Biến phân loại (Categorical Encoding)
- Sử dụng **One-Hot Encoding** cho `origin` (1: USA, 2: Europe, 3: Asia).
- *Insight*: Nếu giữ nguyên [1, 2, 3], mô hình sẽ hiểu nhầm Asia > Europe > USA (theo độ lớn toán học). One-Hot phá bỏ định kiến tuyến tính này, tạo ra 3 vector độc lập không thứ bậc.

### Tạo Đặc trưng Mới (Domain Feature Creation)
Dựa vào kiến thức ngành xe hơi, mình tạo ra đặc trưng kết hợp (interaction feature):
- `weight_per_hp = weight / horsepower` (Tỉ lệ trọng lượng trên mã lực)
- `displacement_per_cylinder = displacement / cylinders`

![New Features vs Target](./Figures/FE/new_features.png)

**Insight**: 
- Nhìn vào biểu đồ trên, `weight_per_hp` có mối tương quan rất chặt chẽ và thuận chiều với `mpg` (xe có tỉ lệ tải trọng/mã lực càng cao thì càng đỡ hao xăng). Việc tạo feature này gom nhóm thông tin đa cộng tuyến từ `weight` và `horsepower` lại, cung cấp một lăng kính mới sắc nét hơn cho mô hình.

### Scaling & Polynomial Features (Quan trọng)
- Đã xem xét nhưng QUYẾT ĐỊNH **CHƯA THỰC HIỆN** ở file này.
- *Insight*: Nếu scale (chuẩn hóa) toàn bộ dữ liệu lúc này, mean và variance của tập Test sẽ trộn lẫn vào tập Train, gây ra **Data Leakage**. Polynomial Features cũng nên được dời sang bước lập mô hình (`Pipeline`) để dễ dàng dò tìm bậc (degree) tốt nhất bằng GridSearch mà không cần tạo lại dataset.

## 4. Output
- Kích thước Feature Set (X): 398 mẫu, 11 đặc trưng.
- Đã lưu thành `X_features.csv` và `y_target.csv`.

## 5. Decision
- Không còn thao tác feature engineering nào cần làm trước Data Split. 
- Chuyển sang **Bước 4 — Data Split** để phân chia Train/Val/Test một cách an toàn.
