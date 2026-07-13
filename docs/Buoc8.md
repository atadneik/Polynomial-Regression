# Final Model Training Report (Phase 8)

## 1. Goal
Huấn luyện và đóng gói (export) mô hình hoàn chỉnh với cấu hình tối ưu nhất đã tìm được ở Bước 7, sẵn sàng cho việc đánh giá và triển khai.

## 2. Input
- **Dataset**: Tập `X_train_full` (Gộp từ Train và Validation, tổng cộng 338 mẫu).
- **Hyperparameters**: Load từ `Results/best_params.json`:
  - `model`: Ridge
  - `degree`: 2
  - `alpha`: 0.29836472402833375

## 3. Tasks Performed & Insights

- **Chiến thuật Gộp Dữ Liệu (Merge Data):** Thay vì chỉ train trên tập `X_train` (278 mẫu) như lúc dò tham số, mình đã gộp thêm tập `X_val` (60 mẫu) vào để tạo thành `X_train_full` (338 mẫu). 
  - *Insight*: Khi cấu trúc mô hình đã được chốt, tập Validation không còn tác dụng để tune nữa. Việc gộp nó vào tập Train giúp mô hình học được nhiều dữ liệu hơn (tăng 21% dung lượng data), từ đó củng cố độ vững vàng (robustness) trước khi chạm trán tập Test.
- **Theo dõi sự hội tụ (Convergence & Time):**
  - Khởi tạo `Pipeline` với toàn bộ cấu trúc: Sinh đa thức bậc 2 $\rightarrow$ Chuẩn hóa Z-score $\rightarrow$ Hồi quy Ridge.
  - *Insight*: Thời gian huấn luyện (fit) toàn bộ Pipeline chỉ mất **16.61 mili-giây (ms)**. Ridge Regression là thuật toán dạng Closed-form solution (hoặc dùng Cholesky solver cực nhanh) nên sự hội tụ diễn ra gần như ngay lập tức, không tốn tài nguyên tính toán như các mô hình Neural Network. Rất lý tưởng để triển khai thực tế.

## 4. Output
- Mô hình đã được huấn luyện thành công.
- Export mô hình nguyên khối dưới định dạng Joblib: `models/final_polynomial_ridge_model.joblib`

## 5. Decision
- Mô hình đã sẵn sàng, hội tụ cực tốt và đã được đóng gói an toàn.
- Chuyển sang "Chiến trường thực sự": **Bước 9 — Evaluation Metrics** để mở niêm phong tập Test và đánh giá hiệu năng thật sự của mô hình dự án này!
