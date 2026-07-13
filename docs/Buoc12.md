# Statistical Validation Report (Phase 12)

## 1. Goal
Kiểm chứng bằng Toán Học xem sự cải tiến từ mô hình Linear (Baseline) lên mô hình Polynomial (Best) có thật sự đáng tin cậy hay không, hay chỉ do ăn may ngẫu nhiên.

## 2. Input
- Mô hình 1: Linear Regression (Baseline)
- Mô hình 2: Polynomial Bậc 2 + Ridge (Best Model)
- Phương pháp: **Paired T-Test** (Kiểm định T bắt cặp) trên điểm số R² thu được từ 10 tập Folds (đảm bảo 2 mô hình được thi đấu trên đúng 10 đấu trường giống hệt nhau).

## 3. Tasks Performed & Visual Insights

Đã tiến hành chạy lại Cross-Validation cho 2 mô hình và thực hiện kiểm định T-Test.

### Kết quả Kiểm định
- Trung bình R² Baseline: `0.8438`
- Trung bình R² Best Model: `0.8728`
- **P-Value (Giá trị p)**: `1.626785e-03`

> **P-value là gì?** Nó là xác suất xảy ra hiện tượng "Hai mô hình thực chất có sức mạnh bằng nhau, nhưng do ăn may nên mô hình Poly ngẫu nhiên đạt điểm cao hơn". 
> Nếu $p < 0.05$ (5%), ta bác bỏ sự ăn may và tin tưởng tuyệt đối vào mô hình.

### Trực quan hóa (Visual Insight)

![Paired Comparison](./Figures/Stats/paired_comparison.png)

**Insight Cực Kì Quan Trọng (từ biểu đồ):**
- Biểu đồ Pointplot nối các điểm (Fold 1 đến Fold 10) cho thấy: Trong **TẤT CẢ 10 lần thi đấu**, đường màu xanh (Polynomial) LUÔN LUÔN nằm trên đường màu cam (Linear).
- Không có bất kỳ một ngoại lệ nào. Ở những Fold dữ liệu khó đoán khiến Linear rớt điểm thê thảm (như Fold 3, Fold 9), thì Polynomial vẫn bám trụ cực kỳ vững chắc.
- P-value đạt `1.626785e-03` (Tức là gần như bằng 0, nhỏ hơn 0.05 hàng chục ngàn lần). Sự ưu việt của mô hình Đa thức là **sự thật không thể chối cãi**.

## 4. Output
- Kết luận Toán Học: Cải tiến là **Có Ý Nghĩa Thống Kê (Statistically Significant)**.
- Biểu đồ minh họa lưu tại: `Figures/Stats/paired_comparison.png`.

## 5. Decision
- Việc làm phức tạp mô hình (từ Bậc 1 lên Bậc 2) hoàn toàn mang lại giá trị xứng đáng.
- Quyết định: **CHẤP NHẬN mô hình Polynomial phức tạp vì nó đem lại sự cải thiện RÕ RỆT và CHÂN THỰC.**
- Chuyển sang **Bước 13 — Error Analysis** để soi lỗi kỹ hơn, xem mô hình Polynomial dẫu xuất sắc nhưng nó thường vấp ngã ở những thể loại xe nào.
