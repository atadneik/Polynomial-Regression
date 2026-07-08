# Experiment Management Report (Phase 11)

## 0. Trả lời câu hỏi: Vì sao R² kỳ vọng chỉ dừng lại ở 87%?
Trước khi lưu vết thí nghiệm, đây là phân tích **Nguyên nhân sâu xa nhất (Root Causes)** giải thích vì sao mô hình không thể đạt được 95% hay 100% trên diện rộng (dù tập Test ngẫu nhiên có vọt lên 94%):

1. **Thiếu vắng Đặc trưng lõi (Omitted Variable Bias):** Tập dữ liệu của chúng ta (Auto MPG) chỉ có các thông số cơ bản của động cơ. Trong thực tế, mức tiêu thụ nhiên liệu (mpg) còn phụ thuộc cực kỳ lớn vào: **Hệ số cản gió (Aerodynamics), Loại hộp số (Manual vs Auto), Tỉ số truyền (Gear Ratios), Loại lốp xe, và Điều kiện thử nghiệm.** Vì ta KHÔNG CÓ những dữ liệu này, nó tạo thành một khoảng mù vĩnh viễn (Irreducible Error). Ta không thể dự đoán những gì ta không thể nhìn thấy!
2. **Nhiễu Đo lường Lịch sử (Measurement Noise):** Dữ liệu này được thu thập từ thập niên 70-80. Tiêu chuẩn đo lường mã lực (horsepower) thời đó không đồng nhất (thay đổi từ SAE gross sang SAE net). Sai số vật lý trong phòng thí nghiệm thời đó đã "ám" vào dữ liệu.
3. **Giới hạn của Thuật toán toàn cục:** Polynomial Regression áp dụng một đường cong toán học chung cho toàn bộ dữ liệu. Nó rất tuyệt, nhưng chưa đủ khả năng nắm bắt những tương tác cục bộ siêu phức tạp (ví dụ: công nghệ động cơ thay đổi đột ngột vào năm 1980 do khủng hoảng dầu mỏ).

> **Kết luận:** 87% là một con số **cực kỳ xuất sắc** cho giới hạn vật lý của tập dữ liệu này. Cố gắng ép mô hình lên 99% bằng cách tăng bậc đa thức sẽ chỉ dẫn đến Overfitting (học thuộc lòng cả lỗi đo lường).

---

## 1. Goal
Theo dõi, quản lý và đảm bảo tính tái lập (Reproducibility). Nếu 1 tháng sau ta quay lại, ta phải biết chính xác công thức nào đã tạo ra kết quả này.

## 2. Input
- Toàn bộ tham số, random seed, cấu trúc Pipeline, và Metrics thu được từ Bước 1 đến Bước 10.

## 3. Tasks Performed & Insights
- Đã gói toàn bộ metadata của dự án thành định dạng JSON.
- **Insight Đột phá nhất của Thí nghiệm (Breakthrough Insight):** Việc chuyển từ Linear (Bậc 1) sang Polynomial (Bậc 2) chính là điểm bùng nổ của dự án, giúp bẻ cong đường dự đoán để khớp với dữ liệu. Kết hợp với `Ridge` chính là chìa khóa để giữ mô hình không bị "ngáo" (Overfitting).

## 4. Output
- File lưu vết thí nghiệm: `Results/experiment_log.json` (Ghi nhận chính xác `random_seed=42`, `alpha=0.2984`, tính năng `weight_per_hp`, v.v.)

## 5. Decision
- Phiên bản mô hình hiện tại đã được chốt và đóng dấu bảo mật. Không ai có thể vô tình làm mất cấu hình này.
- Chuyển sang **Bước 12 — Statistical Validation** để làm một phép thử thống kê toán học xem sự cải tiến này có thực sự ý nghĩa không, hay chỉ là do ăn may!
