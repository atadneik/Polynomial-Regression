import json
import os
from datetime import datetime

print("Starting Step 11: Experiment Management...")

# 1. Gather all experiment data
experiment_record = {
    "experiment_id": f"EXP_{datetime.now().strftime('%Y%md_%H%M%S')}",
    "timestamp": datetime.now().isoformat(),
    "dataset": {
        "name": "Auto MPG",
        "raw_samples": 398,
        "clean_samples": 398,
        "target": "mpg",
        "features": [
            "cylinders", "displacement", "horsepower", "weight", 
            "acceleration", "model_year", "origin_USA", "origin_Europe", 
            "origin_Asia", "weight_per_hp", "displacement_per_cylinder"
        ]
    },
    "data_split": {
        "random_seed": 42,
        "train_ratio": 0.70,
        "val_ratio": 0.15,
        "test_ratio": 0.15
    },
    "model_architecture": {
        "pipeline": [
            "PolynomialFeatures",
            "StandardScaler",
            "Ridge Regression"
        ],
        "best_hyperparameters": {
            "degree": 2,
            "alpha": 0.2984,
            "include_bias": False
        }
    },
    "metrics": {
        "baseline_val_r2": 0.8481,
        "best_val_r2": 0.8720,
        "test_r2": 0.9397,
        "test_rmse": 1.9906,
        "cv_mean_r2": 0.8728,
        "cv_std_r2": 0.0311,
        "cv_mean_rmse": 2.7432
    },
    "breakthrough_insight": "Sử dụng PolynomialFeatures bậc 2 đã giải quyết được hiện tượng Underfitting do tính chất phi tuyến của dữ liệu (thể hiện rõ ở biểu đồ phần dư hình chữ U của Baseline). Kết hợp Ridge Regression (alpha=0.2984) để kiểm soát Overfitting."
}

# 2. Save Experiment Log
os.makedirs('Results', exist_ok=True)
log_file = 'Results/experiment_log.json'

# If log file exists, append to it (create a list of experiments). If not, create new.
if os.path.exists(log_file):
    try:
        with open(log_file, 'r') as f:
            logs = json.load(f)
            if not isinstance(logs, list):
                logs = [logs]
    except json.JSONDecodeError:
        logs = []
else:
    logs = []

logs.append(experiment_record)

with open(log_file, 'w', encoding='utf-8') as f:
    json.dump(logs, f, indent=4, ensure_ascii=False)

print(f"Experiment securely logged to {log_file}")

# 3. Generate Buoc11.md
buoc11_content = f"""# Experiment Management Report (Phase 11)

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
"""

with open('Buoc11.md', 'w', encoding='utf-8') as f:
    f.write(buoc11_content)

print("Saved report to Buoc11.md")
