"""
Áp dụng Polynomial Regression (from scratch) lên dataset Combined Cycle
Power Plant (CCPP) — tải từ UCI, lưu sẵn tại data/ccpp.csv.

Bài toán: dự đoán PE (công suất điện đầu ra, MW) từ AT (nhiệt độ môi trường).
Quan hệ AT -> PE cong mượt, ít nhiễu: nhiệt độ càng cao thì công suất càng
giảm (turbine kém hiệu quả khi không khí nóng).

Chạy:  python3 train_ccpp.py
Kết quả: in metric + lưu đồ thị PNG vào Figures/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # backend không cần màn hình, chỉ lưu file
import matplotlib.pyplot as plt
import seaborn as sns

from polynomial_regression import (
    build_design_matrix,
    normal_equation,
    gradient_descent,
    predict,
    mse,
    r2_score,
    StandardScaler1D,
)

sns.set_theme(style="whitegrid", context="notebook", palette="deep")

RNG = np.random.default_rng(42)
BASE_DIR = os.path.dirname(__file__)
FIG_DIR = os.path.join(BASE_DIR, "Figures", "Train")
RESULT_DIR = os.path.join(BASE_DIR, "Results")
DATA_CSV = os.path.join(BASE_DIR, "data", "ccpp.csv")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


def load_data():
    """Đọc CCPP từ CSV, lấy AT (input) và PE (target)."""
    df = pd.read_csv(DATA_CSV)
    x = df["AT"].to_numpy(dtype=float)
    y = df["PE"].to_numpy(dtype=float)
    return x, y


def train_val_test_split(x, y, val_size=0.2, test_size=0.2):
    """Chia 3 phần train/val/test.

    - train: huấn luyện (tìm w cho từng bậc)
    - val:   chọn siêu tham số (ở đây là bậc đa thức) — KHÔNG đụng test
    - test:  chỉ dùng MỘT lần ở cuối để báo cáo, mô phỏng dữ liệu thật chưa thấy
    """
    idx = RNG.permutation(len(x))
    n_test = int(len(x) * test_size)
    n_val = int(len(x) * val_size)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    return (
        x[train_idx], x[val_idx], x[test_idx],
        y[train_idx], y[val_idx], y[test_idx],
    )


def evaluate(degree, x_tr_s, y_tr, x_eval_s, y_eval):
    """Train trên (x_tr, y_tr), chấm điểm trên (x_eval, y_eval).

    Trả về (w, dict metric) với metric gồm MSE, RMSE, R2 trên tập eval.
    """
    X_tr = build_design_matrix(x_tr_s, degree)
    X_ev = build_design_matrix(x_eval_s, degree)
    w = normal_equation(X_tr, y_tr)
    y_hat = predict(X_ev, w)
    m = mse(y_eval, y_hat)
    return w, {"mse": m, "rmse": m ** 0.5, "r2": r2_score(y_eval, y_hat)}


def main():
    x, y = load_data()
    print(f"Dataset CCPP: {len(x)} mẫu | AT (nhiệt độ) -> PE (công suất MW)")
    x_tr, x_val, x_te, y_tr, y_val, y_te = train_val_test_split(x, y)
    print(f"Chia dữ liệu: train={len(x_tr)} | val={len(x_val)} | "
          f"test={len(x_te)}  (60/20/20)")

    # Scaling: học trên train, áp lên val + test (chống data leakage)
    scaler = StandardScaler1D().fit(x_tr)
    x_tr_s = scaler.transform(x_tr)
    x_val_s = scaler.transform(x_val)
    x_te_s = scaler.transform(x_te)

    # 1) So sánh các bậc trên VALIDATION để thấy bias-variance tradeoff.
    #    Test được giữ kín, KHÔNG dùng để chọn bậc.
    print("=" * 70)
    print(f"{'Bậc':>4} | {'R2 train':>9} | {'R2 val':>9} | "
          f"{'RMSE val':>9} | {'MSE val':>9}")
    print("-" * 70)
    results = {}
    for degree in [1, 2, 3, 4, 5, 8, 12]:
        w, m_tr = evaluate(degree, x_tr_s, y_tr, x_tr_s, y_tr)
        _, m_val = evaluate(degree, x_tr_s, y_tr, x_val_s, y_val)
        results[degree] = (w, m_tr, m_val)
        print(f"{degree:>4} | {m_tr['r2']:>9.4f} | {m_val['r2']:>9.4f} | "
              f"{m_val['rmse']:>9.3f} | {m_val['mse']:>9.3f}")
    print("=" * 70)

    # Chọn bậc theo R2 trên VALIDATION (đúng quy trình chọn siêu tham số)
    best_degree = max(results, key=lambda d: results[d][2]["r2"])
    print(f"\nBậc tốt nhất trên VALIDATION (R2 cao nhất): {best_degree}")

    # 2) Báo cáo cuối: train MÔ HÌNH bậc đã chọn, chấm trên TEST (chưa từng thấy)
    _, m_test = evaluate(best_degree, x_tr_s, y_tr, x_te_s, y_te)
    _, m_tr_best = evaluate(best_degree, x_tr_s, y_tr, x_tr_s, y_tr)
    m_val_best = results[best_degree][2]
    print("\n" + "=" * 70)
    print(f"KẾT QUẢ CUỐI — bậc {best_degree}")
    print("-" * 70)
    print(f"{'Tập':>6} | {'R2':>9} | {'RMSE':>9} | {'MSE':>9}")
    for name, mt in [("train", m_tr_best), ("val", m_val_best),
                     ("test", m_test)]:
        print(f"{name:>6} | {mt['r2']:>9.4f} | {mt['rmse']:>9.3f} | "
              f"{mt['mse']:>9.3f}")
    print("=" * 70)

    save_results(results, best_degree, m_tr_best, m_val_best, m_test,
                 n_tr=len(x_tr), n_val=len(x_val), n_te=len(x_te))

    # 3) Đồ thị đường khớp cho vài bậc tiêu biểu
    plot_fits(x, y, scaler, results, degrees=[1, 2, 5, 12])

    # 4) Kiểm chứng: Normal Equation vs Gradient Descent cho cùng bậc
    compare_methods(x_tr_s, y_tr, degree=best_degree)


def save_results(results, best_degree, m_tr, m_val, m_te,
                 n_tr, n_val, n_te):
    """Ghi bảng metric ra Results/ccpp_metrics.txt + Results/ccpp_metrics.csv."""
    txt_path = os.path.join(RESULT_DIR, "ccpp_metrics.txt")
    csv_path = os.path.join(RESULT_DIR, "ccpp_metrics.csv")

    lines = []
    lines.append("Polynomial Regression — CCPP (AT -> PE)")
    lines.append(f"Chia dữ liệu: train={n_tr} | val={n_val} | test={n_te}")
    lines.append("")
    lines.append("So sánh các bậc trên VALIDATION:")
    lines.append(f"{'degree':>6} | {'R2 train':>9} | {'R2 val':>9} | "
                 f"{'RMSE val':>9} | {'MSE val':>9}")
    lines.append("-" * 60)
    for d, (_, mt, mv) in results.items():
        lines.append(f"{d:>6} | {mt['r2']:>9.4f} | {mv['r2']:>9.4f} | "
                     f"{mv['rmse']:>9.3f} | {mv['mse']:>9.3f}")
    lines.append("")
    lines.append(f"Bậc được chọn (theo R2 val): {best_degree}")
    lines.append("")
    lines.append(f"KẾT QUẢ CUỐI — bậc {best_degree}:")
    lines.append(f"{'split':>6} | {'R2':>9} | {'RMSE':>9} | {'MSE':>9}")
    for name, mt in [("train", m_tr), ("val", m_val), ("test", m_te)]:
        lines.append(f"{name:>6} | {mt['r2']:>9.4f} | {mt['rmse']:>9.3f} | "
                     f"{mt['mse']:>9.3f}")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # CSV: 1 dòng / 1 bậc, kèm dòng cuối cho test của best_degree
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("degree,split,r2,rmse,mse\n")
        for d, (_, mt, mv) in results.items():
            f.write(f"{d},train,{mt['r2']:.6f},{mt['rmse']:.6f},"
                    f"{mt['mse']:.6f}\n")
            f.write(f"{d},val,{mv['r2']:.6f},{mv['rmse']:.6f},"
                    f"{mv['mse']:.6f}\n")
        f.write(f"{best_degree},test,{m_te['r2']:.6f},{m_te['rmse']:.6f},"
                f"{m_te['mse']:.6f}\n")

    print(f"Đã lưu: {txt_path}")
    print(f"Đã lưu: {csv_path}")


def plot_fits(x, y, scaler, results, degrees):
    xx = np.linspace(x.min(), x.max(), 300)
    xx_s = scaler.transform(xx)

    plt.figure(figsize=(12, 7))
    sns.scatterplot(x=x, y=y, s=10, alpha=0.2, color="gray",
                    label="Dữ liệu thật", edgecolor=None)
    palette = sns.color_palette("tab10", n_colors=len(degrees))
    for color, degree in zip(palette, degrees):
        w = results[degree][0]
        yy = predict(build_design_matrix(xx_s, degree), w)
        sns.lineplot(x=xx, y=yy, linewidth=2.2, color=color,
                     label=f"degree = {degree}")
    # Giới hạn trục y theo khoảng dữ liệu thật, tránh bậc cao vọt biên (Runge)
    pad = (y.max() - y.min()) * 0.05
    plt.ylim(y.min() - pad, y.max() + pad)
    plt.xlabel("AT — Nhiệt độ môi trường (°C)")
    plt.ylabel("PE — Công suất điện (MW)")
    plt.title("Polynomial Regression — CCPP (đường khớp theo bậc)",
              fontsize=13)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "polyfit.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Đã lưu: {out}")


def compare_methods(x_tr_s, y_tr, degree):
    X_tr = build_design_matrix(x_tr_s, degree)
    w_ne = normal_equation(X_tr, y_tr)

    # Gradient Descent cần chuẩn hóa TỪNG cột (x, x^2, x^3...) vì dù x đã
    # scale, các luỹ thừa vẫn lệch thang đo nhau -> GD dễ phân kỳ.
    # Cột bias (cột 0) giữ nguyên = 1.
    col_mean = X_tr[:, 1:].mean(axis=0)
    col_std = X_tr[:, 1:].std(axis=0)
    X_gd = X_tr.copy()
    X_gd[:, 1:] = (X_tr[:, 1:] - col_mean) / col_std

    w_gd, cost_history = gradient_descent(X_gd, y_tr, lr=0.1, n_iters=8000)

    mse_ne = mse(y_tr, predict(X_tr, w_ne))
    mse_gd = mse(y_tr, predict(X_gd, w_gd))

    print(f"\nSo sánh w (bậc {degree}):")
    print("  Normal Equation (trên đặc trưng gốc):", np.round(w_ne, 4))
    print("  Gradient Descent (trên đặc trưng đã scale cột):",
          np.round(w_gd, 4))
    print(f"  MSE NE = {mse_ne:.4f} | MSE GD = {mse_gd:.4f}")

    # Lưu lịch sử cost của GD ra CSV (iter, cost) — dùng để vẽ lại / kiểm chứng
    hist_path = os.path.join(RESULT_DIR, "ccpp_gd_cost_history.csv")
    with open(hist_path, "w", encoding="utf-8") as f:
        f.write("iter,cost\n")
        for i, c in enumerate(cost_history, start=1):
            f.write(f"{i},{c:.6f}\n")
    print(f"Đã lưu: {hist_path}")

    # Lưu trọng số cuối + MSE để đối chiếu NE vs GD
    w_path = os.path.join(RESULT_DIR, "ccpp_weights.txt")
    with open(w_path, "w", encoding="utf-8") as f:
        f.write(f"Bậc đa thức: {degree}\n")
        f.write(f"Số vòng lặp GD: {len(cost_history)}\n")
        f.write(f"Learning rate: 0.1\n\n")
        f.write("Normal Equation (đặc trưng gốc):\n")
        f.write("  w = " + np.array2string(w_ne, precision=6) + "\n")
        f.write(f"  MSE train = {mse_ne:.6f}\n\n")
        f.write("Gradient Descent (đặc trưng đã scale từng cột):\n")
        f.write("  w = " + np.array2string(w_gd, precision=6) + "\n")
        f.write(f"  MSE train = {mse_gd:.6f}\n\n")
        f.write(f"Cost ban đầu (iter 1):  {cost_history[0]:.6f}\n")
        f.write(f"Cost cuối  (iter {len(cost_history)}): "
                f"{cost_history[-1]:.6f}\n")
    print(f"Đã lưu: {w_path}")

    plt.figure(figsize=(10, 6))
    iters = np.arange(1, len(cost_history) + 1)
    sns.lineplot(x=iters, y=cost_history, color="crimson", linewidth=2)
    plt.xscale("log")
    plt.yscale("log")  # log-log: thấy rõ cả pha giảm sốc lẫn pha tinh chỉnh
    plt.xlabel("Số vòng lặp (log)")
    plt.ylabel("Cost (MSE, log)")
    plt.title(f"Gradient Descent hội tụ — CCPP (bậc {degree})", fontsize=13)
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "cost_curve.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Đã lưu: {out}")


if __name__ == "__main__":
    main()
