"""
TRAIN — Polynomial Regression (viết tay, Gradient Descent) cho Auto MPG.

Dựa trên kết luận EDA (xem README_EDA.md):
    - Feature tốt nhất: weight, horsepower, displacement (quan hệ CONG với mpg).
    - Polynomial BẬC 2 tối ưu (R^2 tăng mạnh ở bậc 2, chững ở bậc 3).
    - 'acceleration' tương quan yếu -> loại.

Quy trình:
    1. Load & làm sạch dữ liệu (bỏ 6 dòng horsepower thiếu)
    2. Sinh đặc trưng đa thức bậc d (gồm bình phương + tương tác)
    3. Chuẩn hóa z-score + thêm bias, chia train/test
    4. Huấn luyện bằng Gradient Descent (theo dõi cost MSE)
    5. Đánh giá R^2 / RMSE / MAE, so sánh bậc 1 vs bậc 2
    6. Lưu weights, metrics, cost history + vẽ hình

Output: Results/*.{txt,csv} + Figures/Train/*.png
Chạy:   python3 train.py
"""

import os
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "auto-mpg.data")
FIG_DIR = os.path.join(BASE_DIR, "Figures", "Train")
RESULT_DIR = os.path.join(BASE_DIR, "Results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

COLS = ["mpg", "cylinders", "displacement", "horsepower", "weight",
        "acceleration", "model_year", "origin", "car_name"]
FEATURES = ["weight", "horsepower", "displacement"]  # chọn theo EDA
TARGET = "mpg"

# Siêu tham số Gradient Descent
DEGREE = 2
LR = 0.1
N_ITERS = 5000
TEST_RATIO = 0.2
SEED = 42

REPORT = []


def log(msg=""):
    print(msg)
    REPORT.append(str(msg))


# ---------------------------------------------------------------------------
# Dữ liệu & đặc trưng
# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_FILE, sep=r"\s+", names=COLS,
                     quotechar='"', na_values="?")
    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    X = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)
    return X, y


def poly_features(X, degree):
    """Sinh đặc trưng đa thức tới 'degree' (gồm bình phương & tương tác).

    Ví dụ d=3 feature, bậc 2 -> [x1,x2,x3, x1^2,x1x2,x1x3,x2^2,x2x3,x3^2].
    KHÔNG kèm cột bias ở đây (bias thêm sau khi chuẩn hóa).
    """
    n, d = X.shape
    cols = []
    for deg in range(1, degree + 1):
        for combo in combinations_with_replacement(range(d), deg):
            term = np.ones(n)
            for idx in combo:
                term = term * X[:, idx]
            cols.append(term)
    return np.column_stack(cols)


def train_test_split(X, y, ratio, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = int(len(y) * ratio)
    te, tr = idx[:n_test], idx[n_test:]
    return X[tr], X[te], y[tr], y[te]


def standardize(train, test):
    """Chuẩn hóa z-score theo thống kê của TẬP TRAIN (tránh rò rỉ dữ liệu)."""
    mu = train.mean(axis=0)
    sigma = train.std(axis=0)
    sigma[sigma == 0] = 1.0
    return (train - mu) / sigma, (test - mu) / sigma, mu, sigma


def add_bias(X):
    return np.column_stack([np.ones(len(X)), X])


# ---------------------------------------------------------------------------
# Mô hình
# ---------------------------------------------------------------------------
def gradient_descent(X, y, lr, n_iters):
    """Hồi quy tuyến tính trên ma trận đặc trưng X (đã có cột bias)."""
    n, m = X.shape
    w = np.zeros(m)
    history = []
    for _ in range(n_iters):
        err = X @ w - y
        cost = (err @ err) / (2 * n)  # MSE/2
        history.append(cost)
        grad = (X.T @ err) / n
        w = w - lr * grad
    return w, np.array(history)


def metrics(y_true, y_pred):
    err = y_true - y_pred
    ss_res = (err @ err)
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt((err @ err) / len(y_true))
    mae = np.abs(err).mean()
    return r2, rmse, mae


def fit_and_eval(X_raw, y, degree, label):
    """Sinh poly bậc 'degree' -> chuẩn hóa -> GD -> đánh giá train/test."""
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(X_raw, y, TEST_RATIO, SEED)
    Ptr = poly_features(Xtr_raw, degree)
    Pte = poly_features(Xte_raw, degree)
    Ptr, Pte, _, _ = standardize(Ptr, Pte)
    Ptr, Pte = add_bias(Ptr), add_bias(Pte)

    w, hist = gradient_descent(Ptr, ytr, LR, N_ITERS)
    tr = metrics(ytr, Ptr @ w)
    te = metrics(yte, Pte @ w)

    log(f"[{label}] (bậc {degree}, {Ptr.shape[1]} hệ số gồm bias)")
    log(f"  Train : R2={tr[0]:.3f} | RMSE={tr[1]:.2f} | MAE={tr[2]:.2f}")
    log(f"  Test  : R2={te[0]:.3f} | RMSE={te[1]:.2f} | MAE={te[2]:.2f}")
    return {"w": w, "hist": hist, "train": tr, "test": te,
            "Xte_raw": Xte_raw, "yte": yte, "Pte": Pte, "degree": degree}


# ---------------------------------------------------------------------------
# Hình
# ---------------------------------------------------------------------------
def plot_cost(hist):
    plt.figure(figsize=(8, 5))
    plt.plot(hist, color="#4C72B0", lw=2)
    plt.xlabel("Vòng lặp")
    plt.ylabel("Cost (MSE/2)")
    plt.title(f"Đường hội tụ Gradient Descent (lr={LR}, {N_ITERS} vòng)")
    plt.yscale("log")
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "cost_curve.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Đã lưu: {out}")


def plot_pred_vs_actual(res):
    yte = res["yte"]
    yhat = res["Pte"] @ res["w"]
    lim = [min(yte.min(), yhat.min()) - 2, max(yte.max(), yhat.max()) + 2]
    plt.figure(figsize=(6.5, 6.5))
    plt.scatter(yte, yhat, s=25, alpha=0.6, color="#55A868", edgecolor="white")
    plt.plot(lim, lim, "--", color="crimson", lw=2, label="y = ŷ (lý tưởng)")
    plt.xlim(lim)
    plt.ylim(lim)
    plt.xlabel("mpg thực tế")
    plt.ylabel("mpg dự đoán")
    plt.title(f"Dự đoán vs Thực tế (test) — R²={res['test'][0]:.3f}")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "pred_vs_actual.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Đã lưu: {out}")


# ---------------------------------------------------------------------------
# Lưu kết quả
# ---------------------------------------------------------------------------
def save_outputs(res):
    np.savetxt(os.path.join(RESULT_DIR, "weights.txt"), res["w"],
               fmt="%.6f", header="Trọng số Polynomial bậc 2 (gồm bias đầu tiên)")
    pd.DataFrame({"iter": np.arange(len(res["hist"])), "cost": res["hist"]}) \
        .to_csv(os.path.join(RESULT_DIR, "gd_cost_history.csv"), index=False)
    with open(os.path.join(RESULT_DIR, "train_metrics.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(REPORT) + "\n")
    print(f"Đã lưu: {os.path.join(RESULT_DIR, 'weights.txt')}")
    print(f"Đã lưu: {os.path.join(RESULT_DIR, 'gd_cost_history.csv')}")
    print(f"Đã lưu: {os.path.join(RESULT_DIR, 'train_metrics.txt')}")


def main():
    X, y = load_data()
    log("=" * 70)
    log("TRAIN — POLYNOMIAL REGRESSION (Gradient Descent)")
    log("=" * 70)
    log(f"Feature: {FEATURES} | n={len(y)} mẫu | test={TEST_RATIO:.0%}")
    log("")

    # So sánh bậc 1 (tuyến tính) vs bậc 2 (đa thức)
    lin = fit_and_eval(X, y, 1, "Linear")
    log("")
    poly = fit_and_eval(X, y, DEGREE, "Polynomial")
    log("")
    gain = poly["test"][0] - lin["test"][0]
    log(f"=> Bậc 2 cải thiện R² test: {lin['test'][0]:.3f} -> "
        f"{poly['test'][0]:.3f} (+{gain:.3f})")

    plot_cost(poly["hist"])
    plot_pred_vs_actual(poly)
    save_outputs(poly)
    print("\nHoàn tất huấn luyện.")


if __name__ == "__main__":
    main()
