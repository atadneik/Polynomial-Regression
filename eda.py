"""
EDA — Exploratory Data Analysis cho dataset Auto MPG (UCI).

Quy trình gọn 4 bước:
    1. Tổng quan & chất lượng dữ liệu -> info, describe, missing, outlier
    2. Phân phối biến                 -> Histogram target + feature, Countplot
    3. Quan hệ với target             -> Scatter linear-vs-poly2 + Heatmap
    4. Bằng chứng phi tuyến           -> Residual "chữ U" + R^2 theo bậc

Mục tiêu: chứng minh bài toán dự đoán mpg phù hợp với Polynomial Regression
(quan hệ cong, residual hồi quy bậc 1 uốn dạng chữ U, R^2 tăng mạnh ở bậc 2).

Output: Figures/EDA/*.png + Results/eda_summary.txt
Chạy:   python3 eda.py
"""

import os
import io
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # backend chỉ lưu file, không cần màn hình
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "Figures", "EDA")
RESULT_DIR = os.path.join(BASE_DIR, "Results")
DATA_FILE = os.path.join(BASE_DIR, "data", "auto-mpg.data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook", palette="deep")

COLS = ["mpg", "cylinders", "displacement", "horsepower", "weight",
        "acceleration", "model_year", "origin", "car_name"]
TARGET = "mpg"
# Biến liên tục — trọng tâm phân tích quan hệ cong với mpg
CONTINUOUS = ["displacement", "horsepower", "weight", "acceleration"]
# Biến rời rạc / phân loại — dùng countplot
DISCRETE = ["cylinders", "model_year", "origin"]
ORIGIN_MAP = {1: "USA", 2: "Europe", 3: "Japan"}

REPORT = []  # gom dòng tóm tắt để ghi Results/eda_summary.txt


def log(msg=""):
    print(msg)
    REPORT.append(msg)


def save(fig_name):
    out = os.path.join(FIG_DIR, fig_name)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Đã lưu: {out}")


def load_data():
    """Đọc auto-mpg.data (ngăn cách bằng khoảng trắng), '?' -> NaN."""
    return pd.read_csv(DATA_FILE, sep=r"\s+", names=COLS,
                       quotechar='"', na_values="?")


# ---------------------------------------------------------------------------
# BƯỚC 1 — Tổng quan & chất lượng dữ liệu
# ---------------------------------------------------------------------------
def step1_overview(df):
    log("=" * 70)
    log("BƯỚC 1 — TỔNG QUAN & CHẤT LƯỢNG DỮ LIỆU")
    log("=" * 70)
    log(f"Kích thước: {df.shape[0]} dòng x {df.shape[1]} cột")

    buf = io.StringIO()
    df.info(buf=buf)
    log("\n[df.info()]")
    log(buf.getvalue())
    log("[df.describe()]")
    log(df.describe().round(2).to_string())

    # Missing values
    miss = df.isna().sum()
    log("\nGiá trị thiếu (chỉ horsepower):")
    for c in COLS:
        if miss[c] > 0:
            log(f"  {c}: {miss[c]}")

    # Outlier theo IQR + boxplot
    log("Số outlier theo IQR (1.5*IQR):")
    for col in CONTINUOUS:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[col] < lo) | (df[col] > hi)).sum())
        log(f"  {col:14s}: {n_out}")
    log("")

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, col in zip(axes, CONTINUOUS):
        sns.boxplot(y=df[col], ax=ax, color="#C44E52")
        ax.set_title(col)
    fig.suptitle("BƯỚC 1 — Outlier các biến liên tục (Boxplot IQR)", fontsize=14)
    plt.tight_layout()
    save("step1_outliers_boxplot.png")


# ---------------------------------------------------------------------------
# BƯỚC 2 — Phân phối biến (target + feature)
# ---------------------------------------------------------------------------
def step2_distribution(df):
    # 2a. Histogram target + các biến liên tục trên cùng lưới
    cols = [TARGET] + CONTINUOUS
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, col in zip(axes.ravel(), cols):
        sns.histplot(df[col].dropna(), kde=True, bins=30, color="#4C72B0",
                     edgecolor="white", ax=ax)
        ax.axvline(df[col].mean(), color="crimson", ls="--",
                   label=f"mean={df[col].mean():.1f}")
        ax.set_title(col + (" (TARGET)" if col == TARGET else ""))
        ax.legend(fontsize=8)
    axes.ravel()[-1].axis("off")  # ô thừa
    fig.suptitle("BƯỚC 2a — Phân phối target & biến liên tục (Histogram + KDE)",
                 fontsize=14)
    plt.tight_layout()
    save("step2_distributions.png")

    # 2b. Countplot biến rời rạc
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col in zip(axes, DISCRETE):
        data = df[col].map(ORIGIN_MAP) if col == "origin" else df[col]
        sns.countplot(x=data, ax=ax, hue=data, palette="deep", legend=False)
        ax.set_title(col)
        ax.set_xlabel(col)
    fig.suptitle("BƯỚC 2b — Phân phối biến rời rạc (Countplot)", fontsize=14)
    plt.tight_layout()
    save("step2_discrete_count.png")

    log("=" * 70)
    log("BƯỚC 2 — PHÂN PHỐI BIẾN")
    log("=" * 70)
    log(f"  mpg: mean={df[TARGET].mean():.2f} | median={df[TARGET].median():.2f}"
        f" | skew={df[TARGET].skew():+.2f} (lệch phải nhẹ)\n")


# ---------------------------------------------------------------------------
# BƯỚC 3 — Quan hệ với target (Scatter linear-vs-poly2 + Heatmap)
# ---------------------------------------------------------------------------
def step3_relationship(df):
    d = df.dropna(subset=["horsepower"])
    y = d[TARGET].values

    # 3a. Scatter + Linear (bậc 1) vs Polynomial (bậc 2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, f in zip(axes.ravel(), CONTINUOUS):
        x = d[f].values
        ax.scatter(x, y, s=18, alpha=0.35, color="#888")
        xs = np.linspace(x.min(), x.max(), 200)
        ax.plot(xs, np.polyval(np.polyfit(x, y, 1), xs), "--",
                color="gray", lw=2, label="Linear (bậc 1)")
        ax.plot(xs, np.polyval(np.polyfit(x, y, 2), xs),
                color="#d62728", lw=2.5, label="Polynomial (bậc 2)")
        ax.set_title(f"mpg ~ {f}")
        ax.set_xlabel(f)
        ax.set_ylabel("mpg")
        ax.legend(fontsize=8)
    fig.suptitle("BƯỚC 3a — Feature vs Target: Linear vs Polynomial bậc 2",
                 fontsize=14)
    plt.tight_layout()
    save("step3_feature_vs_target.png")

    # 3b. Heatmap tương quan (gồm cylinders để lộ đa cộng tuyến)
    cols = [TARGET] + CONTINUOUS + ["cylinders"]
    corr = d[cols].corr()
    plt.figure(figsize=(8, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("BƯỚC 3b — Ma trận tương quan (Pearson)")
    plt.tight_layout()
    save("step3_heatmap.png")

    log("=" * 70)
    log("BƯỚC 3 — QUAN HỆ VỚI mpg")
    log("=" * 70)
    log("  Tương quan |r| với mpg:")
    for f in CONTINUOUS:
        log(f"    {f:14s}: {corr.loc[TARGET, f]:+.3f}")
    eng = ["cylinders", "displacement", "horsepower", "weight"]
    log("  Đa cộng tuyến trong nhóm kích cỡ động cơ (|r| cao):")
    for i in range(len(eng)):
        for j in range(i + 1, len(eng)):
            log(f"    {eng[i]:12s} ~ {eng[j]:12s}: "
                f"{corr.loc[eng[i], eng[j]]:+.3f}")
    log("")


# ---------------------------------------------------------------------------
# BƯỚC 4 — Bằng chứng phi tuyến (Residual chữ U + R^2 theo bậc)
# ---------------------------------------------------------------------------
def step4_nonlinearity(df):
    d = df.dropna(subset=["horsepower"])
    y = d[TARGET].values

    # 4a. Residual của hồi quy bậc 1 — nếu uốn 'chữ U' => cần bậc cao hơn
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    log("=" * 70)
    log("BƯỚC 4 — BẰNG CHỨNG PHI TUYẾN")
    log("=" * 70)
    log("Residual hồi quy bậc 1 (độ cong còn sót = % phương sai residual mà")
    log("đường cong bậc 2 giải thích thêm; càng cao -> chữ U càng rõ):")
    for ax, f in zip(axes.ravel(), CONTINUOUS):
        x = d[f].values
        resid = y - np.polyval(np.polyfit(x, y, 1), x)
        ax.scatter(x, resid, s=18, alpha=0.35, color="#888")
        ax.axhline(0, color="crimson", ls="--", lw=1.5)
        order = np.argsort(x)
        xs, rs = x[order], resid[order]
        xr = np.linspace(xs.min(), xs.max(), 200)
        ax.plot(xr, np.polyval(np.polyfit(xs, rs, 2), xr),
                color="#d62728", lw=2.5, label="Xu hướng residual (bậc 2)")
        ax.set_title(f"Residual bậc 1: mpg ~ {f}")
        ax.set_xlabel(f)
        ax.set_ylabel("residual (y - ŷ)")
        ax.legend(fontsize=8)

        rs_hat = np.polyval(np.polyfit(xs, rs, 2), xs)
        ss_res = ((rs - rs_hat) ** 2).sum()
        ss_tot = ((rs - rs.mean()) ** 2).sum()
        curv_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        log(f"  {f:14s}: {curv_r2:6.1%}")
    fig.suptitle("BƯỚC 4 — Residual hồi quy bậc 1 uốn 'chữ U' => cần Polynomial",
                 fontsize=14)
    plt.tight_layout()
    save("step4_residuals_linear.png")

    # 4b. R^2 khớp 1 biến theo bậc (1 -> 2 -> 3)
    log("\nR^2 khớp 1 biến (bậc1 -> bậc2 -> bậc3):")
    for f in CONTINUOUS:
        x = d[f].values
        r2 = []
        for deg in (1, 2, 3):
            yh = np.polyval(np.polyfit(x, y, deg), x)
            r2.append(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        log(f"  {f:14s}: {r2[0]:.3f} -> {r2[1]:.3f} -> {r2[2]:.3f}")

    log("")
    log("KẾT LUẬN:")
    log("  1. mpg ~ {displacement, horsepower, weight} CONG rõ rệt.")
    log("  2. R^2 tăng mạnh ở bậc 2, chững ở bậc 3 -> Polynomial BẬC 2 tối ưu.")
    log("  3. 'acceleration' tương quan yếu, ít cong -> không nên dùng.")
    log("  => Auto MPG phù hợp minh họa Polynomial Regression (weight/hp/displ).")


def save_report():
    out = os.path.join(RESULT_DIR, "eda_summary.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(REPORT) + "\n")
    print(f"Đã lưu: {out}")


def main():
    df = load_data()
    step1_overview(df)
    step2_distribution(df)
    step3_relationship(df)
    step4_nonlinearity(df)
    save_report()
    print("\nHoàn tất EDA 4 bước.")


if __name__ == "__main__":
    main()
