"""
EDA — Exploratory Data Analysis cho dataset CCPP (dùng seaborn).

Output: Figures/EDA/*.png + Results/eda_summary.txt
Chạy:  python3 eda_ccpp.py
"""

import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(__file__)
FIG_DIR = os.path.join(BASE_DIR, "Figures", "EDA")
RESULT_DIR = os.path.join(BASE_DIR, "Results")
DATA_CSV = os.path.join(BASE_DIR, "data", "ccpp.csv")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook", palette="deep")

FEATURES = ["AT", "V", "AP", "RH"]
TARGET = "PE"
LABELS = {
    "AT": "AT — Nhiệt độ môi trường (°C)",
    "V":  "V — Chân không hơi xả (cm Hg)",
    "AP": "AP — Áp suất môi trường (mbar)",
    "RH": "RH — Độ ẩm tương đối (%)",
    "PE": "PE — Công suất điện (MW)",
}


def load_data():
    return pd.read_csv(DATA_CSV)


def plot_histograms(df):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    cols = FEATURES + [TARGET]
    for ax, col in zip(axes.ravel(), cols):
        sns.histplot(df[col], kde=True, bins=40, color="#4C72B0", ax=ax,
                     edgecolor="white", alpha=0.85)
        ax.axvline(df[col].mean(), color="crimson", linestyle="--",
                   linewidth=1.5, label=f"mean={df[col].mean():.2f}")
        ax.set_title(LABELS[col], fontsize=10)
        ax.legend(fontsize=8)
    for ax in axes.ravel()[len(cols):]:
        ax.set_visible(False)
    fig.suptitle("CCPP — Phân bố các biến (histogram + KDE)", fontsize=14,
                 y=1.00)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "histograms.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Đã lưu: {out}")


def plot_scatter_vs_target(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, col in zip(axes.ravel(), FEATURES):
        sns.regplot(
            data=df, x=col, y=TARGET, ax=ax,
            scatter_kws={"s": 8, "alpha": 0.2, "color": "#4C72B0"},
            line_kws={"color": "crimson", "linewidth": 2},
            ci=None,
        )
        corr = df[col].corr(df[TARGET])
        ax.set_title(f"{col} vs PE  (r = {corr:+.3f})", fontsize=11)
        ax.set_xlabel(LABELS[col])
        ax.set_ylabel(LABELS[TARGET])
    fig.suptitle("CCPP — Đặc trưng vs PE (đường hồi quy tuyến tính)",
                 fontsize=14, y=1.00)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "scatter_vs_PE.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Đã lưu: {out}")


def plot_corr_heatmap(df):
    cols = FEATURES + [TARGET]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        corr, annot=True, fmt="+.2f", cmap="RdBu_r", vmin=-1, vmax=1,
        center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
        annot_kws={"size": 11}, ax=ax,
    )
    ax.set_title("CCPP — Ma trận tương quan Pearson", fontsize=13)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "corr_heatmap.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Đã lưu: {out}")


def plot_boxplots(df):
    cols = FEATURES + [TARGET]
    # Chuẩn hóa về z-score để vẽ chung 1 trục y (mỗi biến có thang đo khác nhau)
    z = (df[cols] - df[cols].mean()) / df[cols].std()
    long = z.melt(var_name="Biến", value_name="z-score")

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(data=long, x="Biến", y="z-score", hue="Biến",
                palette="Set2", ax=ax, legend=False, width=0.5,
                fliersize=3)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title("CCPP — Boxplot chuẩn hóa (z-score) — phát hiện outlier",
                 fontsize=13)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "boxplots.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Đã lưu: {out}")


def plot_pairplot(df):
    """Pairplot toàn cảnh — quan hệ từng cặp + phân bố trên đường chéo."""
    # Sample bớt cho nhẹ (9568 -> 2000) — đủ thấy structure
    sample = df.sample(n=2000, random_state=42)
    g = sns.pairplot(
        sample[FEATURES + [TARGET]],
        diag_kind="kde",
        plot_kws={"s": 10, "alpha": 0.25, "color": "#4C72B0"},
        diag_kws={"color": "#4C72B0", "fill": True},
        height=2.2,
    )
    g.fig.suptitle("CCPP — Pairplot (sample 2000 mẫu)", y=1.01,
                   fontsize=13)
    out = os.path.join(FIG_DIR, "pairplot.png")
    g.savefig(out, dpi=110)
    plt.close()
    print(f"Đã lưu: {out}")


def save_summary(df):
    cols = FEATURES + [TARGET]
    lines = []
    lines.append("CCPP — Tóm tắt EDA")
    lines.append(f"Kích thước: {df.shape[0]} mẫu x {df.shape[1]} cột")
    lines.append(f"Số giá trị thiếu: {int(df.isna().sum().sum())}")
    lines.append("")
    lines.append("Thống kê mô tả:")
    lines.append(df[cols].describe().round(3).to_string())
    lines.append("")
    lines.append("Tương quan Pearson với PE (sắp xếp |r| giảm dần):")
    corr_pe = df[FEATURES].corrwith(df[TARGET]).sort_values(
        key=lambda s: s.abs(), ascending=False
    )
    for k, v in corr_pe.items():
        lines.append(f"  {k:>3} : r = {v:+.4f}")
    lines.append("")
    lines.append("Ma trận tương quan đầy đủ:")
    lines.append(df[cols].corr().round(3).to_string())

    out = os.path.join(RESULT_DIR, "eda_summary.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Đã lưu: {out}")


def main():
    df = load_data()
    print(f"Loaded CCPP: {df.shape[0]} mẫu x {df.shape[1]} cột "
          f"(missing={int(df.isna().sum().sum())})")
    plot_histograms(df)
    plot_scatter_vs_target(df)
    plot_corr_heatmap(df)
    plot_boxplots(df)
    plot_pairplot(df)
    save_summary(df)


if __name__ == "__main__":
    main()
