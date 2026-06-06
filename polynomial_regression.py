"""
Polynomial Regression from scratch (chỉ dùng numpy cho phần lõi).

Tinh thần lấy từ repo pickus91/Polynomial-Regression-From-Scratch:
- Tự xây ma trận đặc trưng (Vandermonde)
- Tự tìm trọng số w bằng Normal Equation VÀ Gradient Descent
- Tự tính hàm mất mát MSE, R^2

Mọi thứ minh họa đúng câu thần chú: "phi tuyến theo x, tuyến tính theo w".
"""

import numpy as np


def build_design_matrix(x, degree):
    """Biến 1 đặc trưng x thành [1, x, x^2, ..., x^degree] (ma trận Vandermonde).

    x: mảng 1 chiều (m,)
    trả về: ma trận (m, degree+1), cột đầu toàn số 1 ứng với bias w0.
    """
    x = np.asarray(x, dtype=float).ravel()
    return np.vander(x, N=degree + 1, increasing=True)


def normal_equation(X, y):
    """Nghiệm đóng: w = (X^T X)^(-1) X^T y.

    Dùng pinv (giả nghịch đảo) cho ổn định khi X^T X gần suy biến.
    """
    return np.linalg.pinv(X.T @ X) @ X.T @ y


def predict(X, w):
    return X @ w


def mse(y_true, y_pred):
    return float(np.mean((y_pred - y_true) ** 2))


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot)


def gradient_descent(X, y, lr=0.1, n_iters=5000):
    """Tìm w bằng cách lăn xuống dốc hàm mất mát MSE.

    Trả về (w, lịch sử cost) để vẽ đường hội tụ.
    Giả định X đã được scale (nếu không, các cột x^n chênh lệch khổng lồ
    sẽ làm gradient descent phân kỳ).
    """
    m, n = X.shape
    w = np.zeros(n)
    cost_history = []
    for _ in range(n_iters):
        y_pred = X @ w
        error = y_pred - y
        grad = (2 / m) * (X.T @ error)
        w = w - lr * grad
        cost_history.append(mse(y, y_pred))
    return w, cost_history


class StandardScaler1D:
    """Chuẩn hóa z = (x - mean) / std. Học trên train, áp lên test.

    Tách riêng fit/transform để tránh data leakage (không fit lại trên test).
    """

    def fit(self, x):
        x = np.asarray(x, dtype=float).ravel()
        self.mean_ = x.mean()
        self.std_ = x.std()
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=float).ravel()
        return (x - self.mean_) / self.std_

    def fit_transform(self, x):
        return self.fit(x).transform(x)
