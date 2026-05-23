"""
高斯过程回归 — Matern 3/2 核 + 白噪声核。

提供预测均值及 ±2σ 不确定性区间，外推时不确定性单调扩大，
预测自然回归到先验均值（局部价格均价），无发散风险。
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    WhiteKernel,
)


def fit_gp_matern(
    prices: np.ndarray,
    entry_idx: int,
    window: int = 20,
    length_scale: float = 5.0,
    n_restarts: int = 5,
    half_life_weight: int = 10,
    cv_rmse_threshold: float = 0.05,
) -> tuple | None:
    """
    GPR 拟合，Matern(3/2) 核。

    返回 (predict_fn, meta_dict) 或 None。
    predict_fn(t) → 预测价格（均值）。
    meta_dict 含 'std_fn' — std_fn(t) → 预测标准差。
    """
    half = window // 2
    start = max(0, entry_idx - half)
    end = min(len(prices), entry_idx + half)

    if end - start < 4:
        return None

    t = np.arange(start - entry_idx, end - entry_idx, dtype=float)
    y = prices[start:end].astype(float)

    # 转换为 2D 特征
    X = t.reshape(-1, 1)

    # 核: 常数核 × Matern(3/2) + 白噪声
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=length_scale, length_scale_bounds=(1.0, 50.0), nu=1.5)
        + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 1.0))
    )

    # 归一化 y（GPR 对尺度敏感）
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std < 1e-8:
        return None
    y_norm = (y - y_mean) / y_std

    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=n_restarts,
        alpha=1e-6,  # 数值稳定
        normalize_y=False,
        random_state=42,
    )

    try:
        gp.fit(X, y_norm)
    except Exception:
        return None

    def predict(t_val: float | np.ndarray) -> float | np.ndarray:
        t_arr = np.asarray(t_val).reshape(-1, 1)
        y_pred_norm = gp.predict(t_arr, return_std=False)
        return (y_pred_norm * y_std + y_mean).ravel()

    def predict_with_std(
        t_val: float | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        t_arr = np.asarray(t_val).reshape(-1, 1)
        y_pred_norm, std_norm = gp.predict(t_arr, return_std=True)
        return y_pred_norm * y_std + y_mean, std_norm * y_std

    # 验证 f''(0) < 0：在 t=0 附近数值估计二阶导
    h = 0.5
    f_h = float(predict(h))
    f_0 = float(predict(0.0))
    f_neg = float(predict(-h))
    # 中心差分: f''(0) ≈ (f(h) - 2*f(0) + f(-h)) / h²
    second_deriv = (f_h - 2 * f_0 + f_neg) / (h ** 2)
    if second_deriv >= 0:
        return None

    # CV(RMSE) 门控
    y_pred_in = predict(t)
    rmse = np.sqrt(np.mean((y - y_pred_in) ** 2))
    price_scale = np.mean(np.abs(y))
    cv_rmse = rmse / price_scale if price_scale > 0 else float("inf")
    if cv_rmse > cv_rmse_threshold:
        return None

    return predict, {
        "method": "gp_matern",
        "kernel": gp.kernel_,
        "length_scale": gp.kernel_.get_params().get("k1__k2__length_scale", length_scale),
        "cv_rmse": cv_rmse,
        "y_mean": y_mean,
        "y_std": y_std,
        "predict_with_std": predict_with_std,
        "second_deriv_at_0": second_deriv,
    }
