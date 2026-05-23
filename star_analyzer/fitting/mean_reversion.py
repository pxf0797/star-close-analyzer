"""
指数均值回复模型 — f(t) = p₀ + α·(1 - e^(-λt))

纯 numpy 实现，无外部依赖。三参数模型天然避免过拟合，
外推渐近于 p₀ + α，不会发散。
"""

import numpy as np


def _model(t: np.ndarray, p0: float, alpha: float, lam: float) -> np.ndarray:
    """f(t) = p0 + alpha * (1 - exp(-lambda * t))"""
    return p0 + alpha * (1.0 - np.exp(-lam * t))


def _jacobian(t: np.ndarray, p0: float, alpha: float, lam: float) -> np.ndarray:
    """解析雅可比矩阵，加速 LM 迭代。"""
    n = len(t)
    J = np.zeros((n, 3))
    J[:, 0] = 1.0  # ∂f/∂p0
    J[:, 1] = 1.0 - np.exp(-lam * t)  # ∂f/∂alpha
    J[:, 2] = alpha * t * np.exp(-lam * t)  # ∂f/∂lam
    return J


def fit_mean_reversion(
    prices: np.ndarray,
    entry_idx: int,
    window: int = 20,
    half_life_weight: int = 10,
    cv_rmse_threshold: float = 0.05,
    max_iter: int = 100,
) -> tuple | None:
    """
    拟合指数均值回复模型 f(t) = p0 + alpha * (1 - exp(-lambda * t))。

    返回 (predict_fn, meta_dict) 或 None。
    alpha < 0 表示价格从峰值回落（符合做空预期）。
    """
    half = window // 2
    start = max(0, entry_idx - half)
    end = min(len(prices), entry_idx + half)

    if end - start < 4:
        return None

    t = np.arange(start - entry_idx, end - entry_idx, dtype=float)
    y = prices[start:end].astype(float)
    n = len(t)

    # 初始猜测
    p0_init = float(y[-1])  # 渐进均衡价 ≈ 窗口尾端价格
    alpha_init = float(y[0] - y[-1])  # 回撤幅度
    if alpha_init > 0:
        alpha_init = -alpha_init  # 做空预期下跌，alpha < 0
    lam_init = 0.3

    # WLS 权重
    lam_w = np.log(2) / half_life_weight
    w = np.exp(-lam_w * np.abs(t))

    # Levenberg-Marquardt 非线性最小二乘
    beta = np.array([p0_init, alpha_init, lam_init])
    nu = 1e-3  # damping 初始值

    for _ in range(max_iter):
        residual = (y - _model(t, *beta)) * np.sqrt(w)
        J = _jacobian(t, *beta) * np.sqrt(w)[:, None]

        JTJ = J.T @ J
        JTJ_damped = JTJ + nu * np.diag(np.diag(JTJ))
        delta = np.linalg.solve(JTJ_damped, J.T @ residual)

        beta_new = beta + delta

        # 约束: lam > 0
        beta_new[2] = max(beta_new[2], 1e-6)

        old_cost = np.sum(residual ** 2)
        new_residual = (y - _model(t, *beta_new)) * np.sqrt(w)
        new_cost = np.sum(new_residual ** 2)

        if new_cost < old_cost:
            beta = beta_new
            nu = max(nu / 3, 1e-8)
        else:
            nu = min(nu * 3, 1e6)

        if np.max(np.abs(delta)) < 1e-8:
            break

    p0, alpha, lam_r = beta

    # lam 过小意味着基本没有均值回复 → 退化为水平线
    if lam_r < 1e-4:
        return None

    def predict(t_val: float | np.ndarray) -> float | np.ndarray:
        return float(p0) + float(alpha) * (1.0 - np.exp(-float(lam_r) * np.asarray(t_val)))

    # CV(RMSE) 门控
    y_pred = predict(t)
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    price_scale = np.mean(np.abs(y))
    cv_rmse = rmse / price_scale if price_scale > 0 else float("inf")
    if cv_rmse > cv_rmse_threshold:
        return None

    return predict, {
        "method": "mean_reversion",
        "p0": float(p0),
        "alpha": float(alpha),
        "lam": float(lam_r),
        "cv_rmse": cv_rmse,
    }
