"""
轨迹拟合模块 — 开仓时冻结一条三次多项式价格轨迹，后续 tick 不再重算。

Phase A 改进：
- WLS 指数衰减权重（近期 bar 权重更高）
- 解析约束 f'(0)=0（模型降为 3 参数：c0 + c2·t² + c3·t³）
- CV(RMSE) 质量门控（拒绝高噪声拟合）
- 外推距离限制（t_max = min(window*2, 60)）
"""

import numpy as np
from numpy.polynomial import Polynomial


def fit_cubic_trajectory(
    prices: np.ndarray,
    entry_idx: int,
    window: int = 20,
    half_life_weight: int = 10,
    cv_rmse_threshold: float = 0.05,
) -> Polynomial | None:
    """
    使用 WLS + 解析 f'(0)=0 约束拟合三次多项式 P(t)。

    P(t) = c0 + c2·t² + c3·t³  （c1=0，解析满足 f'(0)=0）

    t=0 对应 entry_idx 位置。拟合后验证：
    - f''(0) < 0（做空需要 concave down）
    - CV(RMSE) < cv_rmse_threshold（噪声门控）

    返回 numpy Polynomial 对象，invalid 时返回 None。
    """
    half = window // 2
    start = max(0, entry_idx - half)
    end = min(len(prices), entry_idx + half)

    if end - start < 4:
        return None

    t = np.arange(start - entry_idx, end - entry_idx, dtype=float)
    y = prices[start:end].astype(float)
    n = len(t)

    # ---- WLS 指数衰减权重 ----
    lam = np.log(2) / half_life_weight
    w = np.exp(-lam * np.abs(t))

    # ---- 解析约束: P(t) = c0 + c2·t² + c3·t³  (f'(0)=0 自动满足) ----
    X = np.column_stack([np.ones(n), t ** 2, t ** 3])

    try:
        XtW = X.T * w  # 逐行加权
        beta = np.linalg.solve(XtW @ X, XtW @ y)
    except np.linalg.LinAlgError:
        return None

    c0, c2, c3 = beta
    poly = Polynomial([c0, 0.0, c2, c3])

    # ---- 验证 f''(0) < 0 ----
    if poly.deriv(2)(0) >= 0:
        return None

    # ---- CV(RMSE) 门控 ----
    y_pred = poly(t)
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    price_scale = np.mean(np.abs(y))
    if price_scale == 0:
        return None
    cv_rmse = rmse / price_scale
    if cv_rmse > cv_rmse_threshold:
        return None

    return poly


def find_theoretical_take_profit(
    poly: Polynomial,
    window: int = 20,
    t_max: int | None = None,
) -> float | None:
    """
    寻找三次多项式在 t > 0 的下一个局部极小值 — 理论止盈价。

    外推距离限制在 2×window，上限 60 bar。
    """
    if t_max is None:
        t_max = min(window * 2, 60)

    deriv = poly.deriv(1)
    roots = deriv.roots()

    best_t = None
    for r in roots:
        r_real = float(r.real) if np.iscomplex(r) else float(r)
        if 1 < r_real < t_max and abs(r.imag) < 1e-8:
            if poly.deriv(2)(r_real) > 0:  # 局部极小
                if best_t is None or r_real < best_t:
                    best_t = r_real

    if best_t is None:
        return None
    return float(poly(best_t))


def predict_price(
    poly: Polynomial, t: float | np.ndarray, t_open: int, current_idx: int
) -> float | np.ndarray:
    """
    计算冻结轨迹在时间 t 处的预测价格。
    t = current_idx - entry_idx（即 K 线 bar 数）。
    """
    return poly(t)
