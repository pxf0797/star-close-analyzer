"""
轨迹拟合模块 — 开仓时冻结一条三次多项式价格轨迹，后续 tick 不再重算。
"""

import numpy as np
from numpy.polynomial import Polynomial


def fit_cubic_trajectory(
    prices: np.ndarray,
    entry_idx: int,
    window: int = 20,
) -> Polynomial | None:
    """
    使用 entry_idx 附近 window 根 K 线的价格拟合三次多项式 P(t)。

    t=0 对应 entry_idx 位置。拟合后验证：
    - P'(0) ≈ 0（局部极值）
    - P''(0) < 0（做空需要 concave down，即局部极大）

    返回 numpy Polynomial 对象，invalid 时返回 None。
    """
    half = window // 2
    start = max(0, entry_idx - half)
    end = min(len(prices), entry_idx + half)

    if end - start < 4:
        return None

    t = np.arange(start - entry_idx, end - entry_idx, dtype=float)
    y = prices[start:end].astype(float)

    try:
        poly = Polynomial.fit(t, y, deg=3)
    except np.linalg.LinAlgError:
        return None

    # 验证 P'(0) ≈ 0
    deriv1_at_0 = poly.deriv(1)(0)
    # 验证 P''(0) < 0（做空需要局部极大）
    deriv2_at_0 = poly.deriv(2)(0)

    price_scale = np.mean(np.abs(y))
    if price_scale == 0:
        return None

    # 一阶导相对价格 scale 足够小 & 二阶导为负
    if abs(deriv1_at_0) / max(price_scale, 1e-8) > 0.05:
        return None
    if deriv2_at_0 >= 0:
        return None

    return poly


def find_theoretical_take_profit(poly: Polynomial, t_max: int = 200) -> float | None:
    """
    寻找三次多项式在 t > 0 的下一个局部极小值 — 理论止盈价。
    """
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


def predict_price(poly: Polynomial, t: float | np.ndarray, t_open: int, current_idx: int) -> float | np.ndarray:
    """
    计算冻结轨迹在时间 t 处的预测价格。
    t = current_idx - entry_idx（即 K 线 bar 数）。
    """
    return poly(t)
