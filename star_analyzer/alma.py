"""
ALMA — Arnaud Legoux Moving Average（零滞后高斯核滤波器）

基于高斯分布加权：W(i) = exp(-(i - m)² / (2·σ²))
m 为偏移量控制滞后，σ 控制平滑度。

比 SMA 滞后小得多，适合作为价格趋势参考线。
"""

import numpy as np


def alma(prices: np.ndarray, window: int = 9, offset: float = 0.85, sigma: float = 6.0) -> np.ndarray:
    """
    计算 ALMA 滤波价格。

    Args:
        prices: 价格序列
        window: 窗口大小 (默认 9)
        offset: 偏移量 [0, 1]，越大滞后越小 (默认 0.85)
        sigma: 高斯核宽度，越大越平滑 (默认 6.0)
    Returns:
        与 prices 等长的 ALMA 序列，前 window-1 个值为 NaN
    """
    n = len(prices)
    if n < window:
        return np.full(n, np.nan)

    m_val = offset * (window - 1)
    s = window / sigma

    # 权重: W(i) = exp(-(i - m)² / (2·s²))
    i = np.arange(window, dtype=float)
    weights = np.exp(-0.5 * ((i - m_val) / s) ** 2)
    weights /= weights.sum()  # 归一化

    result = np.full(n, np.nan)
    for t in range(window - 1, n):
        result[t] = np.dot(weights, prices[t - window + 1:t + 1])
    return result
