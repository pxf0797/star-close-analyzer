"""
Elastic Net 正则化三次多项式拟合。

在 f'(0)=0 约束下（即无一次项），对 [c2, c3] 施加 L1+L2 混合惩罚，
抑制噪声驱动的系数膨胀。
"""

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler


def fit_elastic_net(
    prices: np.ndarray,
    entry_idx: int,
    window: int = 20,
    alpha: float = 0.001,
    l1_ratio: float = 0.5,
    half_life_weight: int = 10,
) -> tuple | None:
    """
    Elastic Net 正则化三次拟合，f'(0)=0 约束。

    返回 (predict_fn, meta_dict) 或 None。
    predict_fn(t) → 预测价格。
    """
    half = window // 2
    start = max(0, entry_idx - half)
    end = min(len(prices), entry_idx + half)

    if end - start < 4:
        return None

    t = np.arange(start - entry_idx, end - entry_idx, dtype=float)
    y = prices[start:end].astype(float)
    n = len(t)

    # WLS 权重
    lam = np.log(2) / half_life_weight
    w = np.exp(-lam * np.abs(t))
    sample_weight = w / w.sum() * n  # sklearn 期望的 scale

    # 设计矩阵 [t², t³]（c0 用截距，c1=0 解析约束）
    X = np.column_stack([t ** 2, t ** 3])

    # 标准化特征（正则化要求）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=True,
        max_iter=5000,
        random_state=42,
    )
    model.fit(X_scaled, y, sample_weight=sample_weight)

    # 系数反标准化
    c2_scaled, c3_scaled = model.coef_
    c0 = model.intercept_

    x_mean = scaler.mean_
    x_std = scaler.scale_

    # X_scaled = (X - mean) / std
    # P(t) = c0 + c2_scaled*(t²-μ₁)/σ₁ + c3_scaled*(t³-μ₂)/σ₂
    #      = (c0 - c2_scaled*μ₁/σ₁ - c3_scaled*μ₂/σ₂) + (c2_scaled/σ₁)*t² + (c3_scaled/σ₂)*t³
    delta_c0 = -c2_scaled * x_mean[0] / x_std[0] - c3_scaled * x_mean[1] / x_std[1]
    c0_real = float(c0 + delta_c0)
    c2_real = float(c2_scaled / x_std[0])
    c3_real = float(c3_scaled / x_std[1])

    def predict(t_val: float | np.ndarray) -> float | np.ndarray:
        return c0_real + c2_real * t_val ** 2 + c3_real * t_val ** 3

    # 验证 f''(0) < 0
    if 2 * c2_real >= 0:
        return None

    # CV(RMSE)
    y_pred = predict(t)
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    price_scale = np.mean(np.abs(y))
    cv_rmse = rmse / price_scale if price_scale > 0 else float("inf")

    return predict, {
        "method": "elastic_net",
        "c0": c0_real,
        "c2": c2_real,
        "c3": c3_real,
        "alpha": alpha,
        "l1_ratio": l1_ratio,
        "cv_rmse": cv_rmse,
    }
