"""
拟合质量评估 — 复合 Q-Score 系统。

三级门控管道：
- S1: 结构有效性（f''(0) < 0）
- S2: 残差独立性（Durbin-Watson + Ljung-Box）
- S3: 拟合质量（CV(RMSE) + R² + 外推稳定性）

输出 0-1 综合质量分，用于自动化开仓决策。
"""

import numpy as np
from scipy import stats as sp_stats


def compute_durbin_watson(residuals: np.ndarray) -> float:
    """
    Durbin-Watson 检验 — 一阶残差自相关。

    DW ≈ 2: 无自相关（理想）
    DW < 1.5: 正自相关（残差同号连续 → 模型有系统性偏差）
    DW > 2.5: 负自相关（残差交替符号 → 过拟合振荡）
    """
    if len(residuals) < 3:
        return 2.0
    diff = np.diff(residuals)
    numerator = np.sum(diff ** 2)
    denominator = np.sum(residuals ** 2)
    if denominator < 1e-15:
        return 2.0
    return float(numerator / denominator)


def compute_ljung_box(residuals: np.ndarray, lags: int = 5) -> tuple[float, float]:
    """
    Ljung-Box 检验 — 多阶残差自相关。

    返回 (Q_statistic, p_value)。
    p > 0.05 → 残差为白噪声（通过）。
    """
    n = len(residuals)
    if n <= lags:
        return 0.0, 1.0

    acf = np.zeros(lags + 1)
    r_mean = np.mean(residuals)
    r_var = np.var(residuals)
    if r_var < 1e-15:
        return 0.0, 1.0

    for k in range(lags + 1):
        acf[k] = np.mean((residuals[:n - k] - r_mean) * (residuals[k:] - r_mean)) / r_var

    q_stat = 0.0
    for k in range(1, lags + 1):
        q_stat += acf[k] ** 2 / (n - k)
    q_stat *= n * (n + 2)

    p_value = 1.0 - sp_stats.chi2.cdf(q_stat, lags)
    return float(q_stat), float(p_value)


def compute_cv_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """CV(RMSE) = RMSE / mean(|y|)。无量纲拟合误差。"""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    scale = np.mean(np.abs(y_true))
    if scale < 1e-12:
        return float("inf")
    return float(rmse / scale)


def compute_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² = 1 - SS_res / SS_tot。"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-15:
        return 1.0
    return float(1.0 - ss_res / ss_tot)


def compute_rolling_cv_oos_ratio(
    prices: np.ndarray,
    entry_idx: int,
    window: int = 20,
    n_train: int = 15,
) -> float:
    """
    滚动窗口交叉验证：OOS RMSE / IS RMSE。

    训练窗口 [0:n_train] → 预测 [n_train:window]
    OOS/IS < 2.0 → 泛化能力可接受。
    """
    half = window // 2
    start = max(0, entry_idx - half)
    end = min(len(prices), entry_idx + half)

    if end - start < n_train + 2:
        return 1.0

    t_full = np.arange(start - entry_idx, end - entry_idx, dtype=float)
    y_full = prices[start:end].astype(float)

    # 训练集拟合
    t_train = t_full[:n_train]
    y_train = y_full[:n_train]

    # 三次多项式 OLS（带 f'(0)=0 约束）
    X_train = np.column_stack([np.ones(len(t_train)), t_train ** 2, t_train ** 3])
    try:
        beta = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
    except np.linalg.LinAlgError:
        return 1.0

    c0, c2, c3 = beta
    pred_fn = lambda t_: c0 + c2 * t_ ** 2 + c3 * t_ ** 3

    # IS 误差
    y_pred_is = pred_fn(t_train)
    is_rmse = np.sqrt(np.mean((y_train - y_pred_is) ** 2))

    # OOS 误差
    t_oos = t_full[n_train:]
    y_oos = y_full[n_train:]
    y_pred_oos = pred_fn(t_oos)
    oos_rmse = np.sqrt(np.mean((y_oos - y_pred_oos) ** 2))

    if is_rmse < 1e-12:
        return float(oos_rmse / 1e-6) if oos_rmse > 1e-12 else 1.0

    return float(oos_rmse / is_rmse)


def compute_q_score(
    dw_stat: float,
    lb_pvalue: float,
    cv_rmse: float,
    oos_is_ratio: float,
    r_squared: float,
) -> dict:
    """
    复合 Q-Score。

    权重设计：
    - DW 残差独立性  25%（残差自相关直接宣告模型无效）
    - LB 残差白噪声  20%
    - CV(RMSE) 精度  25%
    - 外推稳定性      20%
    - R² 解释力      10%

    返回 {"q_score": float, "grade": str, "components": dict}
    """
    dw_pass = 1.0 if 1.5 <= dw_stat <= 2.5 else 0.0
    lb_pass = 1.0 if lb_pvalue > 0.05 else 0.0

    q = (
        0.25 * dw_pass
        + 0.20 * lb_pass
        + 0.25 * max(0.0, 1.0 - cv_rmse / 0.10)
        + 0.20 * max(0.0, 1.0 - oos_is_ratio / 2.0)
        + 0.10 * max(0.0, (r_squared - 0.50) / 0.50)
    )

    q_score = float(np.clip(q, 0.0, 1.0))

    if q_score >= 0.75:
        grade = "accept"
    elif q_score >= 0.50:
        grade = "marginal"
    else:
        grade = "reject"

    return {
        "q_score": q_score,
        "grade": grade,
        "components": {
            "dw_stat": float(dw_stat),
            "dw_pass": bool(dw_pass),
            "lb_pvalue": float(lb_pvalue),
            "lb_pass": bool(lb_pass),
            "cv_rmse": float(cv_rmse),
            "oos_is_ratio": float(oos_is_ratio),
            "r_squared": float(r_squared),
        },
    }


def evaluate_fit_quality(
    prices: np.ndarray,
    entry_idx: int,
    predict_fn: callable,
    window: int = 20,
) -> dict | None:
    """
    对一次拟合执行完整质量评估，返回 Q-Score 结果。

    这是质量门控的统一入口。
    """
    half = window // 2
    start = max(0, entry_idx - half)
    end = min(len(prices), entry_idx + half)

    if end - start < 4:
        return None

    t = np.arange(start - entry_idx, end - entry_idx, dtype=float)
    y_true = prices[start:end].astype(float)

    try:
        y_pred = np.atleast_1d(np.asarray(predict_fn(t), dtype=float))
    except Exception:
        return None

    residuals = y_true - y_pred

    dw_stat = compute_durbin_watson(residuals)
    _, lb_pvalue = compute_ljung_box(residuals)
    cv_rmse = compute_cv_rmse(y_true, y_pred)
    r_squared = compute_r_squared(y_true, y_pred)
    oos_is_ratio = compute_rolling_cv_oos_ratio(prices, entry_idx, window)

    return compute_q_score(dw_stat, lb_pvalue, cv_rmse, oos_is_ratio, r_squared)
