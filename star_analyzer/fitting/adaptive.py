"""
Walk-Forward 自适应 — 定期重校准门控阈值和衰减速率。

周期：建议每 2-3 月运行一次（或每 N 笔交易）。
预警：任一参数变动超过 30% 触发人工审核。
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class CalibrationResult:
    """校准结果。"""

    cv_rmse_threshold: float = 0.05
    half_life_weight: int = 10
    q_score_threshold: float = 0.50
    param_drift: dict[str, float] = field(default_factory=dict)  # 参数变动比例
    warnings: list[str] = field(default_factory=list)
    sample_size: int = 0


def calibrate_thresholds(
    historical_results: list[dict],
    current_params: dict | None = None,
    drift_warning_pct: float = 0.30,
) -> CalibrationResult:
    """
    基于历史交易结果重校准阈值。

    historical_results: 每笔交易记录 [{"cv_rmse": ..., "q_score": ..., "pnl_pct": ..., ...}, ...]

    策略：
    - cv_rmse: 取盈利交易的 75 分位数（拒绝误差比此大的拟合）
    - q_score: 取盈利交易中位数（低于此值视为弱信号）
    - half_life: 按波动率取整（保持 10 附近）
    """
    if not historical_results:
        return CalibrationResult()

    winning = [r for r in historical_results if r.get("pnl_pct", 0) > 0]
    if len(winning) < 5:
        return CalibrationResult(sample_size=len(winning),
                                 warnings=["样本量不足（<5 笔盈利），保持当前参数"])

    all_cv = np.array([r["cv_rmse"] for r in historical_results if "cv_rmse" in r])
    win_cv = np.array([r["cv_rmse"] for r in winning if "cv_rmse" in r])
    win_q = np.array([r.get("q_score", 0.5) for r in winning])

    if len(win_cv) == 0:
        return CalibrationResult(sample_size=len(winning))

    # 盈利交易的 75 分位数
    cv_threshold = float(np.percentile(win_cv, 75))
    cv_threshold = np.clip(cv_threshold, 0.02, 0.15)

    # Q-Score 中位数（保守）
    q_threshold = float(np.median(win_q)) if len(win_q) > 0 else 0.50
    q_threshold = np.clip(q_threshold, 0.40, 0.70)

    result = CalibrationResult(
        cv_rmse_threshold=round(cv_threshold, 4),
        q_score_threshold=round(q_threshold, 4),
        half_life_weight=10,  # 保持默认
        sample_size=len(historical_results),
    )

    # 参数漂移检测
    if current_params:
        drift = {}
        for key, old_val in current_params.items():
            if key == "half_life_weight":
                continue
            new_val = getattr(result, key, old_val)
            if old_val > 1e-10:
                delta = abs(new_val - old_val) / old_val
                drift[key] = round(delta, 4)
                if delta > drift_warning_pct:
                    result.warnings.append(
                        f"参数 {key} 变动 {delta:.1%}（{old_val:.4f}→{new_val:.4f}），超出 {drift_warning_pct:.0%} 预警线"
                    )

        result.param_drift = drift

    return result


def walk_forward_validate(
    prices: np.ndarray,
    trades: list,
    fold_count: int = 3,
) -> dict:
    """
    Walk-forward 验证：将数据按时间分为 N 折，逐折训练/验证。

    返回每折和汇总的指标。
    """
    n = len(prices)
    fold_size = n // fold_count
    if fold_size < 50:
        return {"error": "数据量不足以做 walk-forward 分折"}

    folds = []
    for f in range(fold_count):
        train_end = (f + 1) * fold_size
        test_start = train_end
        test_end = min((f + 2) * fold_size, n) if f < fold_count - 1 else n

        train_trades = [t for t in trades if t.entry_idx < train_end]
        test_trades = [t for t in trades if train_end <= t.entry_idx < test_end]

        train_win_rate = (
            sum(1 for t in train_trades if t.pnl > 0) / len(train_trades) * 100
            if train_trades else 0
        )
        test_win_rate = (
            sum(1 for t in test_trades if t.pnl > 0) / len(test_trades) * 100
            if test_trades else 0
        )

        folds.append({
            "fold": f + 1,
            "train_bars": (0, train_end),
            "test_bars": (train_end, test_end),
            "train_trades": len(train_trades),
            "test_trades": len(test_trades),
            "train_win_rate": round(train_win_rate, 1),
            "test_win_rate": round(test_win_rate, 1),
            "overfit_gap": round(train_win_rate - test_win_rate, 1),
        })

    test_wrs = [f["test_win_rate"] for f in folds if f["test_trades"] > 0]
    return {
        "folds": folds,
        "mean_test_win_rate": round(np.mean(test_wrs), 1) if test_wrs else 0,
        "std_test_win_rate": round(np.std(test_wrs), 1) if test_wrs else 0,
        "stability": "stable" if (np.std(test_wrs) < 10 if test_wrs else True) else "unstable",
    }
