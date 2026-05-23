"""
模型集成 — 并行运行多种拟合器，取加权中位数作为集成预测。

支持：
- WLS 三次多项式（Phase A）
- Elastic Net 正则化三次
- 指数均值回复
- 高斯过程回归 (Matern 3/2)

输出 TrajectoryFit，兼容现有 Polynomial 接口。
"""

from dataclasses import dataclass, field
from typing import Callable
import numpy as np

from star_analyzer.trajectory import fit_cubic_trajectory
from star_analyzer.fitting.elastic_net import fit_elastic_net
from star_analyzer.fitting.mean_reversion import fit_mean_reversion
from star_analyzer.fitting.gp_regression import fit_gp_matern


@dataclass
class TrajectoryFit:
    """集成拟合结果，兼容 Polynomial 接口（__call__ / deriv）。"""

    predict_fn: Callable  # predict(t) → price
    method: str = "ensemble"
    cv_rmse: float = 0.0
    models: list[dict] = field(default_factory=list)
    _std_fn: Callable | None = None  # type: ignore[assignment]
    _disagreement: float = 0.0

    def __call__(self, t: float | np.ndarray) -> float | np.ndarray:
        return self.predict_fn(t)

    def deriv(self, n: int = 1):
        """返回数值导数估算器，兼容 Polynomial.deriv(n)(t) 调用模式。"""
        h = 0.5

        class _NumDeriv:
            def __init__(self, parent, order, step):
                self._parent = parent
                self._order = order
                self._step = step

            def __call__(self, t: float) -> float:
                if self._order == 1:
                    return (
                        self._parent(t + self._step) - self._parent(t - self._step)
                    ) / (2 * self._step)
                elif self._order == 2:
                    return (
                        self._parent(t + self._step)
                        - 2 * self._parent(t)
                        + self._parent(t - self._step)
                    ) / (self._step ** 2)
                return 0.0

        return _NumDeriv(self.predict_fn, n, h)

    @property
    def std_fn(self) -> Callable | None:
        """GPR 提供的不确定性函数。std_fn(t) → 标准偏差。"""
        return self._std_fn

    @property
    def disagreement(self) -> float:
        """模型间分歧度（预测的标准偏差 / 均值）。"""
        return self._disagreement


def ensemble_fit(
    prices: np.ndarray,
    entry_idx: int,
    window: int = 20,
    half_life_weight: int = 10,
    cv_rmse_threshold: float = 0.05,
    min_models: int = 1,
) -> TrajectoryFit | None:
    """
    并行运行所有拟合器，返回集成 TrajectoryFit。

    策略：
    1. 运行 WLS 三次 + Elastic Net + 均值回复 + GPR
    2. 收集成功通过的拟合器
    3. 取各模型在 [0, window] 范围的预测中位数作为集成预测
    4. 若 GPR 可用，提供不确定性量化
    """
    half = window // 2
    start = max(0, entry_idx - half)
    end = min(len(prices), entry_idx + half)
    t_local = np.arange(start - entry_idx, end - entry_idx, dtype=float)

    results = []
    predictions_list = []

    # ---- 1. WLS 三次（Phase A） ----
    poly = fit_cubic_trajectory(
        prices, entry_idx, window=window,
        half_life_weight=half_life_weight, cv_rmse_threshold=cv_rmse_threshold,
    )
    if poly is not None:
        results.append({"method": "cubic_wls", "predict": lambda t_, p=poly: p(t_)})
        predictions_list.append(poly(t_local))

    # ---- 2. Elastic Net ----
    en_result = fit_elastic_net(
        prices, entry_idx, window=window,
        half_life_weight=half_life_weight,
    )
    if en_result is not None:
        pred_fn, meta = en_result
        if meta["cv_rmse"] <= cv_rmse_threshold:
            results.append({"method": "elastic_net", "predict": pred_fn, "meta": meta})
            predictions_list.append(pred_fn(t_local))

    # ---- 3. 指数均值回复 ----
    mr_result = fit_mean_reversion(
        prices, entry_idx, window=window,
        half_life_weight=half_life_weight, cv_rmse_threshold=cv_rmse_threshold,
    )
    if mr_result is not None:
        pred_fn, meta = mr_result
        results.append({"method": "mean_reversion", "predict": pred_fn, "meta": meta})
        predictions_list.append(pred_fn(t_local))

    # ---- 4. 高斯过程回归 ----
    gp_result = fit_gp_matern(
        prices, entry_idx, window=window,
        half_life_weight=half_life_weight, cv_rmse_threshold=cv_rmse_threshold,
    )
    std_fn = None
    if gp_result is not None:
        pred_fn, meta = gp_result
        results.append({"method": "gp_matern", "predict": pred_fn, "meta": meta})
        predictions_list.append(pred_fn(t_local))
        std_fn = meta.get("predict_with_std")

    # ---- 不足 min_models → 放弃 ----
    if len(results) < min_models:
        return None

    # ---- 集成预测：各模型预测的中位数 ----
    if len(predictions_list) == 1:
        ensemble_pred_fn = results[0]["predict"]
    else:
        pred_stack = np.array(predictions_list)  # [n_models, n_points]

        def _ensemble_pred(t_val: float | np.ndarray) -> float | np.ndarray:
            t_arr = np.atleast_1d(np.asarray(t_val))
            all_preds = []
            for r in results:
                all_preds.append(r["predict"](t_arr))
            stacked = np.array(all_preds)  # [n_models, len(t)]
            return np.median(stacked, axis=0)

        ensemble_pred_fn = _ensemble_pred

    # ---- 计算分歧度 ----
    if len(predictions_list) > 1:
        pred_stack = np.array(predictions_list)
        disagreement = float(
            np.mean(np.std(pred_stack, axis=0) / (np.abs(np.mean(pred_stack, axis=0)) + 1e-8))
        )
    else:
        disagreement = 0.0

    # ---- 计算集成 CV(RMSE) ----
    y_true = prices[start:end].astype(float)
    y_pred = np.atleast_1d(ensemble_pred_fn(t_local))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    price_scale = np.mean(np.abs(y_true))
    cv_rmse = float(rmse / price_scale) if price_scale > 0 else float("inf")

    return TrajectoryFit(
        predict_fn=ensemble_pred_fn,
        method=f"ensemble({','.join(r['method'] for r in results)})",
        cv_rmse=cv_rmse,
        models=results,
        _std_fn=std_fn,
        _disagreement=disagreement,
    )
