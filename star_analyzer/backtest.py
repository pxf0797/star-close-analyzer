"""
回测引擎 — 在历史 K 线上模拟 trajectory_frozen 平仓策略。
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Any

from star_analyzer.trajectory import fit_cubic_trajectory, find_theoretical_take_profit
from star_analyzer.closing import (
    Action, Direction, PositionState, CloseDecision,
    evaluate_close, check_profit_threshold, compute_fit_score,
)

# Phase B 拟合器（可选）
try:
    from star_analyzer.fitting.ensemble import ensemble_fit, TrajectoryFit
    _HAS_ENSEMBLE = True
except ImportError:
    _HAS_ENSEMBLE = False
    TrajectoryFit = None

# Phase C 质量评估（可选）
try:
    from star_analyzer.fitting.quality import evaluate_fit_quality
    _HAS_QUALITY = True
except ImportError:
    _HAS_QUALITY = False


@dataclass
class Trade:
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    quantity: float
    reason: str
    pnl: float = 0.0
    pnl_pct: float = 0.0
    q_score: float | None = None
    cv_rmse: float | None = None

    def __post_init__(self):
        # 做空: 盈利 = (entry - exit) / entry
        self.pnl = (self.entry_price - self.exit_price) * self.quantity
        self.pnl_pct = (self.entry_price - self.exit_price) / self.entry_price * 100


@dataclass
class TickRecord:
    idx: int
    price: float
    pred_price: float
    fit_score: float
    residual: float
    direction: str
    action: str
    reason: str
    trigger_price: float = 0.0


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    records: list[TickRecord] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


class BacktestEngine:
    """trajectory_frozen 策略回测引擎"""

    def __init__(
        self,
        initial_capital: float = 1000.0,
        fee_rate: float = 0.0008,
        half_life: int = 15,
        confidence_threshold: float = 0.3,
        max_relative_distance: float = 0.05,
        hard_stop_multiplier: float = 2.0,
        short_only: bool = True,
        bars_per_year: int = 365 * 24,
        fitter: str = "cubic_wls",         # "cubic_wls" | "ensemble"
        fitter_kwargs: dict | None = None,
        track_quality: bool = True,          # Phase C: 启用 Q-Score 追踪
        q_score_threshold: float = 0.5,      # Q-Score 最低开仓阈值
    ):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.half_life = half_life
        self.confidence_threshold = confidence_threshold
        self.max_relative_distance = max_relative_distance
        self.hard_stop_multiplier = hard_stop_multiplier
        self.short_only = short_only
        self.bars_per_year = bars_per_year
        self.fitter = fitter
        self.fitter_kwargs = fitter_kwargs or {}
        self.track_quality = track_quality
        self.q_score_threshold = q_score_threshold

    def run(self, prices: np.ndarray, entry_signals: np.ndarray | None = None) -> BacktestResult:
        """
        在价格序列上运行回测。

        entry_signals[i] = True 表示在第 i 根 K 线开空仓。
        若为 None，则在 detected local maxima 处开仓。
        """
        result = BacktestResult()
        result.equity_curve = []
        equity = self.initial_capital

        position: PositionState | None = None
        trade_qty = 0.0

        if entry_signals is None:
            entry_signals = self._detect_entry_signals(prices)

        for i in range(len(prices)):
            price = float(prices[i])
            result.equity_curve.append(equity)  # 每 tick 记录期初权益

            if position is not None:
                current_vol = self._estimate_volatility(prices, i)
                decision = evaluate_close(position, i, price, current_vol)
                pred_price = decision.pred_price

                result.records.append(TickRecord(
                    idx=i,
                    price=price,
                    pred_price=pred_price,
                    fit_score=decision.fit_score,
                    residual=decision.residual,
                    direction=decision.direction.value,
                    action=decision.action.value,
                    reason=decision.reason_code,
                    trigger_price=decision.trigger_price,
                ))

                if decision.action == Action.MARKET_CLOSE_ALL:
                    pnl = (position.entry_price - price) * trade_qty
                    pnl_after_fee = pnl - self.fee_rate * trade_qty * (position.entry_price + price)
                    equity += pnl_after_fee
                    trade = Trade(
                        entry_idx=position.entry_idx,
                        exit_idx=i,
                        entry_price=position.entry_price,
                        exit_price=price,
                        quantity=trade_qty,
                        reason=decision.reason_code,
                    )
                    # Phase C: 传递 Q-Score
                    if hasattr(position, "_q_score"):
                        trade.q_score = position._q_score
                        trade.cv_rmse = position._cv_rmse
                    result.trades.append(trade)
                    position = None
                    trade_qty = 0.0
                else:
                    pass

            elif entry_signals[i] and position is None:
                anchor, tp, q_result = self._fit_entry(prices, i)
                if anchor is not None:
                    # Phase C Q-Score 门控
                    if self.track_quality and q_result is not None:
                        if q_result["q_score"] < self.q_score_threshold:
                            result.records.append(TickRecord(
                                idx=i,
                                price=price,
                                pred_price=price,
                                fit_score=float(q_result["q_score"]),
                                residual=0.0,
                                direction="-",
                                action="hold",
                                reason="quality_gate_reject",
                            ))
                            continue

                    if check_profit_threshold(price, tp, self.fee_rate):
                        trade_qty = equity * 0.95 / price
                        position = PositionState(
                            entry_price=price,
                            entry_idx=i,
                            quantity=trade_qty,
                            anchor=anchor,
                            theoretical_tp=tp,
                            fee_rate=self.fee_rate,
                            half_life=self.half_life,
                            confidence_threshold=self.confidence_threshold,
                            max_relative_distance=self.max_relative_distance,
                            hard_stop_multiplier=self.hard_stop_multiplier,
                        )
                        # 挂载 Q-Score 到 position（平仓时传给 Trade）
                        if q_result is not None:
                            position._q_score = q_result["q_score"]
                            position._cv_rmse = q_result["components"]["cv_rmse"]
                        result.records.append(TickRecord(
                            idx=i,
                            price=price,
                            pred_price=float(anchor(0)),
                            fit_score=float(q_result["q_score"]) if q_result else 1.0,
                            residual=0.0,
                            direction=Direction.ALIGNED.value,
                            action="open_short",
                            reason="entry_signal",
                        ))
                    else:
                        result.records.append(TickRecord(
                            idx=i,
                            price=price,
                            pred_price=price,
                            fit_score=1.0,
                            residual=0.0,
                            direction="-",
                            action="hold",
                            reason="profit_threshold_fail",
                        ))
                else:
                    result.records.append(TickRecord(
                        idx=i,
                        price=price,
                        pred_price=price,
                        fit_score=0.0,
                        residual=0.0,
                        direction="-",
                        action="hold",
                        reason="fit_failed",
                    ))
            else:
                result.records.append(TickRecord(
                    idx=i,
                    price=price,
                    pred_price=price,
                    fit_score=1.0,
                    residual=0.0,
                    direction="-",
                    action="hold",
                    reason="no_signal",
                ))

        # 强制平掉未平仓
        if position is not None:
            final_price = float(prices[-1])
            pnl = (position.entry_price - final_price) * trade_qty
            pnl_after_fee = pnl - self.fee_rate * trade_qty * (position.entry_price + final_price)
            equity += pnl_after_fee
            trade = Trade(
                entry_idx=position.entry_idx,
                exit_idx=len(prices) - 1,
                entry_price=position.entry_price,
                exit_price=final_price,
                quantity=trade_qty,
                reason="force_close_eod",
            )
            if hasattr(position, "_q_score"):
                trade.q_score = position._q_score
                trade.cv_rmse = position._cv_rmse
            result.trades.append(trade)

        result.stats = self._compute_stats(result)
        return result

    def _detect_entry_signals(self, prices: np.ndarray) -> np.ndarray:
        """检测局部极大点作为做空入场信号"""
        signals = np.zeros(len(prices), dtype=bool)
        window = 5
        for i in range(window, len(prices) - window):
            local_max = True
            for j in range(1, window + 1):
                if prices[i] <= prices[i - j] or prices[i] <= prices[i + j]:
                    local_max = False
                    break
            if local_max:
                signals[i] = True
        return signals

    def _estimate_volatility(self, prices: np.ndarray, idx: int, window: int = 14) -> float:
        """估计对数收益率的标准差作为局部波动率"""
        start = max(0, idx - window)
        segment = prices[start:idx + 1]
        if len(segment) < 3:
            return 0.002
        log_returns = np.diff(np.log(np.maximum(segment, 1e-12)))
        vol = float(np.std(log_returns))
        return max(vol, 0.0005)

    def _fit_entry(self, prices: np.ndarray, idx: int):
        """
        根据 fitter 配置调用对应拟合器。

        返回 (anchor, theoretical_tp, q_result) 或 (None, None, None)。
        """
        q_result = None

        if self.fitter == "ensemble" and _HAS_ENSEMBLE:
            fit = ensemble_fit(prices, idx, **self.fitter_kwargs)
            if fit is None:
                return None, None, None
            poly = fit_cubic_trajectory(prices, idx, **self.fitter_kwargs)
            tp = find_theoretical_take_profit(poly) if poly is not None else None
            anchor = fit
        else:
            poly = fit_cubic_trajectory(prices, idx, **self.fitter_kwargs)
            if poly is None:
                return None, None, None
            tp = find_theoretical_take_profit(poly)
            anchor = poly

        # Phase C: Q-Score 评估
        if self.track_quality and _HAS_QUALITY:
            q_result = evaluate_fit_quality(prices, idx, anchor,
                                            window=self.fitter_kwargs.get("window", 20))

        return anchor, tp, q_result

    def _compute_stats(self, result: BacktestResult) -> dict:
        trades = result.trades
        if not trades:
            return {"total_trades": 0, "win_rate": 0, "total_pnl": 0, "total_pnl_pct": 0,
                     "profit_factor": 0, "max_drawdown": 0, "sharpe": 0}

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        win_rate = len(wins) / len(trades) * 100
        total_pnl = sum(t.pnl for t in trades)
        total_pnl_pct = sum(t.pnl_pct for t in trades)

        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0
        profit_factor = abs(sum(t.pnl for t in wins) / sum(t.pnl for t in losses)) if losses else float('inf')

        eq = np.array(result.equity_curve)
        returns = np.diff(eq) / np.maximum(eq[:-1], 1e-8)
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(self.bars_per_year)) if len(returns) > 0 and np.std(returns) > 0 else 0

        max_drawdown = 0.0
        peak = eq[0]
        for v in eq:
            peak = max(peak, v)
            dd = (peak - v) / peak
            max_drawdown = max(max_drawdown, dd)

        stats = {
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 4),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "avg_win": round(avg_win, 4),
            "avg_loss": round(avg_loss, 4),
            "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else "inf",
            "sharpe": round(sharpe, 2),
            "max_drawdown": round(max_drawdown * 100, 2),
            "final_equity": round(float(eq[-1]), 4),
        }

        # ---- Phase C: IC 验证 + 质量分档 ----
        q_trades = [t for t in trades if t.q_score is not None]
        if len(q_trades) >= 5:
            from scipy import stats as sp_stats
            q_scores = np.array([t.q_score for t in q_trades])
            pnls = np.array([abs(t.pnl_pct) for t in q_trades])  # PnL 幅度
            ic, ic_pvalue = sp_stats.spearmanr(q_scores, pnls)
            stats["ic_score"] = round(float(ic), 4)
            stats["ic_pvalue"] = round(float(ic_pvalue), 4)
            stats["ic_effective"] = abs(ic) > 0.15

            # 质量分档
            bins = {"high": [], "mid": [], "low": []}
            for t in q_trades:
                if t.q_score >= 0.75:
                    bins["high"].append(t)
                elif t.q_score >= 0.50:
                    bins["mid"].append(t)
                else:
                    bins["low"].append(t)

            quality_bins = {}
            for label, bucket in bins.items():
                if bucket:
                    quality_bins[label] = {
                        "count": len(bucket),
                        "win_rate": round(sum(1 for t in bucket if t.pnl > 0) / len(bucket) * 100, 1),
                        "avg_pnl_pct": round(np.mean([t.pnl_pct for t in bucket]), 2),
                        "total_pnl_pct": round(sum(t.pnl_pct for t in bucket), 2),
                    }
            stats["quality_bins"] = quality_bins

            # Q-Score 分布摘要
            stats["q_score_summary"] = {
                "mean": round(float(np.mean(q_scores)), 4),
                "median": round(float(np.median(q_scores)), 4),
                "std": round(float(np.std(q_scores)), 4),
                "rejected": sum(1 for r in result.records if r.reason == "quality_gate_reject"),
            }

        return stats
