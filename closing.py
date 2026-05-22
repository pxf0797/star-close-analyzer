"""
平仓决策引擎 — 基于冻结轨迹 + 残差 + Sigmoid 贴合度 + 半衰期衰减。

五条平仓铁律：
1. 利润硬门槛 — 理论止盈价扣除手续费后须为正收益
2. 动态引力挂单 — 贴合度高时死守极值价，贴合度低时保守退让
3. 绝不重算 — 自信度跌破阈值直接市价全平
4. All or Nothing — 触发平仓一次性出清
5. 拔网线止损 — 开仓瞬间反向剧烈拉伸直接市价全平
"""

import math
import enum
from dataclasses import dataclass, field
import numpy as np
from numpy.polynomial import Polynomial


class Action(enum.Enum):
    HOLD = "hold"
    UPSERT_PROTECTION = "upsert_protection"
    MARKET_CLOSE_ALL = "market_close_all"


class Direction(enum.Enum):
    ALIGNED = "aligned"        # 贴合
    FAVORABLE = "favorable"    # 有利突破（做空：实际价低于预测价，下跌超预期）
    ADVERSE = "adverse"        # 不利突破（做空：实际价高于预测价，上涨回撤）


@dataclass
class CloseDecision:
    action: Action
    protection_side: str = ""          # "take_profit" | "stop_loss"
    trigger_price: float = 0.0
    quantity: float = 0.0
    reason_code: str = ""
    fit_score: float = 1.0
    direction: Direction = Direction.ALIGNED
    residual: float = 0.0
    pred_price: float = 0.0
    actual_price: float = 0.0

    def to_dict(self) -> dict:
        return {
            "action": self.action.value,
            "protection_side": self.protection_side,
            "trigger_price": round(self.trigger_price, 6),
            "quantity": round(self.quantity, 6),
            "reason_code": self.reason_code,
            "fit_score": round(self.fit_score, 4),
            "direction": self.direction.value,
            "residual": round(self.residual, 6),
            "pred_price": round(self.pred_price, 6),
            "actual_price": round(self.actual_price, 6),
        }


@dataclass
class PositionState:
    """做空持仓状态"""
    entry_price: float
    entry_idx: int
    quantity: float
    anchor: Polynomial                          # 冻结的三次多项式
    theoretical_tp: float | None = None         # 理论止盈价
    fee_rate: float = 0.0008                    # 双边手续费率
    half_life: int = 15                         # 半衰期（bar 数）
    max_relative_distance: float = 0.05         # 保护单最大距离 5%
    confidence_threshold: float = 0.3           # 自信度跌破阈值 → 全平
    hard_stop_multiplier: float = 2.0           # 2×tol 硬止损


def sigmoid(x: float, k: float = 1.0) -> float:
    """Sigmoid 函数，输出 [0, 1]"""
    return 1.0 / (1.0 + math.exp(-k * x))


def compute_residual(p_pred: float, p_act: float) -> float:
    """对数残差（无量纲）: ln(P_pred / P_act)"""
    if p_act <= 0 or p_pred <= 0:
        return 0.0
    return math.log(p_pred / p_act)


def compute_fit_score(residual: float, k: float = 10.0) -> float:
    """
    贴合度 fit_score ∈ (0, 1]。
    对正负残差对称：fit = 2 / (1 + exp(k * |residual|)) - 1 → (0, 1)
    实际上用双 sigmoid 归一化: fit = 2 * sigmoid(-k * |residual|)
    残差 → 0 时贴合度 → 1；|残差| 大时贴合度 → 0。
    """
    return 2.0 * sigmoid(-k * abs(residual))


def compute_tolerance(fee_rate: float, local_volatility: float) -> float:
    """
    容忍带宽 = max(手续费率, 局部波动率)
    这里用对数空间，均无量纲。
    """
    return max(fee_rate, local_volatility)


def half_life_decay(t: int, half_life: int) -> float:
    """半衰期衰减因子: 2^(-t / half_life)，t 为持仓 bar 数"""
    return math.pow(2, -t / max(half_life, 1))


def evaluate_close(
    position: PositionState,
    current_idx: int,
    actual_price: float,
    local_volatility: float = 0.002,
) -> CloseDecision:
    """
    每个 tick 对做空持仓输出平仓决策。

    优先级（高 → 低）：
    1. 拔网线止损 — 首根 K 线反向剧烈拉伸
    2. 硬止损 — 不利偏离 > max(2*tol, max_relative_distance)
    3. 自信度崩盘 — fit_score < confidence_threshold
    4. 更新保护单（止盈或止损）
    5. 持有
    """

    t = current_idx - position.entry_idx
    if t < 0:
        return CloseDecision(action=Action.HOLD)

    poly = position.anchor
    pred_price = float(poly(t))
    residual = compute_residual(pred_price, actual_price)
    fit_score = compute_fit_score(residual)

    # 偏离方向（做空视角）
    if abs(residual) < 0.002:
        direction = Direction.ALIGNED
    elif residual < 0:
        # residual < 0 → ln(P_pred/P_act) < 0 → P_act > P_pred
        # 实际价高于预测 → 做空不利
        direction = Direction.ADVERSE
    else:
        # residual > 0 → P_act < P_pred → 实际价低于预测
        # 下跌超预期 → 做空有利
        direction = Direction.FAVORABLE

    tol = compute_tolerance(position.fee_rate, local_volatility)
    hard_limit = max(position.hard_stop_multiplier * tol, position.max_relative_distance)

    # 优先级 1: 拔网线止损（首几根 K 线瞬间反向拉伸）
    if t <= 3 and residual < -hard_limit:
        return CloseDecision(
            action=Action.MARKET_CLOSE_ALL,
            reason_code="emergency_stop",
            fit_score=fit_score,
            direction=direction,
            residual=residual,
            pred_price=pred_price,
            actual_price=actual_price,
            quantity=position.quantity,
        )

    # 优先级 2: 硬止损
    if residual < -hard_limit:
        return CloseDecision(
            action=Action.MARKET_CLOSE_ALL,
            reason_code="hard_stop_adverse",
            fit_score=fit_score,
            direction=direction,
            residual=residual,
            pred_price=pred_price,
            actual_price=actual_price,
            quantity=position.quantity,
        )

    # 优先级 3: 自信度崩盘
    if fit_score < position.confidence_threshold:
        return CloseDecision(
            action=Action.MARKET_CLOSE_ALL,
            reason_code="confidence_collapse",
            fit_score=fit_score,
            direction=direction,
            residual=residual,
            pred_price=pred_price,
            actual_price=actual_price,
            quantity=position.quantity,
        )

    # 优先级 4: 更新保护单
    decay = half_life_decay(t, position.half_life)

    if direction == Direction.FAVORABLE or direction == Direction.ALIGNED:
        # 止盈保护单：目标价 = 理论极值价 × fit_score 衰减
        if position.theoretical_tp is not None:
            # 贴合度高 → 靠近理论极值；贴合度低 → 保守退让
            adjusted_tp = position.theoretical_tp + (actual_price - position.theoretical_tp) * (1 - fit_score)
            adjusted_tp = max(adjusted_tp, actual_price * 0.95)  # 不超过 5%
        else:
            adjusted_tp = actual_price * (1 - position.max_relative_distance * fit_score)

        return CloseDecision(
            action=Action.UPSERT_PROTECTION,
            protection_side="take_profit",
            trigger_price=adjusted_tp,
            quantity=position.quantity,
            reason_code="update_tp",
            fit_score=fit_score,
            direction=direction,
            residual=residual,
            pred_price=pred_price,
            actual_price=actual_price,
        )
    else:
        # 保护止损单
        stop_price = actual_price * (1 + tol * fit_score * decay)
        return CloseDecision(
            action=Action.UPSERT_PROTECTION,
            protection_side="stop_loss",
            trigger_price=min(stop_price, actual_price * (1 + position.max_relative_distance)),
            quantity=position.quantity,
            reason_code="update_sl",
            fit_score=fit_score,
            direction=direction,
            residual=residual,
            pred_price=pred_price,
            actual_price=actual_price,
        )


def check_profit_threshold(
    entry_price: float,
    theoretical_tp: float | None,
    fee_rate: float,
) -> bool:
    """铁律 1：理论止盈须覆盖双边手续费。做空视角：entry > tp 才有利润。"""
    if theoretical_tp is None:
        return False
    if theoretical_tp >= entry_price:
        return False
    profit_ratio = (entry_price - theoretical_tp) / entry_price
    return profit_ratio > 2 * fee_rate
