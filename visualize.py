"""
可视化模块 — 绘制价格轨迹、预测曲线、残差、贴合度与平仓信号。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
from typing import Any

from backtest import BacktestResult, TickRecord
from trajectory import fit_cubic_trajectory, find_theoretical_take_profit


plt.rcParams["font.family"] = ["Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def plot_analysis(
    prices: np.ndarray,
    result: BacktestResult,
    save_path: str = "star_analysis.png",
    title: str = "星空策略 · 轨迹冻结平仓分析",
    entry_idx: int | None = None,
):
    """主可视化：价格+轨迹、残差+贴合度、平仓信号、权益曲线"""

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(5, 1, height_ratios=[3, 1.5, 1.5, 0.8, 1.5], hspace=0.35)

    ax_price = fig.add_subplot(gs[0])
    ax_residual = fig.add_subplot(gs[1], sharex=ax_price)
    ax_fit = fig.add_subplot(gs[2], sharex=ax_price)
    ax_signal = fig.add_subplot(gs[3], sharex=ax_price)
    ax_equity = fig.add_subplot(gs[4])

    x = np.arange(len(prices))

    # === 1. 价格与轨迹 ===
    ax_price.plot(x, prices, color="#1a1a2e", linewidth=0.8, alpha=0.7, label="Price")
    ax_price.fill_between(x, prices, alpha=0.05, color="#1a1a2e")

    # 标注交易
    colors_trade = {"hard_stop_adverse": "#e74c3c", "confidence_collapse": "#e67e22",
                     "emergency_stop": "#c0392b", "force_close_eod": "#95a5a6", "update_tp": "#27ae60"}
    for trade in result.trades:
        ax_price.axvspan(trade.entry_idx, trade.exit_idx, alpha=0.08,
                           color="red" if trade.pnl <= 0 else "green")
        marker = "v" if trade.pnl > 0 else "^"
        clr = "#27ae60" if trade.pnl > 0 else "#e74c3c"
        ax_price.scatter(trade.exit_idx, trade.exit_price, c=clr, s=80, zorder=5,
                          marker=marker, edgecolors="white", linewidth=0.5)
        ax_price.annotate(
            f"{trade.reason}\n{trade.pnl_pct:+.2f}%",
            (trade.exit_idx, trade.exit_price),
            textcoords="offset points", xytext=(8, 15 if trade.pnl > 0 else -20),
            fontsize=7, color=clr, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor=clr, linewidth=0.5),
        )

    # 画轨迹曲线（如果有开仓）
    for rec in result.records:
        if rec.action == "open_short":
            poly = fit_cubic_trajectory(prices, rec.idx)
            if poly is not None:
                t_traj = np.arange(0, min(len(prices) - rec.idx, 200))
                traj_y = poly(t_traj)
                ax_price.plot(rec.idx + t_traj, traj_y, "--", color="#8e44ad",
                               linewidth=1.5, alpha=0.6)
                ax_price.scatter(rec.idx, rec.price, c="#8e44ad", s=100, zorder=6,
                                  marker="D", edgecolors="white", linewidth=0.8)

                tp = find_theoretical_take_profit(poly)
                if tp is not None:
                    tp_t = np.argmin(np.abs(poly(t_traj) - tp))
                    ax_price.scatter(rec.idx + tp_t, tp, c="#27ae60", s=120, zorder=6,
                                      marker="*", edgecolors="white", linewidth=0.5)
                    ax_price.annotate(f"TP: {tp:.4f}", (rec.idx + tp_t, tp),
                                       textcoords="offset points", xytext=(5, -15),
                                       fontsize=8, color="#27ae60", fontweight="bold")

    ax_price.set_ylabel("Price", fontsize=11)
    ax_price.set_title(title, fontsize=14, fontweight="bold")
    ax_price.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax_price.grid(True, alpha=0.2)

    # === 2. 残差 ===
    idxs = [r.idx for r in result.records if r.action != "hold"]
    residuals = [r.residual for r in result.records if r.action != "hold"]
    directions = [r.direction for r in result.records if r.action != "hold"]
    color_map = {"aligned": "#7f8c8d", "favorable": "#27ae60", "adverse": "#e74c3c"}
    bar_colors = [color_map.get(d, "#7f8c8d") for d in directions]
    ax_residual.bar(idxs, residuals, color=bar_colors, alpha=0.7, width=1.2)
    ax_residual.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax_residual.fill_between(x, -0.002, 0.002, alpha=0.1, color="#7f8c8d")
    ax_residual.set_ylabel("Residual\nln(P_pred/P_act)", fontsize=10)
    ax_residual.set_title("预测残差（绿=有利 红=不利 灰=贴合）", fontsize=10, color="#555")
    ax_residual.grid(True, alpha=0.2)

    # === 3. 贴合度 ===
    fit_idxs = [r.idx for r in result.records if r.action not in ("hold", "open_short")]
    fit_vals = [r.fit_score for r in result.records if r.action not in ("hold", "open_short")]
    if fit_vals:
        ax_fit.fill_between(fit_idxs, fit_vals, alpha=0.2, color="#3498db")
        ax_fit.plot(fit_idxs, fit_vals, color="#2980b9", linewidth=1.5)
    ax_fit.axhline(0.3, color="#e74c3c", linewidth=0.8, linestyle="--", alpha=0.6)
    ax_fit.annotate("崩盘阈值 0.3", (x[0], 0.3), textcoords="offset points",
                     xytext=(5, 3), fontsize=7, color="#e74c3c")
    ax_fit.set_ylabel("Fit Score", fontsize=10)
    ax_fit.set_title("贴合度 (0=完全偏离, 1=完美贴合)", fontsize=10, color="#555")
    ax_fit.set_ylim(-0.05, 1.1)
    ax_fit.grid(True, alpha=0.2)

    # === 4. 信号热力图 ===
    actions = [r.action for r in result.records]
    signal_map = {"open_short": 2, "hold": 0, "upsert_protection": 1, "market_close_all": -1}
    signal_colors = {2: "#8e44ad", 1: "#3498db", 0: "#ecf0f1", -1: "#e74c3c"}
    signals = np.array([signal_map.get(a, 0) for a in actions])
    ax_signal.bar(x, signals, color=[signal_colors[s] for s in signals], width=1.2, alpha=0.8)
    ax_signal.set_ylabel("Signal", fontsize=9)
    ax_signal.set_title("信号 (紫=开仓 蓝=保护单 红=全平 灰=持有)", fontsize=9, color="#555")
    ax_signal.set_yticks([-1, 0, 1, 2])
    ax_signal.set_yticklabels(["全平", "持有", "保护单", "开仓"], fontsize=7)
    ax_signal.grid(True, alpha=0.2, axis="y")

    # === 5. 权益曲线 ===
    equity = np.array(result.equity_curve)
    ax_equity.plot(np.arange(len(equity)) / len(prices) * 100, equity, color="#2c3e50", linewidth=1.5)
    ax_equity.fill_between(np.arange(len(equity)) / len(prices) * 100, equity, result.equity_curve[0],
                            alpha=0.15, color="#2c3e50")
    ax_equity.axhline(result.equity_curve[0], color="gray", linewidth=0.5, linestyle="--")
    ax_equity.set_xlabel("进度 %", fontsize=10)
    ax_equity.set_ylabel("Equity", fontsize=10)
    ax_equity.set_title("权益曲线", fontsize=11)
    ax_equity.grid(True, alpha=0.2)

    # 统计信息文本框
    stats = result.stats
    stats_text = (
        f"总交易: {stats.get('total_trades', 0)} | "
        f"胜率: {stats.get('win_rate', 0)}% | "
        f"总盈亏: {stats.get('total_pnl_pct', 0):.2f}% | "
        f"盈亏比: {stats.get('profit_factor', 'N/A')} | "
        f"最大回撤: {stats.get('max_drawdown', 0):.1f}%"
    )
    fig.text(0.5, 0.01, stats_text, ha="center", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#dee2e6"))

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return save_path


def plot_trajectory_detail(prices: np.ndarray, entry_idx: int, save_path: str = "trajectory_detail.png"):
    """绘制单次开仓的轨迹拟合细节"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 左侧：拟合窗口
    half = 15
    start = max(0, entry_idx - half)
    end = min(len(prices), entry_idx + half)
    t_range = np.arange(start, end)
    ax1.plot(t_range, prices[start:end], "o-", markersize=3, color="#1a1a2e", linewidth=1, label="Price")
    ax1.axvline(entry_idx, color="#8e44ad", linestyle="--", linewidth=1.2, label="Entry")

    poly = fit_cubic_trajectory(prices, entry_idx)
    if poly is not None:
        t_fit = np.arange(start - entry_idx, end - entry_idx, dtype=float)
        ax1.plot(t_range, poly(t_fit), "-", color="#e74c3c", linewidth=2, alpha=0.8, label="Cubic Fit")

        tp = find_theoretical_take_profit(poly)
        if tp is not None:
            tp_t_approx = entry_idx + int(np.argmin(np.abs(poly(np.arange(0, 200)) - tp)))
            ax1.scatter(tp_t_approx, tp, c="#27ae60", s=150, zorder=6, marker="*",
                         edgecolors="white", linewidth=0.8, label=f"TP: {tp:.4f}")

    ax1.set_title(f"Entry at idx={entry_idx}, Price={prices[entry_idx]:.4f}", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)

    # 右侧：导数分析
    if poly is not None:
        t_ext = np.linspace(-half, half, 200)
        d1 = poly.deriv(1)(t_ext)
        d2 = poly.deriv(2)(t_ext)

        ax2.plot(t_ext, d1, color="#2980b9", linewidth=1.5, label="一阶导 f'(t)")
        ax2.plot(t_ext, d2, color="#e67e22", linewidth=1.5, label="二阶导 f''(t)")
        ax2.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax2.axvline(0, color="#8e44ad", linestyle="--", linewidth=0.8, alpha=0.5)

        ax2.set_title(f"f'(0)={poly.deriv(1)(0):.6f}, f''(0)={poly.deriv(2)(0):.6f} (< 0)", fontsize=11, fontweight="bold")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return save_path
