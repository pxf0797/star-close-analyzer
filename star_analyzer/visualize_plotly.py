"""
Plotly 交互图表模块 — hover/zoom/框选的回测可视化。
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from star_analyzer.backtest import BacktestResult
from star_analyzer.trajectory import fit_cubic_trajectory, find_theoretical_take_profit

# 统一调色板
C_PRICE    = "#1a1a2e"
C_TRAJ     = "#8e44ad"
C_TP       = "#27ae60"
C_WIN      = "#27ae60"
C_LOSS     = "#e74c3c"
C_FIT      = "#2980b9"
C_FIT_BG   = "rgba(52,152,219,0.15)"
C_EQUITY   = "#2c3e50"
C_EQUITY_BG= "rgba(44,62,80,0.1)"
C_ALIGNED  = "#7f8c8d"
C_FAVORABLE= "#27ae60"
C_ADVERSE  = "#e74c3c"
C_SIGNAL   = {2: "#8e44ad", 1: "#3498db", 0: "rgba(0,0,0,0.03)", -1: "#e74c3c"}
SIGNAL_LABEL = {2: "开仓", 1: "保护单", 0: "", -1: "全平"}
RESID_COLORS = {"aligned": C_ALIGNED, "favorable": C_FAVORABLE, "adverse": C_ADVERSE}


def build_interactive_chart(
    prices: np.ndarray,
    result: BacktestResult,
    title: str = "星空策略 · 轨迹冻结平仓分析",
) -> go.Figure:
    """3 面板: (1)价格+信号条 (2)残差+贴合度双轴 (3)权益曲线"""

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.45, 0.30, 0.25],
        subplot_titles=(title, "残差 & 贴合度", "权益曲线"),
        specs=[[{"secondary_y": False}],
               [{"secondary_y": True}],
               [{"secondary_y": False}]],
    )

    x = np.arange(len(prices))

    # ═══════════ Panel 1: 价格 + 轨迹 + 信号色带 ═══════════
    # 信号背景色带（极细，叠在底部）
    signal_map = {"open_short": 2, "hold": 0, "upsert_protection": 1, "market_close_all": -1}
    signals = np.array([signal_map.get(r.action, 0) for r in result.records])
    sig_colors = [C_SIGNAL[s] for s in signals]

    fig.add_trace(
        go.Bar(x=x, y=np.ones(len(x)), marker_color=sig_colors,
               name="Signal", showlegend=False, width=1.0,
               hovertemplate="idx=%{x}<br>%{customdata}<extra></extra>",
               customdata=[SIGNAL_LABEL[s] for s in signals],
               marker_line_width=0),
        row=1, col=1,
    )

    # 价格线
    fig.add_trace(
        go.Scatter(x=x, y=prices, mode="lines", name="Price",
                   line=dict(color=C_PRICE, width=0.9),
                   hovertemplate="idx=%{x}<br>Price=%{y:.2f}<extra></extra>"),
        row=1, col=1,
    )

    # 持仓区间着色
    for trade in result.trades:
        color = "rgba(39,174,96,0.06)" if trade.pnl > 0 else "rgba(231,76,60,0.06)"
        fig.add_vrect(x0=trade.entry_idx, x1=trade.exit_idx, fillcolor=color,
                       layer="below", line_width=0, row=1, col=1)

    # 轨迹 + 入场 + 止盈
    for rec in result.records:
        if rec.action == "open_short":
            poly = fit_cubic_trajectory(prices, rec.idx)
            if poly is None:
                continue
            max_t = 200
            for t in result.trades:
                if t.entry_idx == rec.idx:
                    max_t = min(t.exit_idx - rec.idx + 20, 50)
                    break
            t_traj = np.arange(0, min(len(prices) - rec.idx, max_t))
            traj_y = poly(t_traj)
            traj_x = rec.idx + t_traj

            fig.add_trace(go.Scatter(
                x=traj_x, y=traj_y, mode="lines", name=f"Traj #{rec.idx}",
                line=dict(dash="dash", color=C_TRAJ, width=1.2),
                hovertemplate="Pred=%{y:.2f}<extra></extra>", showlegend=False,
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=[rec.idx], y=[rec.price], mode="markers", name=f"Entry #{rec.idx}",
                marker=dict(symbol="diamond", size=10, color=C_TRAJ, line=dict(width=1, color="white")),
                hovertemplate="Entry #%{x}<br>%{y:.2f}<extra></extra>", showlegend=False,
            ), row=1, col=1)

            tp = find_theoretical_take_profit(poly)
            if tp is not None:
                tp_t = rec.idx + np.argmin(np.abs(poly(t_traj) - tp))
                fig.add_trace(go.Scatter(
                    x=[tp_t], y=[tp], mode="markers", name=f"TP #{rec.idx}",
                    marker=dict(symbol="star", size=14, color=C_TP, line=dict(width=1, color="white")),
                    hovertemplate="TP=%{y:.2f}<extra></extra>", showlegend=False,
                ), row=1, col=1)

    # 出场标注
    for trade in result.trades:
        clr = C_WIN if trade.pnl > 0 else C_LOSS
        sym = "triangle-down" if trade.pnl > 0 else "triangle-up"
        fig.add_trace(go.Scatter(
            x=[trade.exit_idx], y=[trade.exit_price], mode="markers",
            name=f"Exit #{trade.entry_idx}",
            marker=dict(symbol=sym, size=10, color=clr, line=dict(width=1, color="white")),
            customdata=[f"{trade.pnl_pct:+.2f}%"],
            hovertemplate="Exit #%{x}<br>PnL=%{customdata}<extra></extra>", showlegend=False,
        ), row=1, col=1)

    # ═══════════ Panel 2: 残差(bar) + 贴合度(line) 双Y轴 ═══════════
    active_recs = [r for r in result.records if r.action != "hold"]
    resid_x = [r.idx for r in active_recs]
    resid_y = [r.residual for r in active_recs]
    resid_colors = [RESID_COLORS.get(r.direction, C_ALIGNED) for r in active_recs]

    fig.add_trace(
        go.Bar(x=resid_x, y=resid_y, marker_color=resid_colors,
               name="残差", showlegend=True,
               hovertemplate="idx=%{x}<br>residual=%{y:.6f}<extra></extra>"),
        row=2, col=1, secondary_y=False,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="black", line_width=0.5, row=2, col=1)

    fit_recs = [r for r in result.records if r.action not in ("hold", "open_short")]
    if fit_recs:
        fit_x = [r.idx for r in fit_recs]
        fit_y = [r.fit_score for r in fit_recs]
        fig.add_trace(
            go.Scatter(x=fit_x, y=fit_y, mode="lines", name="贴合度",
                       line=dict(color=C_FIT, width=1.8),
                       hovertemplate="idx=%{x}<br>fit=%{y:.4f}<extra></extra>"),
            row=2, col=1, secondary_y=True,
        )
    fig.add_hline(y=0.3, line_dash="dash", line_color=C_LOSS, line_width=0.8,
                   annotation_text="崩盘阈值", annotation_position="bottom left",
                   row=2, col=1, secondary_y=True)

    fig.update_yaxes(title_text="残差", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="贴合度", range=[-0.05, 1.1], row=2, col=1, secondary_y=True)

    # ═══════════ Panel 3: 权益曲线 ═══════════
    equity = np.array(result.equity_curve)
    eq_x = np.arange(len(equity)) / len(prices) * 100
    fig.add_trace(
        go.Scatter(x=eq_x, y=equity, mode="lines", name="Equity",
                   line=dict(color=C_EQUITY, width=1.5),
                   fill="tozeroy", fillcolor=C_EQUITY_BG,
                   hovertemplate="进度=%{x:.1f}%<br>Equity=%{y:.2f}<extra></extra>"),
        row=3, col=1,
    )
    fig.add_hline(y=result.equity_curve[0], line_dash="dot", line_color="gray",
                   line_width=0.5, row=3, col=1)

    # 权益曲线上标记交易起止
    for trade in result.trades:
        ex = (trade.exit_idx / len(prices)) * 100
        clr = C_WIN if trade.pnl > 0 else C_LOSS
        fig.add_trace(go.Scatter(
            x=[ex], y=[result.equity_curve[0]], mode="markers",
            marker=dict(symbol="line-ns", size=8, color=clr, line=dict(width=1)),
            showlegend=False, hovertemplate="Exit #%{x}<extra></extra>",
        ), row=3, col=1)

    # Layout
    fig.update_xaxes(title_text="K 线索引", row=2, col=1)
    fig.update_xaxes(title_text="进度 %", row=3, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Equity", row=3, col=1)

    fig.update_layout(
        height=750,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=60, r=30, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=10)),
    )

    return fig


def build_trajectory_detail_plotly(
    prices: np.ndarray,
    entry_idx: int,
) -> go.Figure | None:
    """单次轨迹拟合详情（拟合窗口 + 导数分析）"""

    poly = fit_cubic_trajectory(prices, entry_idx)
    if poly is None:
        return None

    v0 = poly.deriv(1)(0)
    a0 = poly.deriv(2)(0)
    entry_price = prices[entry_idx]

    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        f"Entry #{entry_idx}  |  Price={entry_price:.4f}",
        f"V = f'(0) = {v0:.6f}   |   A = f''(0) = {a0:.6f}  {'(局部极大 ✅)' if a0 < 0 else '(⚠️)'}"
    ))

    half = 15
    start = max(0, entry_idx - half)
    end = min(len(prices), entry_idx + half)
    t_range = np.arange(start, end)

    fig.add_trace(
        go.Scatter(x=t_range, y=prices[start:end], mode="markers+lines",
                   name="Price", marker=dict(size=4, color=C_PRICE),
                   line=dict(width=1, color=C_PRICE)),
        row=1, col=1,
    )
    t_fit = np.arange(start - entry_idx, end - entry_idx, dtype=float)
    fig.add_trace(
        go.Scatter(x=t_range, y=poly(t_fit), mode="lines",
                   name="Cubic Fit", line=dict(width=2, color=C_TRAJ)),
        row=1, col=1,
    )
    fig.add_vline(x=entry_idx, line_dash="dash", line_color=C_TRAJ, annotation_text="Entry", row=1, col=1)

    tp = find_theoretical_take_profit(poly)
    if tp is not None:
        tp_t = entry_idx + int(np.argmin(np.abs(poly(np.arange(0, 200)) - tp)))
        fig.add_trace(
            go.Scatter(x=[tp_t], y=[tp], mode="markers", name=f"TP: {tp:.4f}",
                       marker=dict(symbol="star", size=16, color=C_TP, line=dict(width=1, color="white"))),
            row=1, col=1,
        )

    # 导数
    t_ext = np.linspace(-half, half, 200)
    fig.add_trace(
        go.Scatter(x=t_ext, y=poly.deriv(1)(t_ext), mode="lines",
                   name="f'(t) 速度", line=dict(width=1.5, color=C_FIT)),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(x=t_ext, y=poly.deriv(2)(t_ext), mode="lines",
                   name="f''(t) 加速度", line=dict(width=1.5, color="#e67e22")),
        row=1, col=2,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="black", line_width=0.5, row=1, col=2)
    fig.add_vline(x=0, line_dash="dash", line_color=C_TRAJ, line_width=0.8, row=1, col=2)

    fig.update_yaxes(range=[entry_price * 0.9, entry_price * 1.1], row=1, col=1)
    fig.update_layout(height=400, template="plotly_white", hovermode="x unified")
    return fig


def build_replay_chart(prices: np.ndarray, result: BacktestResult, current_idx: int) -> go.Figure:
    """Tick 回放 — 2 面板: (1)价格+轨迹 (2)残差+贴合度+信号"""

    # 找到当前持仓获取 V/A 和浮盈
    active_trade = None
    for trade in result.trades:
        if trade.entry_idx <= current_idx < trade.exit_idx:
            active_trade = trade
            break

    va_info = ""
    if active_trade:
        poly = fit_cubic_trajectory(prices, active_trade.entry_idx)
        if poly is not None:
            v0 = poly.deriv(1)(0)
            a0 = poly.deriv(2)(0)
            unrealized = (active_trade.entry_price - prices[current_idx]) / active_trade.entry_price * 100
            va_info = f"  |  V={v0:.4f}  A={a0:.4f}  浮盈 {unrealized:+.2f}%  [Entry #{active_trade.entry_idx}]"

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.55, 0.45],
        subplot_titles=(
            f"tick={current_idx}  Price={prices[current_idx]:.2f}{va_info}",
            "残差 & 贴合度",
        ),
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]],
    )

    x = np.arange(min(current_idx + 1, len(prices)))

    # ═══ Panel 1: 价格 + 轨迹 ═══
    fig.add_trace(
        go.Scatter(x=x, y=prices[:current_idx + 1], mode="lines",
                   name="Price", line=dict(color=C_PRICE, width=0.9),
                   hovertemplate="idx=%{x}<br>Price=%{y:.2f}<extra></extra>"),
        row=1, col=1,
    )

    entries_before = [r for r in result.records if r.action == "open_short" and r.idx <= current_idx]
    exits_before = [t for t in result.trades if t.exit_idx <= current_idx]

    for rec in entries_before:
        poly = fit_cubic_trajectory(prices, rec.idx)
        if poly is None:
            continue
        t_end = min(len(prices) - rec.idx, current_idx - rec.idx + 10, 40)
        if t_end <= 0:
            continue
        t_traj = np.arange(0, t_end)
        fig.add_trace(
            go.Scatter(x=rec.idx + t_traj, y=poly(t_traj), mode="lines",
                       name=f"Traj #{rec.idx}",
                       line=dict(dash="dash", color=C_TRAJ, width=1), showlegend=False),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=[rec.idx], y=[rec.price], mode="markers",
                       marker=dict(symbol="diamond", size=8, color=C_TRAJ, line=dict(width=1, color="white")),
                       showlegend=False),
            row=1, col=1,
        )

    for trade in exits_before:
        clr = C_WIN if trade.pnl > 0 else C_LOSS
        fig.add_trace(
            go.Scatter(x=[trade.exit_idx], y=[trade.exit_price], mode="markers",
                       marker=dict(symbol="triangle-down" if trade.pnl > 0 else "triangle-up",
                                   size=8, color=clr, line=dict(width=1, color="white")),
                       showlegend=False),
            row=1, col=1,
        )

    # 当前 tick 垂直线 + 持仓区间着色
    fig.add_vline(x=current_idx, line_dash="dot", line_color=C_LOSS, line_width=1.5, row=1, col=1)
    if active_trade:
        fig.add_vrect(x0=active_trade.entry_idx, x1=current_idx,
                       fillcolor="rgba(142,68,173,0.06)", layer="below", line_width=0, row=1, col=1)

    fig.update_yaxes(range=[prices[current_idx] * 0.95, prices[current_idx] * 1.05], row=1, col=1)

    # ═══ Panel 2: 残差(bar) + 贴合度(line) 双Y轴 ═══
    recs_up_to = [r for r in result.records if r.idx <= current_idx and r.action != "hold"]
    res_x = [r.idx for r in recs_up_to]
    res_y = [r.residual for r in recs_up_to]
    res_colors = [RESID_COLORS.get(r.direction, C_ALIGNED) for r in recs_up_to]

    fig.add_trace(
        go.Bar(x=res_x, y=res_y, marker_color=res_colors, name="残差",
               showlegend=False, hovertemplate="idx=%{x}<br>%{y:.6f}<extra></extra>"),
        row=2, col=1, secondary_y=False,
    )

    fit_recs = [r for r in recs_up_to if r.action not in ("open_short",)]
    if fit_recs:
        fit_x = [r.idx for r in fit_recs]
        fit_y = [r.fit_score for r in fit_recs]
        fig.add_trace(
            go.Scatter(x=fit_x, y=fit_y, mode="lines+markers", name="贴合度",
                       marker=dict(size=3, color=C_FIT),
                       line=dict(width=1.5, color=C_FIT),
                       hovertemplate="idx=%{x}<br>fit=%{y:.4f}<extra></extra>"),
            row=2, col=1, secondary_y=True,
        )

    fig.add_hline(y=0, line_dash="dot", line_color="black", line_width=0.5, row=2, col=1)
    fig.add_hline(y=0.3, line_dash="dash", line_color=C_LOSS, line_width=0.5, row=2, col=1)
    fig.add_vline(x=current_idx, line_dash="dot", line_color=C_LOSS, line_width=1, row=2, col=1)

    fig.update_yaxes(title_text="残差", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="贴合度", range=[-0.05, 1.1], row=2, col=1, secondary_y=True)

    fig.update_layout(
        height=500,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=50, r=50, t=50, b=30),
    )
    return fig
