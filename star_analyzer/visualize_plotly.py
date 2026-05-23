"""
Plotly 交互图表模块 — hover/zoom/框选的 5 面板回测可视化。
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from star_analyzer.backtest import BacktestResult
from star_analyzer.trajectory import fit_cubic_trajectory, find_theoretical_take_profit


def build_interactive_chart(
    prices: np.ndarray,
    result: BacktestResult,
    title: str = "星空策略 · 轨迹冻结平仓分析",
) -> go.Figure:
    """
    构建 5 面板 Plotly 交互图表：
    (1) 价格+轨迹 (2) 残差 (3) 贴合度 (4) 信号 (5) 权益曲线
    面板 1-4 共享 x 轴（K 线索引），面板 5 独立。
    """

    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.35, 0.17, 0.17, 0.10, 0.21],
        subplot_titles=(title, "预测残差 ln(P_pred/P_act)", "贴合度 Fit Score",
                        "信号", "权益曲线"),
    )

    x = np.arange(len(prices))

    # ═══════════════════════════════════════
    # Panel 1: 价格 + 轨迹 + 入场/出场标注
    # ═══════════════════════════════════════
    fig.add_trace(
        go.Scatter(x=x, y=prices, mode="lines", name="Price",
                   line=dict(color="#1a1a2e", width=0.8),
                   hovertemplate="idx=%{x}<br>Price=%{y:.2f}<extra></extra>"),
        row=1, col=1,
    )

    # 轨迹曲线 & 入场点
    for rec in result.records:
        if rec.action == "open_short":
            poly = fit_cubic_trajectory(prices, rec.idx)
            if poly is not None:
                max_t = 200
                for t in result.trades:
                    if t.entry_idx == rec.idx:
                        max_t = min(t.exit_idx - rec.idx + 20, 50)
                        break
                t_traj = np.arange(0, min(len(prices) - rec.idx, max_t))
                traj_y = poly(t_traj)
                traj_x = rec.idx + t_traj
                fig.add_trace(
                    go.Scatter(x=traj_x, y=traj_y, mode="lines", name=f"Trajectory #{rec.idx}",
                               line=dict(dash="dash", color="#8e44ad", width=1.2),
                               hovertemplate="Pred=%{y:.2f}<extra></extra>",
                               showlegend=False),
                    row=1, col=1,
                )
                # 入场菱形
                fig.add_trace(
                    go.Scatter(x=[rec.idx], y=[rec.price], mode="markers",
                               name=f"Entry #{rec.idx}",
                               marker=dict(symbol="diamond", size=10, color="#8e44ad",
                                           line=dict(width=1, color="white")),
                               hovertemplate="Entry #%{x}<br>Price=%{y:.2f}<extra></extra>",
                               showlegend=False),
                    row=1, col=1,
                )
                # 理论止盈星
                tp = find_theoretical_take_profit(poly)
                if tp is not None:
                    tp_t = rec.idx + np.argmin(np.abs(poly(t_traj) - tp))
                    fig.add_trace(
                        go.Scatter(x=[tp_t], y=[tp], mode="markers",
                                   name=f"TP #{rec.idx}",
                                   marker=dict(symbol="star", size=14, color="#27ae60",
                                               line=dict(width=1, color="white")),
                                   hovertemplate="TP=%{y:.2f}<extra></extra>",
                                   showlegend=False),
                        row=1, col=1,
                    )

    # 出场标注
    for trade in result.trades:
        color = "#27ae60" if trade.pnl > 0 else "#e74c3c"
        symbol = "triangle-down" if trade.pnl > 0 else "triangle-up"
        fig.add_trace(
            go.Scatter(x=[trade.exit_idx], y=[trade.exit_price], mode="markers",
                       name=f"Exit #{trade.entry_idx}",
                       marker=dict(symbol=symbol, size=10, color=color,
                                   line=dict(width=1, color="white")),
                       hovertemplate="Exit #%{x}<br>PnL=%{customdata}<extra></extra>",
                       customdata=[f"{trade.pnl_pct:+.2f}%"],
                       showlegend=False),
            row=1, col=1,
        )

    # 持仓区间着色
    for trade in result.trades:
        color = "rgba(39,174,96,0.08)" if trade.pnl > 0 else "rgba(231,76,60,0.08)"
        fig.add_vrect(x0=trade.entry_idx, x1=trade.exit_idx, fillcolor=color,
                       layer="below", line_width=0, row=1, col=1)

    # ═══════════════════════════════════════
    # Panel 2: 残差柱状图
    # ═══════════════════════════════════════
    active_records = [r for r in result.records if r.action != "hold"]
    resid_x = [r.idx for r in active_records]
    resid_y = [r.residual for r in active_records]
    color_map = {"aligned": "#7f8c8d", "favorable": "#27ae60", "adverse": "#e74c3c"}
    resid_colors = [color_map.get(r.direction, "#7f8c8d") for r in active_records]
    resid_dir = [r.direction for r in active_records]

    fig.add_trace(
        go.Bar(x=resid_x, y=resid_y, marker_color=resid_colors,
               name="Residual", showlegend=False,
               customdata=resid_dir,
               hovertemplate="idx=%{x}<br>residual=%{y:.6f}<br>%{customdata}<extra></extra>"),
        row=2, col=1,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="black", line_width=0.5, row=2, col=1)

    # ═══════════════════════════════════════
    # Panel 3: 贴合度
    # ═══════════════════════════════════════
    fit_records = [r for r in result.records if r.action not in ("hold", "open_short")]
    if fit_records:
        fit_x = [r.idx for r in fit_records]
        fit_y = [r.fit_score for r in fit_records]
        fig.add_trace(
            go.Scatter(x=fit_x, y=fit_y, mode="lines", name="Fit Score",
                       line=dict(color="#2980b9", width=1.5),
                       fill="tozeroy", fillcolor="rgba(52,152,219,0.15)",
                       hovertemplate="idx=%{x}<br>fit=%{y:.4f}<extra></extra>"),
            row=3, col=1,
        )
    fig.add_hline(y=0.3, line_dash="dash", line_color="#e74c3c", line_width=0.8,
                   annotation_text="崩盘阈值 0.3", annotation_position="bottom left",
                   row=3, col=1)
    fig.update_yaxes(range=[-0.05, 1.1], row=3, col=1)

    # ═══════════════════════════════════════
    # Panel 4: 信号热力图
    # ═══════════════════════════════════════
    signal_map = {"open_short": 2, "hold": 0, "upsert_protection": 1, "market_close_all": -1}
    signal_colors = {2: "#8e44ad", 1: "#3498db", 0: "#ecf0f1", -1: "#e74c3c"}
    signal_labels = {2: "开仓", 1: "保护单", 0: "持有", -1: "全平"}
    signals = np.array([signal_map.get(r.action, 0) for r in result.records])
    sig_colors = [signal_colors[s] for s in signals]
    sig_labels = [signal_labels[s] for s in signals]

    fig.add_trace(
        go.Bar(x=x, y=signals, marker_color=sig_colors,
               name="Signal", showlegend=False,
               customdata=sig_labels,
               hovertemplate="idx=%{x}<br>%{customdata}<extra></extra>"),
        row=4, col=1,
    )
    fig.update_yaxes(tickvals=[-1, 0, 1, 2], ticktext=["全平", "持有", "保护单", "开仓"],
                      row=4, col=1)

    # ═══════════════════════════════════════
    # Panel 5: 权益曲线
    # ═══════════════════════════════════════
    equity = np.array(result.equity_curve)
    eq_x_pct = np.arange(len(equity)) / len(prices) * 100
    fig.add_trace(
        go.Scatter(x=eq_x_pct, y=equity, mode="lines", name="Equity",
                   line=dict(color="#2c3e50", width=1.5),
                   fill="tozeroy", fillcolor="rgba(44,62,80,0.1)",
                   hovertemplate="进度=%{x:.1f}%<br>Equity=%{y:.2f}<extra></extra>"),
        row=5, col=1,
    )
    fig.add_hline(y=result.equity_curve[0], line_dash="dot", line_color="gray",
                   line_width=0.5, row=5, col=1)

    # ═══════════════════════════════════════
    # Layout
    # ═══════════════════════════════════════
    fig.update_xaxes(title_text="K 线索引", row=4, col=1)
    fig.update_xaxes(title_text="进度 %", row=5, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Residual", row=2, col=1)
    fig.update_yaxes(title_text="Fit", row=3, col=1)
    fig.update_yaxes(title_text="Equity", row=5, col=1)

    fig.update_layout(
        height=1000,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=60, r=30, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def build_trajectory_detail_plotly(
    prices: np.ndarray,
    entry_idx: int,
) -> go.Figure | None:
    """Plotly 版单次轨迹拟合详情（拟合窗口 + 导数分析）"""

    poly = fit_cubic_trajectory(prices, entry_idx)
    if poly is None:
        return None

    v0 = poly.deriv(1)(0)
    a0 = poly.deriv(2)(0)
    entry_price = prices[entry_idx]
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        f"Entry at idx={entry_idx}, Price={entry_price:.4f}",
        f"V=f'(0)={v0:.6f} (速度/斜率) | A=f''(0)={a0:.6f} (加速度/曲率)"
    ))

    half = 15
    start = max(0, entry_idx - half)
    end = min(len(prices), entry_idx + half)
    t_range = np.arange(start, end)

    # 左侧：拟合窗口
    fig.add_trace(
        go.Scatter(x=t_range, y=prices[start:end], mode="markers+lines",
                   name="Price", marker=dict(size=4, color="#1a1a2e"),
                   line=dict(width=1, color="#1a1a2e")),
        row=1, col=1,
    )

    t_fit = np.arange(start - entry_idx, end - entry_idx, dtype=float)
    fig.add_trace(
        go.Scatter(x=t_range, y=poly(t_fit), mode="lines",
                   name="Cubic Fit", line=dict(width=2, color="#e74c3c")),
        row=1, col=1,
    )

    fig.add_vline(x=entry_idx, line_dash="dash", line_color="#8e44ad",
                   annotation_text="Entry", row=1, col=1)

    tp = find_theoretical_take_profit(poly)
    if tp is not None:
        tp_t = entry_idx + int(np.argmin(np.abs(poly(np.arange(0, 200)) - tp)))
        fig.add_trace(
            go.Scatter(x=[tp_t], y=[tp], mode="markers",
                       name=f"TP: {tp:.4f}",
                       marker=dict(symbol="star", size=16, color="#27ae60",
                                   line=dict(width=1, color="white"))),
            row=1, col=1,
        )

    # 右侧：导数
    t_ext = np.linspace(-half, half, 200)
    fig.add_trace(
        go.Scatter(x=t_ext, y=poly.deriv(1)(t_ext), mode="lines",
                   name="f'(t)", line=dict(width=1.5, color="#2980b9")),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(x=t_ext, y=poly.deriv(2)(t_ext), mode="lines",
                   name="f''(t)", line=dict(width=1.5, color="#e67e22")),
        row=1, col=2,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="black", line_width=0.5, row=1, col=2)
    fig.add_vline(x=0, line_dash="dash", line_color="#8e44ad", line_width=0.8, row=1, col=2)

    fig.update_yaxes(range=[entry_price * 0.9, entry_price * 1.1], row=1, col=1)
    fig.update_layout(height=400, template="plotly_white", hovermode="x unified")
    return fig


def build_replay_chart(prices: np.ndarray, result: BacktestResult, current_idx: int) -> go.Figure:
    """构建 tick 回放视图 — 3 面板：价格(含轨迹)、残差+贴合度、信号+持仓状态"""

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.45, 0.30, 0.25],
        subplot_titles=(
            f"tick={current_idx} | Price={prices[current_idx]:.4f}",
            "残差 & 贴合度",
            "信号状态",
        ),
    )

    # 查找当前是否有活跃持仓 → 展示 V 和 A
    active_va = None
    for trade in result.trades:
        if trade.entry_idx <= current_idx < trade.exit_idx:
            poly_va = fit_cubic_trajectory(prices, trade.entry_idx)
            if poly_va is not None:
                v_va = poly_va.deriv(1)(0)
                a_va = poly_va.deriv(2)(0)
                active_va = (v_va, a_va, trade.entry_idx)
            break

    if active_va:
        v_val, a_val, e_idx = active_va
        # 在 Panel 3 标题中附加 V/A 信息
        fig.layout.annotations[-1].update(
            text=f"信号状态 | V={v_val:.6f} (速度/斜率) A={a_val:.6f} (加速度/曲率) [Entry #{e_idx}]"
        )

    x = np.arange(min(current_idx + 1, len(prices)))

    # Panel 1: Price up to current tick
    fig.add_trace(
        go.Scatter(x=x, y=prices[:current_idx + 1], mode="lines",
                   name="Price", line=dict(color="#1a1a2e", width=0.8),
                   hovertemplate="idx=%{x}<br>Price=%{y:.2f}<extra></extra>"),
        row=1, col=1,
    )

    # 历史入场和轨迹
    entries_before = [r for r in result.records
                      if r.action == "open_short" and r.idx <= current_idx]
    exits_before = [t for t in result.trades if t.exit_idx <= current_idx]

    for rec in entries_before:

        poly = fit_cubic_trajectory(prices, rec.idx)
        if poly is not None:
            t_end = min(len(prices) - rec.idx, current_idx - rec.idx + 10, 40)
            if t_end > 0:
                t_traj = np.arange(0, t_end)
                traj_y = poly(t_traj)
                fig.add_trace(
                    go.Scatter(x=rec.idx + t_traj, y=traj_y, mode="lines",
                               name=f"Traj #{rec.idx}",
                               line=dict(dash="dash", color="#8e44ad", width=1),
                               showlegend=False),
                    row=1, col=1,
                )
                fig.add_trace(
                    go.Scatter(x=[rec.idx], y=[rec.price], mode="markers",
                               marker=dict(symbol="diamond", size=8, color="#8e44ad",
                                           line=dict(width=1, color="white")),
                               showlegend=False),
                    row=1, col=1,
                )

    # 历史出场
    for trade in exits_before:
        clr = "#27ae60" if trade.pnl > 0 else "#e74c3c"
        fig.add_trace(
            go.Scatter(x=[trade.exit_idx], y=[trade.exit_price], mode="markers",
                       marker=dict(symbol="triangle-down" if trade.pnl > 0 else "triangle-up",
                                   size=8, color=clr, line=dict(width=1, color="white")),
                       showlegend=False),
            row=1, col=1,
        )

    # 当前 tick 垂直线
    fig.add_vline(x=current_idx, line_dash="dot", line_color="#e74c3c",
                   line_width=1.5, row=1, col=1)

    # Panel 2: residuals and fit
    recs_up_to = [r for r in result.records if r.idx <= current_idx and r.action != "hold"]
    res_x = [r.idx for r in recs_up_to]
    res_y = [r.residual for r in recs_up_to]
    color_map = {"aligned": "#7f8c8d", "favorable": "#27ae60", "adverse": "#e74c3c"}
    res_colors = [color_map.get(r.direction, "#7f8c8d") for r in recs_up_to]

    fig.add_trace(
        go.Bar(x=res_x, y=res_y, marker_color=res_colors, name="Residual",
               showlegend=False,
               hovertemplate="idx=%{x}<br>%{y:.6f}<extra></extra>"),
        row=2, col=1,
    )

    fit_recs = [r for r in recs_up_to if r.action not in ("open_short",)]
    if fit_recs:
        fit_x = [r.idx for r in fit_recs]
        fit_y = [r.fit_score for r in fit_recs]
        fig.add_trace(
            go.Scatter(x=fit_x, y=fit_y, mode="lines+markers", name="Fit",
                       yaxis="y3", marker=dict(size=3, color="#2980b9"),
                       line=dict(width=1.2, color="#2980b9"),
                       hovertemplate="idx=%{x}<br>fit=%{y:.4f}<extra></extra>"),
            row=2, col=1,
        )
    fig.add_hline(y=0, line_dash="dot", line_color="black", line_width=0.5, row=2, col=1)
    fig.add_hline(y=0.3, line_dash="dash", line_color="#e74c3c", line_width=0.5, row=2, col=1)

    # Panel 3: signal state
    signal_map = {"open_short": 2, "hold": 0, "upsert_protection": 1, "market_close_all": -1}
    signal_colors = {2: "#8e44ad", 1: "#3498db", 0: "#ecf0f1", -1: "#e74c3c"}
    sigs = [signal_map.get(r.action, 0) for r in result.records if r.idx <= current_idx]
    sig_x = np.arange(len(sigs))
    sig_colors_list = [signal_colors[s] for s in sigs]

    fig.add_trace(
        go.Bar(x=sig_x, y=sigs, marker_color=sig_colors_list, name="Signal",
               showlegend=False,
               hovertemplate="idx=%{x}<extra></extra>"),
        row=3, col=1,
    )
    fig.update_yaxes(tickvals=[-1, 0, 1, 2], ticktext=["全平", "持有", "保护单", "开仓"],
                      row=3, col=1)

    # 当前 tick 高亮
    fig.add_vline(x=current_idx, line_dash="dot", line_color="#e74c3c",
                   line_width=1, row=2, col=1)
    fig.add_vline(x=current_idx, line_dash="dot", line_color="#e74c3c",
                   line_width=1, row=3, col=1)

    fig.update_yaxes(range=[prices[current_idx] * 0.95, prices[current_idx] * 1.05], row=1, col=1)

    fig.update_layout(
        height=600,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=50, r=20, t=50, b=30),
    )
    return fig
