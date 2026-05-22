"""
星空策略 · 动态回测面板 (Phase 0 — Streamlit)
"""

import sys
import time
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="星空策略 · 回测面板",
    page_icon="🌌",
    layout="wide",
)

from datasource import (
    create_source, SinSource, PolySource, GbmSource,
    KrakenSource, CoinGeckoSource, OkxSource, CsvSource,
)
from backtest import BacktestEngine
from visualize import plot_analysis, plot_trajectory_detail
from trajectory import fit_cubic_trajectory, find_theoretical_take_profit
from closing import check_profit_threshold


# ═══════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════

st.sidebar.title("🌌 星空策略")
st.sidebar.caption("轨迹冻结平仓分析 · 回测面板")

# 数据源
st.sidebar.subheader("数据源")
source_name = st.sidebar.selectbox(
    "选择数据源",
    ["sin", "poly", "gbm", "kraken", "coingecko", "okx"],
    format_func=lambda x: {
        "sin": "正弦叠加 (演示)",
        "poly": "分段样条+噪声 (轨迹测试)",
        "gbm": "几何布朗运动",
        "kraken": "Kraken BTC/USD",
        "coingecko": "CoinGecko BTC/USD",
        "okx": "OKX BTC/USDT",
    }[x],
)

source_kwargs = {}
if source_name in ("sin", "poly", "gbm"):
    n_bars = st.sidebar.slider("K线数量", 100, 2000, 500, 50)
    seed = st.sidebar.number_input("随机种子", 0, 999, 42)
    source_kwargs = {"n": n_bars, "seed": seed}
elif source_name in ("kraken", "okx"):
    source_kwargs = {}
elif source_name == "coingecko":
    days = st.sidebar.slider("天数", 1, 90, 30)
    source_kwargs = {"days": days}

# 策略参数
st.sidebar.subheader("策略参数")
half_life = st.sidebar.slider("半衰期 (bar)", 5, 60, 15, 1,
    help="衰减因子半衰期。越小越敏感，越大越宽容")
confidence = st.sidebar.slider("自信度阈值", 0.10, 0.80, 0.30, 0.05,
    help="贴合度跌破此值触发自信度崩盘全平")
fee_rate = st.sidebar.number_input("双边手续费率", 0.0001, 0.01, 0.0008, format="%.4f")
max_distance = st.sidebar.slider("保护单最大距离", 0.01, 0.20, 0.05, 0.01,
    help="保护单触发价相对现价的最大偏移比例")
hard_stop_mul = st.sidebar.slider("硬止损倍数", 1.0, 5.0, 2.0, 0.5,
    help="硬止损 = max(N×tol, 最大距离)")
capital = st.sidebar.number_input("初始资金", 100, 100000, 1000, 100)

# 单次轨迹分析
st.sidebar.subheader("单次轨迹分析")
entry_idx = st.sidebar.number_input("入场 K 线索引", 0, 2000, 0,
    help="设为 0 则运行完整回测；设为具体索引则对该点做轨迹拟合分析")

run_btn = st.sidebar.button("▶ 运行分析", type="primary", use_container_width=True)


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

st.title("🌌 星空策略 · 轨迹冻结平仓分析")
st.caption("基于泰勒展开式与三次多项式轨迹拟合的量化做空策略回测系统")

if not run_btn:
    st.info("👈 在侧栏选择数据源和参数后，点击「运行分析」")
    st.markdown("""
    ### 快速开始
    - **数据源**: `正弦叠加` 最快启动，`Kraken` 拉取真实 BTC 行情
    - **半衰期**: 控制持仓期间对残差的容忍度衰减速度
    - **自信度阈值**: 贴合度跌破此值 → 市价全平
    - **入场索引**: 设为 0 运行完整回测；设为具体值做单次轨迹分析
    """)
    st.stop()


# 加载数据
with st.spinner("加载数据…"):
    try:
        source = create_source(source_name, **source_kwargs)
        prices = source.fetch()
        meta = source.meta
    except Exception as e:
        st.error(f"数据加载失败: {e}")
        st.stop()

st.success(f"已加载 {meta.length} 根 K 线 — {meta.name}")

# 运行分析
if entry_idx > 0:
    # === 单次轨迹分析 ===
    st.subheader(f"单次轨迹分析 — Entry #{entry_idx}")

    if entry_idx >= len(prices):
        st.error(f"入场索引 {entry_idx} 超出数据范围 (0-{len(prices)-1})")
        st.stop()

    col1, col2 = st.columns([3, 2])

    with col1:
        fig_detail = plot_trajectory_detail(prices, entry_idx, save_path="")
        st.pyplot(fig_detail)
        import matplotlib.pyplot as plt
        plt.close(fig_detail)

    with col2:
        poly = fit_cubic_trajectory(prices, entry_idx)
        if poly is None:
            st.error("轨迹拟合失败 — 不满足 f'(0)≈0 且 f''(0)<0")
        else:
            tp = find_theoretical_take_profit(poly)
            entry_price = prices[entry_idx]

            st.markdown("**三次多项式系数**")
            st.code(str(poly), language=None)
            st.metric("f'(0)", f"{poly.deriv(1)(0):.6f}", delta="≈0" if abs(poly.deriv(1)(0)) < 1 else "⚠️ 偏离")
            st.metric("f''(0)", f"{poly.deriv(2)(0):.6f}", delta="<0 ✅" if poly.deriv(2)(0) < 0 else "⚠️ ≥0")

            if tp:
                profit_pct = (entry_price - tp) / entry_price * 100
                fee_ok = check_profit_threshold(entry_price, tp, fee_rate)
                st.metric("理论止盈价", f"${tp:.2f}", delta=f"{profit_pct:+.2f}%")
                st.metric("覆盖手续费", "✅ 是" if fee_ok else "❌ 否")

else:
    # === 完整回测 ===
    with st.spinner("回测计算中…"):
        t0 = time.time()
        engine = BacktestEngine(
            initial_capital=capital,
            fee_rate=fee_rate,
            half_life=half_life,
            confidence_threshold=confidence,
            max_relative_distance=max_distance,
            hard_stop_multiplier=hard_stop_mul,
        )
        result = engine.run(prices)
        elapsed = time.time() - t0

    # 统计卡片
    stats = result.stats
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("总交易", stats["total_trades"])
    c2.metric("胜率", f"{stats['win_rate']}%")
    c3.metric("总盈亏", f"{stats['total_pnl_pct']:.2f}%")
    c4.metric("盈亏比", stats["profit_factor"])
    c5.metric("最大回撤", f"{stats['max_drawdown']:.2f}%")
    c6.metric("夏普比", stats["sharpe"])
    c7.metric("耗时", f"{elapsed:.2f}s")

    # 主图表
    st.subheader("回测分析图")
    fig_main = plot_analysis(prices, result, save_path="", title="星空策略 · 轨迹冻结平仓分析")
    st.pyplot(fig_main)
    import matplotlib.pyplot as plt
    plt.close(fig_main)

    # 交易详情
    if result.trades:
        st.subheader("逐笔交易详情")
        trade_data = []
        for i, t in enumerate(result.trades):
            tag = "✅" if t.pnl > 0 else "❌"
            trade_data.append({
                "#": i + 1,
                "入场": t.entry_idx,
                "出场": t.exit_idx,
                "持仓 bar": t.exit_idx - t.entry_idx,
                "入场价": f"${t.entry_price:.2f}",
                "出场价": f"${t.exit_price:.2f}",
                "盈亏": f"{t.pnl_pct:+.2f}%",
                "原因": t.reason,
                "结果": tag,
            })

        import pandas as pd
        df_trades = pd.DataFrame(trade_data)
        st.dataframe(df_trades, use_container_width=True, hide_index=True)

        # 各平仓原因统计
        reasons = {}
        for t in result.trades:
            reasons[t.reason] = reasons.get(t.reason, 0) + 1
        st.caption("平仓原因分布: " + " | ".join(f"{k}: {v}" for k, v in reasons.items()))
