"""
星空策略 · 轨迹冻结平仓分析工具
==================================
基于泰勒展开式与三次多项式轨迹拟合的量化做空策略平仓分析系统。

用法:
    python main.py                          # 使用合成数据演示
    python main.py --data btc_hourly.csv    # 使用 CSV 数据
    python main.py --data btc_hourly.csv --entry-idx 150  # 指定入场点单次分析
"""

import argparse
import sys
import time
import numpy as np
import pandas as pd

from star_analyzer.trajectory import fit_cubic_trajectory, find_theoretical_take_profit
from star_analyzer.closing import (
    Action, Direction, PositionState, evaluate_close,
    check_profit_threshold, compute_residual, compute_fit_score,
    half_life_decay,
)
from star_analyzer.backtest import BacktestEngine, BacktestResult
from star_analyzer.visualize import plot_analysis, plot_trajectory_detail


def generate_demo_data(n: int = 500) -> np.ndarray:
    """生成具有涨跌周期的合成价格数据用于演示"""
    np.random.seed(42)
    t = np.arange(n, dtype=float)
    # 多周期叠加
    trend1 = 100 + 10 * np.sin(2 * np.pi * t / 200)
    trend2 = 5 * np.sin(2 * np.pi * t / 60)
    trend3 = 2 * np.sin(2 * np.pi * t / 25)
    noise = np.random.randn(n) * 1.5
    prices = trend1 + trend2 + trend3 + noise
    return np.maximum(prices, 1.0)


def load_csv(filepath: str, col: str = "close") -> np.ndarray:
    """从 CSV 加载价格数据"""
    df = pd.read_csv(filepath)
    if col not in df.columns:
        cols = [c for c in df.columns if c.lower() in ("close", "price", "收盘", "c")]
        if cols:
            col = cols[0]
        else:
            col = df.columns[-1]
    return df[col].values.astype(float)


def run_single_analysis(prices: np.ndarray, entry_idx: int):
    """单次入场分析"""
    print(f"\n{'='*60}")
    print(f"  单次轨迹分析 — Entry Index: {entry_idx}, Price: {prices[entry_idx]:.4f}")
    print(f"{'='*60}")

    poly = fit_cubic_trajectory(prices, entry_idx)
    if poly is None:
        print("❌ 轨迹拟合失败（不满足一阶导≈0 且 二阶导<0）")
        return

    tp = find_theoretical_take_profit(poly)
    print(f"\n📐 三次多项式系数: {poly}")
    print(f"  f'(0) = {poly.deriv(1)(0):.8f} (≈0 ✅)")
    print(f"  f''(0) = {poly.deriv(2)(0):.8f} (<0 ✅)")

    if tp:
        profit_pct = (prices[entry_idx] - tp) / prices[entry_idx] * 100
        print(f"\n🎯 理论止盈价: {tp:.6f} (盈利 {profit_pct:.2f}%)")
        fee_check = check_profit_threshold(prices[entry_idx], tp, fee_rate=0.0008)
        print(f"  覆盖手续费: {'✅ 是' if fee_check else '❌ 否'}")
    else:
        print("\n⚠️  未找到 t>0 的局部极小值")

    feats = {
        "pos": ("favorable", "📉 下跌超预期 → 动能强 → 移动止损吃满趋势"),
        "neg": ("adverse", "📈 反弹回撤 → 做空不利 → 考虑止损"),
        "near": ("aligned", "➖ 贴合预测 → 持仓等待"),
    }

    print(f"\n{'─'*60}")
    print(f"{'t':>4} {'Price':>10} {'Pred':>10} {'Residual':>10} {'Fit':>7} {'Dir':>10} {'Action':>20}")
    print(f"{'─'*60}")

    vol = 0.002
    fee_rate = 0.0008
    half_life = 15

    pos = PositionState(
        entry_price=float(prices[entry_idx]),
        entry_idx=entry_idx,
        quantity=1.0,
        anchor=poly,
        theoretical_tp=tp,
        fee_rate=fee_rate,
        half_life=half_life,
    )

    for i in range(entry_idx, min(entry_idx + 40, len(prices))):
        t = i - entry_idx
        decision = evaluate_close(pos, i, float(prices[i]), vol)

        decay = half_life_decay(t, half_life)
        extra = ""
        if decision.action == Action.UPSERT_PROTECTION:
            extra = f" trigger={decision.trigger_price:.4f}"
        elif decision.action == Action.MARKET_CLOSE_ALL:
            extra = f" ⚡{decision.reason_code}"

        print(f"{t:>4} {prices[i]:>10.4f} {decision.pred_price:>10.4f} "
              f"{decision.residual:>10.6f} {decision.fit_score:>7.4f} "
              f"{decision.direction.value:>10} {decision.action.value:>20}{extra}")

        if decision.action == Action.MARKET_CLOSE_ALL:
            break

    print(f"{'─'*60}")

    plot_trajectory_detail(prices, entry_idx)
    print(f"\n📊 轨迹拟合细节图已保存: trajectory_detail.png")


def run_backtest(prices: np.ndarray, args):
    """运行完整回测"""
    print(f"\n🏃 运行回测 — {len(prices)} 根 K 线")

    engine = BacktestEngine(
        initial_capital=args.capital,
        fee_rate=args.fee_rate,
        half_life=args.half_life,
        confidence_threshold=args.confidence,
        max_relative_distance=args.max_distance,
        hard_stop_multiplier=args.hard_stop_mul,
    )

    start = time.time()
    result = engine.run(prices)
    elapsed = time.time() - start

    print(f"\n{'='*60}")
    print(f"  回测完成 ({elapsed:.2f}s)")
    print(f"{'='*60}")
    print(f"  总交易: {result.stats['total_trades']}")
    print(f"  胜率:   {result.stats['win_rate']}%")
    print(f"  总盈亏: {result.stats['total_pnl_pct']:.2f}%")
    print(f"  盈亏比: {result.stats['profit_factor']}")
    print(f"  最大回撤: {result.stats['max_drawdown']:.2f}%")
    print(f"  夏普比: {result.stats['sharpe']}")
    print(f"  终值:   {result.stats['final_equity']:.2f}")

    print(f"\n  交易详情:")
    for i, trade in enumerate(result.trades):
        tag = "✅" if trade.pnl > 0 else "❌"
        print(f"  {i+1}. [{trade.entry_idx}→{trade.exit_idx}] "
              f"{trade.pnl_pct:+.2f}% | {trade.reason} {tag}")

    output = args.output or "star_analysis.png"
    plot_analysis(prices, result, save_path=output)
    print(f"\n📊 分析图表已保存: {output}")


def main():
    parser = argparse.ArgumentParser(
        description="星空策略 · 轨迹冻结平仓分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                              # 合成数据演示
  python main.py --data btc_hourly.csv        # CSV 回测
  python main.py --data btc_hourly.csv --entry-idx 150  # 单次分析
  python main.py --half-life 20 --confidence 0.25        # 调参
        """,
    )
    parser.add_argument("--data", help="CSV 价格数据文件路径")
    parser.add_argument("--col", default="close", help="价格列名 (默认: close)")
    parser.add_argument("--entry-idx", type=int, help="指定入场 K 线索引 (单次分析模式)")
    parser.add_argument("--output", help="图表输出路径")
    parser.add_argument("--capital", type=float, default=1000.0, help="初始资金 (默认: 1000)")
    parser.add_argument("--fee-rate", type=float, default=0.0008, help="双边手续费率 (默认: 0.08%%)")
    parser.add_argument("--half-life", type=int, default=15, help="半衰期 bar 数 (默认: 15)")
    parser.add_argument("--confidence", type=float, default=0.3, help="自信度崩盘阈值 (默认: 0.3)")
    parser.add_argument("--max-distance", type=float, default=0.05, help="保护单最大相对距离 (默认: 0.05)")
    parser.add_argument("--hard-stop-mul", type=float, default=2.0, help="硬止损倍数 (默认: 2.0)")

    args = parser.parse_args()

    # 加载数据
    if args.data:
        print(f"📂 加载数据: {args.data}")
        prices = load_csv(args.data, args.col)
    else:
        print("🔧 使用合成演示数据")
        prices = generate_demo_data(500)

    print(f"  价格序列长度: {len(prices)}")
    print(f"  范围: [{prices.min():.4f}, {prices.max():.4f}]")

    if args.entry_idx is not None:
        run_single_analysis(prices, args.entry_idx)
    else:
        run_backtest(prices, args)


if __name__ == "__main__":
    main()
