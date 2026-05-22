"""
星空策略 · 轨迹冻结平仓分析系统
==================================
基于泰勒展开式与三次多项式轨迹拟合的量化做空策略平仓分析。
"""

from star_analyzer.trajectory import fit_cubic_trajectory, find_theoretical_take_profit
from star_analyzer.closing import (
    Action, Direction, PositionState, CloseDecision,
    evaluate_close, check_profit_threshold, compute_residual,
    compute_fit_score, half_life_decay,
)
from star_analyzer.backtest import BacktestEngine, BacktestResult, Trade, TickRecord
from star_analyzer.datasource import (
    DataSource, DataMeta, create_source,
    SinSource, PolySource, GbmSource,
    KrakenSource, CoinGeckoSource, OkxSource, CsvSource,
)
from star_analyzer.visualize import plot_analysis, plot_trajectory_detail
from star_analyzer.visualize_plotly import (
    build_interactive_chart, build_trajectory_detail_plotly, build_replay_chart,
)
