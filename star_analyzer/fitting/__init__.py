"""
曲线拟合模块 — 多模型拟合器 + 集成。

Phase B 提供四种拟合器：
- fit_cubic_wls: WLS 三次多项式（Phase A 改进版）
- fit_elastic_net: Elastic Net 正则化三次多项式
- fit_mean_reversion: 指数均值回复模型
- fit_gp_matern: 高斯过程回归 (Matern 3/2)

集成入口：
- ensemble_fit: 并行运行所有拟合器，取加权中位数预测
"""

from star_analyzer.fitting.elastic_net import fit_elastic_net
from star_analyzer.fitting.mean_reversion import fit_mean_reversion
from star_analyzer.fitting.gp_regression import fit_gp_matern
from star_analyzer.fitting.ensemble import ensemble_fit, TrajectoryFit

__all__ = [
    "fit_elastic_net",
    "fit_mean_reversion",
    "fit_gp_matern",
    "ensemble_fit",
    "TrajectoryFit",
]
