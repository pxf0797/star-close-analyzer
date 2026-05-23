"""
曲线拟合模块 — 多模型拟合器 + 集成 + 质量评估 + 自适应。

拟合器（Phase B）：
- fit_elastic_net: Elastic Net 正则化三次多项式
- fit_mean_reversion: 指数均值回复模型
- fit_gp_matern: 高斯过程回归 (Matern 3/2)
- ensemble_fit: 多模型集成

质量评估（Phase C）：
- evaluate_fit_quality: 完整 Q-Score 评估
- compute_q_score: 复合质量评分

自适应（Phase C）：
- calibrate_thresholds: Walk-forward 参数重校准
- walk_forward_validate: 分折验证
"""

from star_analyzer.fitting.elastic_net import fit_elastic_net
from star_analyzer.fitting.mean_reversion import fit_mean_reversion
from star_analyzer.fitting.gp_regression import fit_gp_matern
from star_analyzer.fitting.ensemble import ensemble_fit, TrajectoryFit
from star_analyzer.fitting.quality import evaluate_fit_quality, compute_q_score
from star_analyzer.fitting.adaptive import calibrate_thresholds, walk_forward_validate, CalibrationResult

__all__ = [
    "fit_elastic_net",
    "fit_mean_reversion",
    "fit_gp_matern",
    "ensemble_fit",
    "TrajectoryFit",
    "evaluate_fit_quality",
    "compute_q_score",
    "calibrate_thresholds",
    "walk_forward_validate",
    "CalibrationResult",
]
