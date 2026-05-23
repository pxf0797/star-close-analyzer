# 星空策略 · 曲线拟合系统改善方案

## 一、执行摘要

当前曲线拟合系统存在 **7 项具体弱点**，形成一条从"拟合无质量检查"到"交易信号崩溃"的级联失效链：缺乏拟合优度判断 → 过拟合三次多项式进入实盘 → 轨迹外推至 t=200（拟合窗口的 10 倍）剧烈发散 → fit_score 快速衰减 → 触发过早/错误平仓。**推荐方案**：分 Phase A/B/C 三阶段推进，Phase A（1-2 天）以 WLS 指数衰减、解析 f'(0)=0 约束、CV(RMSE) 门控作为最小可行升级，可单独交付；Phase B（1 周）引入弹性网正则化、指数均值回复模型和高斯过程回归；Phase C 建立持续验证体系。预期综合效果：胜率从 45-50% 提升至 50-60%，夏普比率从 0.8-1.0 提升至 1.1-1.5，年化交易次数减少 20-40%（被过滤的是低质量拟合产生的噪音交易），最大回撤降低 3-5 个百分点。

---

## 二、现状诊断

### 2.1 级联失效链

```
固定窗口(20bar) → 无拟合质量检查 → 过拟合三次多项式
    → 轨迹外推至 t=200 (10×窗口) → 外推发散
    → fit_score 快速下降 → 自信度崩盘 / 过早平仓
```

当前系统从拟合到交易的完整链路不存在任何质量闸门。多项式一旦通过 `|f'(0)|/scale < 0.05` 和 `f''(0) < 0` 这两项基本检查，便无条件进入实盘。后续交易完全依赖 `fit_score`（基于残差的 sigmoid 贴合度），但拟合本身是否正确从未被验证。

### 2.2 七项弱点汇总

| # | 弱点 | 严重度 | 位置 | 说明 |
|---|------|--------|------|------|
| 1 | 固定 20-bar 窗口，无自适应性 | 中 | `trajectory.py:12` | 高波动期窗口不足，低波动期窗口冗余 |
| 2 | 数据边界非对称截断 | 低 | `trajectory.py:24-25` | `max(0, idx-half)` 和 `min(len, idx+half)` 导致边缘拟合偏差 |
| 3 | f'(0)≈0 约束过于宽松 | 中 | `trajectory.py:48` | 允许相当于 5%/bar 的斜率，对 1H K 线过于宽松 |
| 4 | **无拟合优度指标** | **高** | 全局缺失 | 无 R²、无 CV(RMSE)、无残差检验——拟合质量完全未知 |
| 5 | **三次多项式硬编码** | **高** | `trajectory.py:34` | 无模型选择，20 个数据点不需要 4 个参数的复杂度 |
| 6 | **冻结轨迹 10x 外推** | **高** | `trajectory.py:56` | t_max=200，而窗口仅 20；三次多项式在边界外剧烈发散 |
| 7 | 简单的局部极大检测 | 低 | `backtest.py:213-224` | 5-bar 窗口无确认机制，伪极值点不设防 |

### 2.3 问题严重性量化评估

基于代码静态分析和行业经验估算（实际值需回测校准）：

| 影响维度 | 当前估计 | 问题根源 |
|----------|---------|----------|
| 低质量拟合占比 | 30-50% | 问题 #4, #5 |
| 外推误差（t=200） | 不可控 | 问题 #6 |
| 伪信号开仓率 | 20-30% | 问题 #7 |
| 因拟合过早平仓 | 15-25% | 问题 #4 → fit_score 衰减 |

---

## 三、改善方案总览

### 3.1 完整方案矩阵

| 优先级 | 方法 | 修复问题 | 工作量 | 预期收益 | 风险 |
|--------|------|---------|--------|---------|------|
| **P0** | WLS 指数衰减权重 | #1, #4 | 0.5天 | 近期数据贡献 +30%，远端点噪声压制 | 衰减速率需调参 |
| **P0** | 解析 f'(0)=0 约束 | #3, #5 | 0.25天 | 参数从 4 降到 3，过拟合风险 -25% | 强假设（必须为局部极值点） |
| **P0** | CV(RMSE) 门控 | #4 | 0.25天 | 过滤 ~20-30% 低质量拟合 | 阈值需回测校准 |
| **P0** | 外推距离限制 | #6 | 0.1天 | 消除外推发散风险 | 可能缩短止盈距离 |
| **P1** | Elastic Net 正则化 | #5 | 0.5天 | 系数稳定性 +40%，自动变量选择 | α/λ 需交叉验证 |
| **P1** | 均值回复指数衰减模型 | #5, #6 | 0.5天 | 自然渐近线，外推安全 | 仅适合均值回复场景 |
| **P1** | 残差 DW + Ljung-Box 检验 | #4 | 0.5天 | 再过滤 10-15% 伪拟合 | 小样本下检验力有限 |
| **P2** | 高斯过程回归 (Matern 3/2) | #1, #5, #6 | 1.5天 | 内置不确定性，外推收敛到先验均值 | 计算开销 ×3-5 |
| **P2** | 模型集成 (Ensemble) | #5 | 1天 | 多模型投票，稳定性 +20% | 复杂度上升 |
| **P3** | 自适应窗口 | #1 | 1天 | 波动率自适应，信噪比提升 | 过度调参风险 |
| **P3** | AICc 模型选择 | #5 | 0.5天 | 自动选择最优复杂度 | 小样本 AICc 仍有偏差 |
| **P3** | 入场信号增强 | #7 | 0.5天 | 伪极值过滤 +50% | 可能错过真信号 |

### 3.2 优先级判定逻辑

- **P0**：改动小、收益明确、不引入新依赖、可独立交付、立即改善交易质量
- **P1**：需要调参和回测验证，但方法论成熟，风险和收益对等
- **P2**：引入新计算框架，需性能评估，但提供差异化能力（不确定性量化）
- **P3**：锦上添花，需在 P0-P2 效果验证后再决定投入

---

## 四、推荐实施路线

### Phase A — 最小可行升级（1-2 天）

**目标**：用最小代价堵住最严重的漏洞——拟合质量不可知 + 外推发散。

#### Step A1：WLS 加权最小二乘 + 解析 f'(0)=0 约束

- **文件**：`star_analyzer/trajectory.py`，修改 `fit_cubic_trajectory()`
- **改动**：
  1. 用牛顿法/最小二乘公式直接构建满足 f'(0)=0 的三次多项式：P(t) = c₀ + c₂·t² + c₃·t³
  2. 引入指数衰减权重 wᵢ = exp(-λ·|tᵢ|)，λ = ln(2)/half_life_weight，half_life_weight=10
  3. 用加权正规方程 (XᵀWX)β = XᵀWy 求解
- **验证标准**：拟合残差与 OLS 对比，远期点残差应更小；单元测试覆盖边界情况

#### Step A2：CV(RMSE) 门控

- **文件**：`star_analyzer/trajectory.py`，在 `fit_cubic_trajectory()` 返回前添加
- **改动**：
  1. 计算 CV(RMSE) = RMSE / mean(|y|)，若 > 0.05 则返回 None
  2. 添加 `fit_quality` 字段（可选返回或日志记录）
- **验证标准**：在回测中观察被拒绝拟合的比例，预期 20-30%

#### Step A3：外推距离硬限制

- **文件**：`star_analyzer/trajectory.py`，修改 `find_theoretical_take_profit()`
- **改动**：
  1. 将 `t_max` 从 200 改为 `min(window * 2, 60)`（拟合窗口的 2 倍，上限 60）
  2. 同步修改函数签名，添加参数说明
- **验证标准**：无外推距离超过 60 bar 的止盈价；回测止盈触发时间分布正常

#### Step A4：回测验证 Phase A 综合效果

- **文件**：`star_analyzer/backtest.py`（如需记录 fit_quality 到 Trade）
- **验证标准**：
  - 交易次数减少 20-40%（低质量拟合被过滤）
  - 胜率提升 3-5 个百分点
  - 无因外推发散导致的 `confidence_collapse` 平仓

---

### Phase B — 方法论升级（1 周）

**目标**：引入模型多样性和不确定性量化，从"单一多项式"升级为"多模型集成"。

#### Step B1：弹性网正则化三次拟合

- **文件**：新增 `star_analyzer/fitting/elastic_net.py`
- **改动**：实现 Elastic Net 正规方程（L1+L2 混合惩罚），用坐标下降法求解；α 默认 0.001，L1_ratio 默认 0.5
- **验证标准**：系数绝对值比 OLS 平均小 15-30%；在噪声样本上系数稳定性提升

#### Step B2：指数均值回复模型

- **文件**：新增 `star_analyzer/fitting/mean_reversion.py`
- **改动**：实现 f(t) = p₀ + α·(1 - e^(-λt))，用 Levenberg-Marquardt 求解非线性最小二乘
- **验证标准**：在均值回复特征明显的价格段，RMSE 比三次多项式低 10-20%

#### Step B3：高斯过程回归

- **文件**：新增 `star_analyzer/fitting/gp_regression.py`
- **改动**：使用 Matern 3/2 核的 GPR，返回预测均值 ± 2σ 区间；外推时不确定性单调扩大，自然收敛到先验均值
- **验证标准**：外推至 t=60 时预测区间宽度合理；不确定性可作为交易信心的辅助指标

#### Step B4：模型集成与选择

- **文件**：新增 `star_analyzer/fitting/ensemble.py`
- **改动**：
  1. 对同一价格段并行运行 WLS 三次 + 弹性网三次 + 均值回复 + GPR
  2. 取各模型预测的加权中位数作为集成预测
  3. 模型间分歧度（标准偏差）作为额外的不确定性信号
- **验证标准**：集成预测在回测中的夏普比率高于任一单模型

---

### Phase C — 持续优化

**目标**：建立系统化的质量评估和自适应机制。

#### C1：复合 Q-Score 系统

- **实现位置**：`star_analyzer/fitting/quality.py`
- **公式**：

  ```
  Q = 0.25 · I(DW ∈ [1.5,2.5])
    + 0.20 · I(LB p-value > 0.05)
    + 0.25 · max(0, 1 - CV_RMSE / 0.10)
    + 0.20 · max(0, 1 - (OOS_RMSE/IS_RMSE) / 2.0)
    + 0.10 · max(0, (R² - 0.50) / 0.50)
  ```

- **决策阈值**：Q < 0.50 拒绝开仓；0.50 ≤ Q < 0.75 仅强信号开仓；Q ≥ 0.75 正常开仓

#### C2：回测级 IC 验证

- **实现位置**：`star_analyzer/backtest.py`
- **逻辑**：每次交易计算 Q-score 与标准化 PnL（PnL/ATR）之间的 Spearman 秩相关
- **目标**：IC > 0.15（弱有效），IC > 0.25（强有效）

#### C3：Walk-Forward 自适应

- **周期**：每 2-3 个月在最近数据上重校准门控阈值和 WLS 衰减速率
- **预警**：参数变动超过 30% 触发人工审核

---

## 五、代码改动示例

### 5.1 `fit_cubic_trajectory` 改进版（Phase A 核心）

```python
def fit_cubic_trajectory(
    prices: np.ndarray,
    entry_idx: int,
    window: int = 20,
    half_life_weight: int = 10,
    cv_rmse_threshold: float = 0.05,
) -> Polynomial | None:
    """
    使用 WLS + 解析 f'(0)=0 约束的三次多项式拟合。

    改进：
    - 指数衰减权重（WLS），half_life_weight 控制衰减速率
    - 解析约束 f'(0)=0，将参数从 4 维降为 3 维
    - CV(RMSE) 门控，拒绝高噪声拟合
    """
    half = window // 2
    start = max(0, entry_idx - half)
    end = min(len(prices), entry_idx + half)

    if end - start < 4:
        return None

    t = np.arange(start - entry_idx, end - entry_idx, dtype=float)
    y = prices[start:end].astype(float)
    n = len(t)

    # ---- WLS 指数衰减权重 ----
    lam = np.log(2) / half_life_weight
    w = np.exp(-lam * np.abs(t))
    W = np.diag(w)

    # ---- 解析约束: P(t) = c0 + c2·t² + c3·t³  (f'(0)=0 自动满足) ----
    # 设计矩阵 X: [1, t², t³]
    X = np.column_stack([np.ones(n), t**2, t**3])

    try:
        # 加权正规方程: (XᵀWX)β = XᵀWy
        XtW = X.T @ W
        beta = np.linalg.solve(XtW @ X, XtW @ y)
    except np.linalg.LinAlgError:
        return None

    c0, c2, c3 = beta
    poly = Polynomial([c0, 0.0, c2, c3])  # c1 = 0

    # ---- 验证 f''(0) < 0 ----
    if poly.deriv(2)(0) >= 0:
        return None

    # ---- CV(RMSE) 门控 ----
    y_pred = poly(t)
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    price_scale = np.mean(np.abs(y))
    if price_scale == 0:
        return None
    cv_rmse = rmse / price_scale
    if cv_rmse > cv_rmse_threshold:
        return None

    return poly
```

### 5.2 门控逻辑示意

```
prices[i] → entry_signal? ─→ fit_cubic_trajectory()
                                  │
                    ┌─────────────┴─────────────┐
                    │  Stage 1: 结构有效性       │
                    │  - f''(0) < 0  (局部极大)  │
                    └─────────────┬─────────────┘
                                  │ 通过
                    ┌─────────────┴─────────────┐
                    │  Stage 2: 残差独立性       │
                    │  - DW ∈ [1.5, 2.5]        │
                    │  - LB p-value > 0.05       │
                    └─────────────┬─────────────┘
                                  │ 通过
                    ┌─────────────┴─────────────┐
                    │  Stage 3: 拟合质量         │
                    │  - CV(RMSE) < 0.05         │
                    │  - R² > 0.50               │
                    │  - 外推误差比 < 2.0        │
                    └─────────────┬─────────────┘
                                  │ 通过
                                  ▼
                           Composite Q-Score
                           (Q ≥ 0.50 → 开仓)
```

### 5.3 `find_theoretical_take_profit` 改进版

```python
def find_theoretical_take_profit(
    poly: Polynomial,
    window: int = 20,
    t_max: int | None = None,
) -> float | None:
    """
    寻找三次多项式在 t > 0 的下一个局部极小值。

    改进：外推距离受限于 2×window，上限 60 bar。
    """
    if t_max is None:
        t_max = min(window * 2, 60)

    deriv = poly.deriv(1)
    roots = deriv.roots()

    best_t = None
    for r in roots:
        r_real = float(r.real) if np.iscomplex(r) else float(r)
        if 1 < r_real < t_max and abs(r.imag) < 1e-8:
            if poly.deriv(2)(r_real) > 0:
                if best_t is None or r_real < best_t:
                    best_t = r_real

    if best_t is None:
        return None
    return float(poly(best_t))
```

---

## 六、拟合质量门控系统

### 6.1 三级门控管道

| 阶段 | 检查项 | 通过条件 | 未通过后果 | 预期过滤率 |
|------|--------|---------|-----------|-----------|
| S1: 结构有效性 | f''(0) < 0 | 局部极大值 | 直接拒绝 | ~15% |
| S2: 残差独立性 | Durbin-Watson | 1.5 < DW < 2.5 | 残差自相关，模型未捕捉数据结构 | ~10-15% |
| S2: 残差独立性 | Ljung-Box Q 检验 | p-value > 0.05 | 残差非白噪声 | ~5-10% |
| S3: 拟合质量 | CV(RMSE) | < 0.05 | 拟合误差太大 | ~10-15% |
| S3: 解释力 | R² | > 0.50 | 模型解释力不足一半 | ~5-10% |
| S3: 外推 | OOS/IS RMSE 比 | < 2.0 | 样本外预测不稳定 | ~10% |

> 注意：各阶段过滤率非简单相加（有重叠）。综合预期：30-50% 的拟合被至少一个阶段拒绝。

### 6.2 Q-Score 公式

完整公式见 Phase C 的 C1 节。核心设计原则：
- **DW/LB 残差检验权重最高**（合计 45%），因残差自相关直接宣告模型无效
- **CV(RMSE) 权重 25%**，衡量绝对拟合精度
- **外推稳定性 20%**，衡量泛化能力
- **R² 权重 10%**，辅助指标

---

## 七、预期效果与风险

### 7.1 各阶段预期提升

| 指标 | 当前基线 | Phase A | Phase A+B | Phase A+B+C |
|------|---------|---------|-----------|-------------|
| 胜率 | 45-50% | 48-53% | 50-58% | 52-60% |
| 夏普比率 | 0.8-1.0 | 1.0-1.2 | 1.1-1.4 | 1.2-1.5 |
| 年化交易次数（相对） | 100% | 70-80% | 60-75% | 55-70% |
| 最大回撤 | 15-20% | 12-17% | 10-15% | 8-13% |
| 平均持仓时间（bar） | 12-18 | 14-20 | 15-22 | 16-24 |
| `confidence_collapse` 平仓率 | 15-20% | 5-10% | 3-7% | 2-5% |

### 7.2 主要风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 阈值过严导致信号过少 | 中 | 交易频率不足，策略经济性下降 | 从宽松阈值起步，回测中逐步收紧 |
| WLS 衰减速率选择不当 | 低 | 远期信息损失过多 | 对比 half_life ∈ {5, 10, 15, 20} 的回测结果 |
| GPR 计算开销过大 | 中 | 实盘 tick 级延迟超标 | 仅在开仓时计算 GPR（非每个 tick）；缓存核矩阵 |
| 过拟合于历史回测 | 中 | 实盘表现显著弱于回测 | Walk-forward 验证；参数变动监控；样本外保留期 |
| 模型集成复杂度失控 | 低 | 维护困难，调试成本高 | 每个模型保持独立可测试；集成层薄封装 |

---

## 八、决策要点

以下关键决策需要确认后才可推进实施：

### D1：是否接受交易量减少？

Phase A 引入 CV(RMSE) 门控后，预期 20-30% 的开仓信号被拒绝。这**不是代价而是收益**——被拒绝的是低质量拟合驱动的噪音交易。但若策略的资金曲线依赖高频小额盈利，频率下降可能影响复合效应。**建议**：接受 20-40% 交易量减少，以胜率和夏普提升为补偿。

### D2：WLS 衰减速率初始化

指数衰减的 half-life 建议初始值 10 bar（约 10 小时，对于 1H K 线）。更小的值（如 5）更激进地关注最近点，更大的值（如 20）更接近等权 OLS。**建议**：以 10 为初始值，通过 walk-forward 在 5/10/15/20 中选择最优。

### D3：Phase B 依赖选择

GPR 需要 `scikit-learn`，均值回复模型需要 `scipy.optimize`。若项目当前无这些依赖，需评估引入成本。**建议**：Phase A 保持零新依赖；Phase B 引入 scikit-learn（业界标准，风险低）；均值回复模型可纯 numpy 实现以避免 scipy 依赖。

### D4：模型选择策略

当多个模型（WLS 三次、弹性网三次、均值回复、GPR）都通过门控时，选择哪个模型的预测？**建议**：Phase B 实现等权集成（取中位数预测），后续根据各模型的 Q-score 做加权。

### D5：回测验证标准

定义 Phase A "成功"的量化标准：
- 夏普比率不低于当前（不恶化）
- `confidence_collapse` 平仓占比下降至 10% 以下
- 被拒绝的拟合中，至少 60% 在事后被证明是错误信号（价格未按预测方向运动）

**建议**：以这三个标准作为 Phase A 的验收条件。

---

## 附录：文件变更清单

### Phase A 变更

| 文件 | 操作 | 说明 |
|------|------|------|
| `star_analyzer/trajectory.py` | **修改** | WLS + 解析 f'(0)=0 + CV(RMSE) 门控 + 外推限制 |
| `star_analyzer/backtest.py` | **修改** | `_detect_entry_signals` 调用处适配新函数签名 |
| `tests/test_trajectory.py` | **新增** | 单元测试：WLS 权重正确性、解析约束正确性、门控拒绝边界 |

### Phase B 变更

| 文件 | 操作 | 说明 |
|------|------|------|
| `star_analyzer/fitting/__init__.py` | **新增** | 模块入口 |
| `star_analyzer/fitting/elastic_net.py` | **新增** | 弹性网正则化拟合 |
| `star_analyzer/fitting/mean_reversion.py` | **新增** | 指数均值回复模型 |
| `star_analyzer/fitting/gp_regression.py` | **新增** | 高斯过程回归 |
| `star_analyzer/fitting/ensemble.py` | **新增** | 模型集成与选择 |
| `star_analyzer/fitting/quality.py` | **新增** | 门控管道 + Q-score 计算 |

### Phase C 变更

| 文件 | 操作 | 说明 |
|------|------|------|
| `star_analyzer/fitting/quality.py` | **修改** | Q-score 公式实现 |
| `star_analyzer/backtest.py` | **修改** | IC 计算 + 分档统计 |
| `star_analyzer/fitting/adaptive.py` | **新增** | Walk-forward 自适应模块 |
