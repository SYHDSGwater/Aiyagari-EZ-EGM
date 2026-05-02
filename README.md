# Aiyagari-EZ-EGM

本仓库实现一个带 Epstein-Zin recursive utility 的 Aiyagari 异质性家庭模型，并使用 Endogenous Grid Method (EGM) 求解家庭储蓄问题。代码基于公开项目 `ezegm` 的 E-Z EGM 思路改写，将单家庭 consumption-saving problem 扩展到一般均衡 Aiyagari 环境。

核心目标是：

- 用 EGM 求解 Epstein-Zin 偏好下的家庭最优消费和储蓄策略；
- 在给定资本 `K` 下计算价格、家庭资产供给和稳态分布；
- 通过二分搜索求解一般均衡资本、利率和工资；
- 保留一份 V2 实验代码，用于继续探索更快的分布迭代和 lottery interpolation 方法。

## 文件结构

```text
Aiyagari with EZ EGM/
├── aiyagari with e-z egm.py        # 当前 baseline：E-Z EGM + Aiyagari 均衡求解
├── aiyagari_ez_egm_v2.py           # V2 实验版：lottery interpolation + 分布迭代法
├── CODE_WALKTHROUGH.md             # 中文代码解读
├── docs/
│   └── ez_aiyagari_egm_algorithm.tex  # 算法介绍 TeX 文档
├── logs/                           # 运行日志，已在 .gitignore 中忽略
├── README.md
└── LICENSE
```

原始 `ezegm` 项目和论文 PDF 不放入本仓库提交历史。

## 环境依赖

建议使用 Python 3.10+。主要依赖：

```bash
pip install jax jaxlib numpy scipy matplotlib
```

如果需要编译 TeX 文档，还需要本地 LaTeX / Tectonic 环境。

## 运行 baseline

```bash
python "aiyagari with e-z egm.py"
```

baseline 脚本会依次执行：

1. 单次家庭 EGM 求解 benchmark；
2. 策略函数转资产策略；
3. 稳态资产分布计算；
4. 一般均衡资本搜索；
5. 均衡价格、分布和策略图。

运行时会自动在 `logs/` 下保存终端输出，例如：

```text
logs/run_YYYYMMDD_HHMMSS.log
```

日志采用边运行边刷新方式，即使在画图前中断，已打印内容也会尽量保留下来。

## 运行 V2 实验版

```bash
python aiyagari_ez_egm_v2.py
```

V2 代码用于在 baseline 上探索更快求解方法，结构分为：

- `Model`：firm、household、price 和网格构造；
- `Solver`：E-Z EGM、lottery policy、分布迭代、均衡搜索；
- `Comparison plots`：多组 `gamma` 和 `psi` 的 consumption / saving 对比图；
- `Main`：默认 benchmark 和均衡计算。

V2 的画图比较块默认关闭：

```python
RUN_COMPARISON_PLOTS = False
```

需要画多组参数对比时，可以改为 `True`，或直接调用 `plot_consumption_saving_comparisons(..., enabled=True)`。

## 当前默认校准

baseline 和 V2 的默认 Aiyagari 校准为：

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `discount` / `β` | 主观贴现因子 | 0.96 |
| `gamma` / `γ` | 风险厌恶系数 | 2.0 |
| `psi` / `ψ` | 跨期替代弹性 EIS | 4.0 |
| `capital_share` / `α` | 资本份额 | 0.33 |
| `depreciation` / `δ` | 折旧率 | 0.05 |
| `income_grid` | 收入状态 | `[0.1, 1.0]` |
| `transition` / `Π` | 收入转移矩阵 | `[[0.9, 0.1], [0.1, 0.9]]` |
| `asset_size`, `cash_size` | 网格规模 | 2750 |

在当前默认参数下，本地测试得到的均衡大约为：

```text
K = 7.7049
r = 0.0340
w = 1.3143
```

也就是均衡利率约为 3.4%。

## 算法概要

家庭状态变量采用 cash-on-hand `m` 和收入状态 `z`。EGM 内部使用下一期资产 `a'` 作为外生网格：

1. 给定当前猜测的消费函数和价值函数；
2. 在资产网格上计算下一期 cash-on-hand；
3. 插值得到下一期消费和价值；
4. 计算 Epstein-Zin 确定性等价项和 Euler 方程右侧；
5. 直接反解当前消费，无需 root-finding；
6. 用 `m = c + a'` 构造 endogenous grid；
7. 插值回统一的 `m_grid`；
8. 更新价值函数和消费策略；
9. 收敛后计算资产策略和稳态分布；
10. 在一般均衡外层用二分法搜索资本市场出清。

V2 与 baseline 的主要差异是：baseline 将连续资产策略映射到最近网格点，再构造转移矩阵求稳态分布；V2 使用 lottery interpolation，把资产策略质量分配到相邻两个资产网格点，并直接迭代分布，从而避免构造和求解大型稠密转移矩阵。

## 性能说明

本项目求解的是完整 Aiyagari 一般均衡，不是原始 `ezegm` 论文中的单家庭 consumption-saving benchmark。原论文中 100ms 以内的结果对应较小网格、JAX JIT warmup 后的单家庭 E-Z EGM 求解；本项目默认网格为 `2750 x 2`，并且一般均衡搜索会多次调用家庭求解器和分布计算。

最近一次本地 V2 默认参数测试：

```text
K = 7.70487404
r = 0.03401968
w = 1.31433911
full equilibrium elapsed ≈ 16.5s
last G(K) elapsed ≈ 0.76s
```

计时会随 CPU / GPU、JAX backend、首次 JIT 编译和网格规模变化。

## 参考

- Aiyagari, S. R. (1994). Uninsured Idiosyncratic Risk and Aggregate Saving. *Quarterly Journal of Economics*.
- Epstein, L. G., & Zin, S. E. (1989). Substitution, Risk Aversion, and the Temporal Behavior of Consumption and Asset Returns. *Econometrica*.
- Lujan, A. (2026). *The Endogenous Grid Method for Epstein-Zin Preferences*. `ezegm` project.

## License

MIT License. See [LICENSE](LICENSE).
