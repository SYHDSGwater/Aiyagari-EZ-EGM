# 代码走读

这份文档解释 `aiyagari with e-z egm.py` 这个单文件项目。代码可以分成三层：

1. 企业端：给定总资本 `K`，计算利率 `r` 和工资 `w`。
2. 家庭端：用 EGM 求解 Epstein-Zin 偏好下的消费储蓄问题。
3. 均衡端：寻找使家庭资产供给等于企业资本需求的 Aiyagari 一般均衡。

## 1. 模型对象

模型中有异质性劳动收入风险和不完全市场。家庭进入当期时持有资产 `a`，并处于收入状态 `z`。给定价格后，当期 cash-on-hand 是：

```text
m = (1 + r) * a + w * z
```

家庭选择消费 `c`，下一期资产为：

```text
a' = m - c
```

代码通过把消费限制在 cash-on-hand 以下来执行借贷约束，所以数值上有：

```text
a' >= 0
```

企业端是 Cobb-Douglas 生产函数：

```text
Y = A K^alpha N^(1-alpha)
```

给定总资本 `K` 后，`r_given_k` 计算资本边际产出对应的净利率，`r_to_w` 计算同一套一阶条件下的工资。

## 2. 网格结构

家庭对象定义为：

```python
Household = namedtuple('Household', ('β', 'a_grid', 'm_grid', 'z_grid', 'Π'))
```

这里有两个容易混淆的网格。

`a_grid` 是 EGM 内部网格，表示期末资产，也就是家庭本期储蓄后的资产。EGM 的反向思路是：先固定一组可能的期末资产 `a'`，再通过欧拉方程反推出对应的消费和 cash-on-hand。

`m_grid` 是外部策略网格。消费策略 `c_policy` 和价值函数 `V` 都存储在统一的矩形网格上：

```text
m_grid x z_grid
```

这沿用了原始 `ezegm` 项目的设计。这样做很关键，因为下一期 cash-on-hand 同时取决于资产和下一期收入冲击：

```text
m' = (1 + r) * a' + w * z'
```

因此，策略函数应该按 cash-on-hand 插值，而不是直接按资产下标索引。

## 3. Epstein-Zin 变换

Epstein-Zin 偏好把风险厌恶和跨期替代弹性分开。代码里输入的是 `ψ`，即 EIS；然后设：

```python
ρ = 1.0 / ψ
θ = (1 - γ) / (1 - ρ)
```

其中：

```text
γ = 风险厌恶系数
ψ = 跨期替代弹性
ρ = EIS 的倒数
θ = Epstein-Zin 递归偏好里的辅助参数
```

原始 `ezegm` 项目的关键技巧是幂变换：

```text
W = V^(1-rho)
```

变换后，Bellman 聚合器变成加法形式：

```text
W(m,z) = (1-beta) * c(m,z)^(1-rho) + beta * mu(a,z)
```

其中确定性等价项为：

```text
mu(a,z) = E[ W(m',z')^theta | z ]^(1/theta)
```

这个变换把原来嵌套在价值函数里的非线性结构集中到 `mu` 里面，使欧拉方程可以显式反解。

## 4. 一次 EGM 迭代

核心函数是 `ezegm_step`。

输入包括：

```python
V
c_policy
household
prices
γ
ψ
```

其中 `V` 和 `c_policy` 的形状都是：

```text
(m_size, z_size)
```

也就是说，它们都定义在 `m_grid x z_grid` 上。

### 4.1 计算下一期 cash-on-hand

代码先在内部资产网格上计算所有可能的下一期 cash-on-hand：

```python
m_next = R * a_grid[:, None] + w * z_grid[None, :]
```

这里：

```text
R = 1 + r
```

`m_next` 的形状是：

```text
(a_size, z_size)
```

行对应资产网格点，列对应下一期收入状态。

### 4.2 插值得到下一期消费和值函数

对每一个下一期收入状态 `z'`，代码从 `m_grid` 上插值得到：

```text
c(m', z')
V(m', z')
```

这一步很重要：虽然 EGM 内部在 `a_grid` 上工作，但策略函数本身不是定义在 `a_grid` 上，而是定义在 `m_grid` 上。

### 4.3 转换到 W 空间

插值得到 `V_next` 后，代码计算：

```python
W_next = V_next ** (1 - ρ)
```

后面的确定性等价和欧拉方程都在这个变换后的 `W` 空间里表达。

### 4.4 计算确定性等价 mu

代码先计算：

```python
W_theta = W_next ** θ
E_W_theta = W_theta @ Π.T
μ = E_W_theta ** (1 / θ)
```

经济含义是：对下一期收入状态做条件期望，得到当前状态 `z` 下的 Epstein-Zin 确定性等价。

矩阵乘法 `@ Π.T` 的方向是为了得到：

```text
E[ ... | 当前 z ]
```

最终 `μ` 的形状也是：

```text
(a_size, z_size)
```

也就是每个期末资产和当前收入状态都有一个确定性等价。

### 4.5 计算欧拉期望项

Epstein-Zin 的欧拉方程不仅依赖下期消费，还依赖下期价值函数。代码中的欧拉期望项是：

```python
euler_integrand = (W_next ** (θ - 1)) * (c_next ** (-ρ))
E_euler = euler_integrand @ Π.T
```

这对应公式：

```text
Xi(a,z) = E[ W(m',z')^(theta-1) * c(m',z')^(-rho) | z ]
```

这是把 EGM 搬到 Epstein-Zin 偏好时最容易漏掉的部分。标准 CRRA EGM 只需要下期边际效用；Epstein-Zin EGM 还需要价值函数项。

### 4.6 反解欧拉方程

变换后的欧拉方程为：

```text
c^(-rho) = beta * R * mu(a,z)^(1-theta) * Xi(a,z)
```

因此可以直接反解消费：

```python
rhs = β * R * (μ ** (1 - θ)) * E_euler
c_endog = rhs ** (-1 / ρ)
```

这就是 EGM 的核心：不用在每个状态点上做数值优化或求根，而是直接由欧拉方程得到消费。

### 4.7 构造 endogenous grid

EGM 此时知道的是：

```text
期末资产 a
对应消费 c
```

所以可以反推出对应的 cash-on-hand：

```python
m_endog = c_endog + a_grid[:, None]
```

这组 `m_endog` 不是事先固定的外部网格，而是由欧拉方程反推出的 endogenous grid。

代码随后在前面补上约束边界：

```python
m = 0
c = 0
```

这个点用于锚定借贷约束附近的插值。

### 4.8 插回固定 m_grid

EGM 内部得到的是 `m_endog` 上的消费。为了进入下一轮迭代，代码需要把它插回统一的 `m_grid`：

```python
c_j = jnp.interp(m_grid, m_endog[:, j], c_endog[:, j])
μ_j = jnp.interp(m_grid, m_endog[:, j], μ_endog[:, j])
```

插值得到的 `c_out` 和 `μ_out` 又回到形状：

```text
(m_size, z_size)
```

代码还会把消费夹在可行范围内：

```python
c_out <= m
c_out >= EPS
```

### 4.9 更新价值函数

最后，根据 Bellman 方程更新 `W` 和 `V`：

```python
W_out = (1 - β) * (c_out ** (1 - ρ)) + β * μ_out
V_out = W_out ** (1 / (1 - ρ))
```

函数返回：

```python
V_out, c_out
```

## 5. 家庭问题求解器

`solve_household_egm` 负责反复调用 `ezegm_step`。

初始值为：

```python
V = ones
c_policy = 0.1 * m + 0.01
```

然后每轮计算：

```python
V_new, c_new = ezegm_step(...)
```

并检查两个误差：

```python
max(abs(V_new - V))
max(abs(c_new - c_policy))
```

原始 `ezegm` 项目主要监控策略函数收敛。这里同时检查价值函数和消费策略，比较保守，但也更容易发现值函数没有稳定的问题。

函数最终返回：

```python
c_policy
V
```

二者都定义在：

```text
m_grid x z_grid
```

## 6. 从消费策略转换为资产策略

稳态分布部分是在 `a_grid x z_grid` 上做的，所以需要把 `m_grid` 上的消费策略转换为资产策略。

函数 `get_policy_from_consumption` 对每个当前资产和收入状态计算：

```python
m = (1 + r) * a + w * z
```

然后从 `m_grid` 上插值得到消费：

```python
c = c_policy(m,z)
```

接着计算下一期资产：

```python
a' = m - c
```

由于分布算法使用离散状态，代码把连续的 `a'` 映射到最接近的 `a_grid` 下标。

这个 nearest-grid 做法简单，但会带来额外离散化误差。更精确的做法是 lottery/interpolation，把质量按距离分给相邻两个资产网格点。

## 7. 稳态分布

`compute_asset_stationary` 根据资产策略构造 Markov 转移矩阵。

一个状态由两个下标组成：

```text
(a_idx, z_idx)
```

资产策略给出下一期资产下标：

```text
ap_idx = σ[a_idx, z_idx]
```

收入转移矩阵 `Π` 给出下一期收入状态概率。因此，从当前状态 `(a_idx, z_idx)` 出发，概率质量会转移到：

```text
(ap_idx, z'_idx)
```

概率为：

```text
Π[z_idx, z'_idx]
```

构造完整转移矩阵后，`compute_stationary` 解不变分布。

返回的 `g` 形状为：

```text
(a_size, z_size)
```

`g_a` 是对收入状态求和后的资产边际分布。

## 8. 资本供给

给定稳态资产边际分布，家庭资产供给为：

```python
sum(g_a * a_grid)
```

这就是 Aiyagari 模型中家庭部门愿意持有的总资产，也就是资本供给。

## 9. 一般均衡

`G(K, ...)` 是均衡映射。给定一个候选总资本 `K`，它做五件事：

1. 用企业一阶条件计算 `r` 和 `w`。
2. 在这些价格下求解家庭问题。
3. 把消费策略转换成资产策略。
4. 计算稳态资产分布。
5. 返回家庭资产供给。

`compute_equilibrium` 要解的是：

```text
K = G(K)
```

代码写成：

```python
objective(k) = k - G(k)
```

然后用 SciPy 的 `bisect` 找零点。

如果初始区间 `[a,b]` 没有包含符号变化，代码会扩大搜索区间，再尝试二分法。

## 10. 主程序做了什么

`if __name__ == '__main__':` 下面的主程序主要用于跑 benchmark 和画图。

第一部分测量单次家庭问题求解的时间：

1. 用固定测试价格 `r=0.03, w=1.0`。
2. 先跑一次 warm-up，让 JAX 完成 JIT 编译。
3. 再正式计时。
4. 额外计时策略转换和稳态分布计算。

第二部分计算一般均衡：

1. 调 `compute_equilibrium` 得到 `K_star`。
2. 根据 `K_star` 得到 `r_star` 和 `w_star`。
3. 重新在均衡价格下求解家庭问题。
4. 计算资产策略和稳态分布。
5. 画资产分布。

第三部分做比较静态：

1. 固定 `γ`，改变 `ψ`，比较不同 EIS 下的消费和储蓄函数。
2. 固定 `ψ`，改变 `γ`，比较不同风险厌恶下的消费和储蓄函数。

## 11. 已修复的绘图辅助函数问题

主程序里 `compute_with_params` 用于画消费和储蓄函数。这个函数原来有两个明显问题：

1. `Household` 实际有五个字段，但原代码只解包四个字段。
2. `c_policy` 是定义在 `m_grid` 上的，但原代码直接用 `a_idx` 去索引 `c_policy[a_idx, z_idx]`。

正确逻辑应该是：给定当前资产 `a` 和收入状态 `z`，先计算 cash-on-hand：

```python
m = (1 + r) * a + w * z
```

再从 `m_grid` 上插值得到消费：

```python
c = jnp.interp(m, m_grid, c_policy[:, z_idx])
```

然后得到下一期资产和储蓄：

```python
ap = m - c
s = ap - a
```

现在代码已经按这个逻辑修正。

## 12. 还需要重点检查的地方

核心 Epstein-Zin EGM 公式已经和原始 `ezegm` 项目对齐，包括容易漏掉的欧拉期望项 `E_euler`。

剩下的风险主要在 Aiyagari 外层。

第一，`m_grid` 的范围是根据假定的最大价格 `r_max` 和 `w_max` 生成的。如果均衡搜索过程中实际价格超过这个范围，`jnp.interp` 会在边界处外推成常数，可能影响资产供给。

第二，`ψ = 1` 还没有真正支持。因为此时 `ρ = 1`，而主 EGM 步骤里有 `1 / (1 - ρ)`。原始 `ezegm` 项目对这个情形是直接报错，不是用普通公式处理。

第三，稳态分布现在构造的是 dense transition matrix。小网格没问题，但默认 `a_size=2750` 时矩阵会很大，内存和速度压力明显。

第四，资产策略转换使用 nearest-grid，而不是 lottery interpolation。这会引入额外离散化误差，尤其会影响资产分布的平滑性和资本供给精度。

第五，代码没有检查 endogenous grid 是否单调。典型校准下通常没问题，但如果出现非单调，直接 `jnp.interp` 就不够，需要 upper envelope 方法。
