# Aiyagari with Epstein-Zin Preferences (EGM)

Discrete-time, discrete-state **Aiyagari** heterogeneous-agent equilibrium with **Epstein-Zin** recursive utility, solved by the **Endogenous Grid Method (EGM)**. Implemented in **JAX** with a single-file script: household problem, stationary distribution, and equilibrium capital via bisection.

---

## Quick start

**Dependencies:** Python 3.8+, JAX, NumPy, SciPy, Matplotlib.

```bash
pip install jax jaxlib numpy scipy matplotlib
```

**Run:**

```bash
python "aiyagari with e-z egm.py"
```

The script prints benchmark timings, computes equilibrium capital and prices, and optionally shows asset distributions and policy plots. To save terminal output:

```bash
python "aiyagari with e-z egm.py" > run_log.txt 2>&1
```

---

## What this code does

- **Household problem:** Epstein-Zin preferences (risk aversion \(\gamma\), EIS \(\psi\)). EGM inverts the Euler equation on an endogenous grid using the power transformation \(W = V^{1-\rho}\), \(\rho = 1/\psi\); no root-finding in the inner loop. Policies are on a unified cash-on-hand grid `m_grid`; an internal asset grid `a_grid` is used for the EGM step.
- **Firm:** Cobb-Douglas production; interest rate and wage as functions of aggregate capital \(K\).
- **Equilibrium:** Bisection on \(K\) so that capital demand equals household asset supply. Stationary distribution is computed from the optimal policy and used to aggregate assets.

---

## EGM vs Howard Policy Iteration (speed)

The same Aiyagari–Epstein-Zin setup was run with **EGM** (this repo) and with **Howard policy iteration** (value-function iteration with policy improvement) on the same machine. Typical run (grid sizes `a_size = m_size = 2750`, default calibration):

| Metric | EGM | Howard PI |
|--------|-----|-----------|
| Single household solve | 305.6 ms | 329.6 ms |
| Policy conversion | 220.8 ms | — |
| Stationary distribution | 304.5 ms | 216.2 ms |
| **Total per household** | **830.8 ms** | **545.8 ms** |
| **Full equilibrium** \(K^*\) | **28.09 s** | **120.89 s** |

EGM converges in about 212 iterations for the reported run. For **full equilibrium**, EGM is roughly **4.3× faster** (28 s vs 121 s) because the equilibrium loop repeatedly solves the household problem; EGM’s convergence and cost per iteration make the overall search much cheaper than with Howard PI.

*(Numbers from Python terminal output; your timings may vary with hardware and JAX backend.)*

---

## Calibration (defaults)

| Parameter | Symbol | Value |
|-----------|--------|--------|
| Discount factor | $\beta$ | 0.96 |
| Risk aversion | $\gamma$ | 2.0 |
| EIS | $\psi$ | 4.0 |
| Income grid | $z$ | [0.1, 1.0] |
| Transition | $\Pi$ | 2×2 (e.g. 0.9/0.1) |
| Capital share | $\alpha$ | 0.33 |
| Depreciation | $\delta$ | 0.05 |
| Grid sizes | `a_size`, `m_size` | 2750 |

Edit `create_firm`, `create_household`, and the $\gamma,\psi$ arguments in the script to change calibration.

---

## File layout

```
Aiyagari with EZ EGM/
├── README.md
├── aiyagari with e-z egm.py   # EGM + equilibrium + plots
└── aiyagari with e-z _git.md   # Optional notes
```

---

## References

- Aiyagari, S. R. (1994). Uninsured Idiosyncratic Risk and Aggregate Saving. *Quarterly Journal of Economics*, 109(3), 659–684.
- Epstein, L. G., & Zin, S. E. (1989). Substitution, Risk Aversion, and the Temporal Behavior of Consumption and Asset Returns. *Econometrica*, 57(4), 937–969.

---

## License

MIT.
