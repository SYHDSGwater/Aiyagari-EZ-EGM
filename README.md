# Aiyagari with Epstein-Zin Preferences (EGM)

Discrete-time, discrete-state **Aiyagari** heterogeneous-agent equilibrium with **Epstein-Zin** recursive utility, solved by the **Endogenous Grid Method (EGM)**. Implemented in **JAX** with a single-file baseline script: household problem, stationary distribution, and equilibrium capital via bisection.

---

## Quick start

**Dependencies:** Python 3.8+, JAX, NumPy, SciPy, Matplotlib.

```bash
pip install jax jaxlib numpy scipy matplotlib
```

**Run the baseline version:**

```bash
python "aiyagari with e-z egm.py"
```

The script prints benchmark timings, computes equilibrium capital and prices, saves terminal output under `logs/`, and optionally shows asset distributions and policy plots.

**Run the V2 experimental version:**

```bash
python aiyagari_ez_egm_v2.py
```

---

## What this code does

- **Household problem:** Epstein-Zin preferences (risk aversion $\gamma$, EIS $\psi$). EGM inverts the Euler equation on an endogenous grid using the power transformation $W = V^{1-\rho}$, $\rho = 1/\psi$; no root-finding in the inner loop. Policies are on a unified cash-on-hand grid `m_grid`; an internal asset grid `a_grid` is used for the EGM step.
- **Firm:** Cobb-Douglas production; interest rate and wage as functions of aggregate capital $K$.
- **Equilibrium:** Bisection on $K$ so that capital demand equals household asset supply. Stationary distribution is computed from the optimal policy and used to aggregate assets.

---

## V2 Experimental Solver

`aiyagari_ez_egm_v2.py` is an experimental speed-oriented version built from the baseline. It keeps the same economic model but changes the distribution step:

- converts continuous saving choices into lottery weights on adjacent asset-grid points;
- iterates directly on the stationary distribution instead of building and solving a dense transition matrix;
- keeps model construction, solver logic, and comparison plotting in separate code blocks;
- includes a disabled-by-default comparison plotting block for multiple `gamma` and `psi` values.

Turn on the comparison plots by setting:

```python
RUN_COMPARISON_PLOTS = True
```

---

## EGM vs Howard Policy Iteration (speed)

The same Aiyagari-Epstein-Zin setup was run with **EZ-EGM** (this repo) and with **Howard policy iteration** (value-function iteration with policy improvement) on the same machine. Typical run (grid sizes `a_size = m_size = 2750`, default calibration):

| Metric | EZ-EGM | Howard PI |
|--------|--------|-----------|
| Single household solve | 296.3 ms | 329.6 ms |
| Policy conversion | 150.0 ms | - |
| Stationary distribution | 146.2 ms | 216.2 ms |
| **Total per household** | **592.5 ms** | **545.8 ms** |
| **Full equilibrium** $K^*$ | **18.28 s** | **120.89 s** |

The current baseline converges to approximately $K^* = 7.7048$ and $r^* = 3.40\%$ under the default calibration. V2 gives the same equilibrium rate in local tests and reduces the last capital-supply evaluation to about 0.76 seconds, with a full equilibrium solve around 16.5 seconds.

*(Numbers from Python terminal output; your timings may vary with hardware and JAX backend.)*

---

## Calibration (defaults)

| Parameter | Symbol | Value |
|-----------|--------|--------|
| Discount factor | $\beta$ | 0.96 |
| Risk aversion | $\gamma$ | 2.0 |
| EIS | $\psi$ | 4.0 |
| Income grid | $z$ | [0.1, 1.0] |
| Transition | $\Pi$ | 2x2 (e.g. 0.9/0.1) |
| Capital share | $\alpha$ | 0.33 |
| Depreciation | $\delta$ | 0.05 |
| Grid sizes | `a_size`, `m_size` | 2750 |

Edit `create_firm`, `create_household`, and the $\gamma,\psi$ arguments in the script to change calibration.

---

## File layout

```text
Aiyagari with EZ EGM/
├── README.md
├── LICENSE
├── aiyagari with e-z egm.py        # Baseline EZ-EGM + equilibrium + plots
├── aiyagari_ez_egm_v2.py           # V2 solver: lottery interpolation + distribution iteration
├── CODE_WALKTHROUGH.md             # Chinese code walkthrough
├── docs/
│   ├── ez_aiyagari_egm_algorithm.tex
│   └── ez_aiyagari_egm_algorithm.pdf
└── logs/                           # Runtime logs, ignored by git
```

---

## References

- Aiyagari, S. R. (1994). Uninsured Idiosyncratic Risk and Aggregate Saving. *Quarterly Journal of Economics*, 109(3), 659-684.
- Epstein, L. G., & Zin, S. E. (1989). Substitution, Risk Aversion, and the Temporal Behavior of Consumption and Asset Returns. *Econometrica*, 57(4), 937-969.
- Lujan, A. (2026). *The Endogenous Grid Method for Epstein-Zin Preferences*. `ezegm` project.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
