from collections import namedtuple
import jax
import jax.numpy as jnp
from scipy.optimize import bisect
from scipy.interpolate import interp1d
import time
import matplotlib.pyplot as plt

# ------------------------------
jax.config.update("jax_enable_x64", True)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']
dpi = 300

# ------------------------------
# Numerical constants
EPS = 1e-10  # Small number to avoid division by zero
SOLVER_TOL = 1e-5  # Solver tolerance

# ------------------------------
# Firms(Cobb-Douglas Production Function)
Firm = namedtuple('Firm', ('A', 'N', 'α', 'δ'))
def create_firm(A=1.0, N=1.0, α=0.33, δ=0.05):
    return Firm(A=A, N=N, α=α, δ=δ)

def r_given_k(K, firm):
    A, N, α, δ = firm
    return A * α * (N / K)**(1 - α) - δ

def r_to_w(r, firm):
    A, N, α, δ = firm
    return A * (1 - α) * (A * α / (r + δ))**(α / (1 - α))

# ------------------------------
# Household
# Modified to include m_grid (cash-on-hand grid) independent of z
Household = namedtuple('Household', ('β', 'a_grid', 'm_grid', 'z_grid', 'Π'))
def create_household(β=0.96, 
                     Π=[[0.9, 0.1], [0.1, 0.9]], 
                     z_grid=[0.1, 1.0], 
                     a_min=1e-10,
                     a_max=55,
                     a_size=2750,
                     m_size=2750,     # External grid size for policy function
                     r_max=0.06,     # Estimated max interest rate for grid range
                     w_max=1.5):     # Estimated max wage for grid range
    """
    Create household with separate a_grid (internal EGM) and m_grid (external policy).
    
    Following ezegm project design:
    - a_grid: used internally by EGM for Euler equation inversion
    - m_grid: unified grid for policy function, independent of z
    - m_grid covers next-period states: m_max = R*a_max + w*z_max
    """
    z_grid = jnp.array(z_grid)
    Π = jnp.array(Π)
    
    # Asset grid (internal for EGM)
    a_grid = jnp.linspace(a_min, a_max, a_size)
    
    # Compute m_grid range to cover next-period states
    R_max = 1 + r_max
    z_max = float(z_grid[-1])
    m_max = R_max * a_max + w_max * z_max  # Cover max possible m'
    
    # Unified m_grid (external, independent of z)
    m_grid = jnp.linspace(0.0, m_max, m_size)
    
    return Household(β=β, a_grid=a_grid, m_grid=m_grid, z_grid=z_grid, Π=Π)

Prices = namedtuple('Prices', ('r', 'w'))
def create_prices(r, w):
    return Prices(r=r, w=w)

# ------------------------------
# Power transformation for Epstein-Zin preferences
# Following the ezegm project: W = V^(1-ρ) where ρ = 1/ψ
def V_to_W(V, ρ):
    """Transform value function V to W = V^(1-ρ)"""
    V_safe = jnp.maximum(V, EPS)
    return jnp.where(
        jnp.abs(ρ - 1.0) < 1e-8,
        jnp.log(V_safe),
        V_safe**(1 - ρ)
    )

def W_to_V(W, ρ):
    """Transform W back to value function V = W^(1/(1-ρ))"""
    W_safe = jnp.maximum(W, EPS)
    return jnp.where(
        jnp.abs(ρ - 1.0) < 1e-8,
        jnp.exp(W_safe),
        W_safe**(1 / (1 - ρ))
    )

# ------------------------------
# CRRA utility function
def crra_utility(c, γ):
    """CRRA utility with risk aversion γ"""
    c_safe = jnp.maximum(c, EPS)
    return jnp.where(
        jnp.abs(γ - 1.0) < 1e-8,
        jnp.log(c_safe),
        (c_safe**(1 - γ) - 1) / (1 - γ)
    )

# ------------------------------
# Interpolation helper (JAX-friendly)
def interpolate_wrapper(x_new, x, y):
    """Linear interpolation with extrapolation"""
    return jnp.interp(x_new, x, y)

# ------------------------------
# EGM Algorithm for Epstein-Zin preferences
# Key insight from ezegm: use power transformation W = V^(1-ρ) to make EGM work

@jax.jit
def ezegm_step(V, c_policy, household, prices, γ, ψ):
    """
    One step of EZ-EGM algorithm (corrected implementation with unified m_grid)
    
    Following the ezegm project implementation exactly:
    1. c_policy and V are defined on (m_grid, z_grid) - unified m_grid independent of z
    2. Compute μ(a,z) and Ξ(a,z) on asset grid
    3. Invert Euler equation: c = (βR μ^(1-θ) Ξ)^(-1/ρ)
    4. Build endogenous m grid: m = c + a
    5. Interpolate back to unified m_grid
    6. Update value function
    
    Parameters:
    -----------
    V : array (m_size, z_size)
        Current value function on (m_grid, z_grid)
    c_policy : array (m_size, z_size)
        Current consumption policy on (m_grid, z_grid)
    household : Household
        Household parameters including m_grid
    prices : Prices
        Prices (r, w)
    γ : float
        Risk aversion
    ψ : float
        Elasticity of intertemporal substitution (EIS)
    
    Returns:
    --------
    V_new : array (m_size, z_size)
        Updated value function on (m_grid, z_grid)
    c_policy_new : array (m_size, z_size)
        Updated consumption policy on (m_grid, z_grid)
    """
    β, a_grid, m_grid, z_grid, Π = household
    r, w = prices
    n_m, n_a, n_z = len(m_grid), len(a_grid), len(z_grid)
    R = 1 + r
    
    # Convert parameters: ρ = 1/ψ (inverse of EIS)
    ρ = 1.0 / ψ
    
    # Compute θ = (1-γ)/(1-ρ) as in ezegm
    θ = jnp.where(jnp.abs(ρ - 1.0) > 1e-8, (1 - γ) / (1 - ρ), (1 - γ))
    
    # =========================================================================
    # Step 1: Compute next-period m' = R*a + w*z' for all (a, z') pairs
    # Then interpolate c and V at (m', z')
    # =========================================================================
    m_next = R * a_grid[:, None] + w * z_grid[None, :]  # shape (n_a, n_z)
    
    # Batch interpolation: for each z', interpolate c and V from m_grid
    def batch_interp_column(k):
        """Interpolate c and V for z'=k at all a grid points."""
        m_q = m_next[:, k]
        # Interpolate from m_grid (unified, independent of z)
        c_k = jnp.interp(m_q, m_grid, c_policy[:, k])
        V_k = jnp.interp(m_q, m_grid, V[:, k])
        return c_k, V_k
    
    # Vectorize over z' columns
    c_next, V_next = jax.vmap(batch_interp_column)(jnp.arange(n_z))
    c_next = jnp.maximum(c_next.T, EPS)  # shape (n_a, n_z)
    V_next = jnp.maximum(V_next.T, EPS)  # shape (n_a, n_z)
    
    # =========================================================================
    # Step 2: Compute W and expectations on the a-grid
    # =========================================================================
    W_next = V_next ** (1 - ρ)  # inline _V_to_W
    
    # Certainty equivalent μ(a, z) = (E[W^θ | z])^{1/θ}
    W_theta = W_next ** θ
    E_W_theta = W_theta @ Π.T  # shape (n_a, n_z) - expectation over z'
    E_W_theta = jnp.maximum(E_W_theta, EPS)
    μ = E_W_theta ** (1 / θ)
    
    # Euler integrand expectation
    euler_integrand = (W_next ** (θ - 1)) * (c_next ** (-ρ))
    E_euler = euler_integrand @ Π.T  # shape (n_a, n_z)
    
    # =========================================================================
    # Step 3: Invert Euler equation to get c on endogenous grid
    # =========================================================================
    rhs = jnp.maximum(β * R * (μ ** (1 - θ)) * E_euler, EPS)
    c_endog = rhs ** (-1 / ρ)  # shape (n_a, n_z)
    
    # =========================================================================
    # Step 4: Endogenous grid m_endog = c + a, with constraint boundary
    # =========================================================================
    m_endog = c_endog + a_grid[:, None]  # shape (n_a, n_z)
    
    # Prepend (m=0, c=0, μ=μ[0]) for constraint boundary
    m_endog = jnp.vstack([jnp.zeros(n_z), m_endog])  # shape (n_a+1, n_z)
    c_endog = jnp.vstack([jnp.zeros(n_z), c_endog])  # shape (n_a+1, n_z)
    μ_endog = jnp.vstack([μ[0, :], μ])  # shape (n_a+1, n_z) - prepend μ at a=0
    
    # =========================================================================
    # Step 5: Interpolate c AND μ from endogenous grid to unified m_grid
    # =========================================================================
    def interp_to_m_grid(j):
        """For income state z_j, interpolate c and μ onto m_grid."""
        c_j = jnp.interp(m_grid, m_endog[:, j], c_endog[:, j])
        μ_j = jnp.interp(m_grid, m_endog[:, j], μ_endog[:, j])
        return c_j, μ_j
    
    c_out, μ_out = jax.vmap(interp_to_m_grid)(jnp.arange(n_z))
    c_out = jnp.maximum(c_out.T, EPS)  # shape (n_m, n_z)
    μ_out = jnp.maximum(μ_out.T, EPS)  # shape (n_m, n_z)
    
    # Ensure consumption doesn't exceed cash-on-hand
    c_out = jnp.minimum(c_out, m_grid[:, None] - EPS)
    c_out = jnp.maximum(c_out, EPS)
    
    # =========================================================================
    # Step 6: Compute V directly from c and μ
    # =========================================================================
    W_out = (1 - β) * (c_out ** (1 - ρ)) + β * μ_out
    V_out = jnp.maximum(W_out, EPS) ** (1 / (1 - ρ))  # inline _W_to_V
    
    return V_out, c_out



def solve_household_egm(household, prices, γ, ψ, tol=1e-5, max_iter=1000, verbose=False):
    """
    Solve household problem using EGM for Epstein-Zin preferences
    
    Parameters:
    -----------
    household : Household
        Household parameters (includes m_grid)
    prices : Prices
        Prices (r, w)
    γ : float
        Risk aversion
    ψ : float
        Elasticity of intertemporal substitution
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations
    verbose : bool
        Print progress
    
    Returns:
    --------
    c_policy : array (m_size, z_size)
        Optimal consumption policy on (m_grid, z_grid)
    V : array (m_size, z_size)
        Value function on (m_grid, z_grid)
    """
    β, a_grid, m_grid, z_grid, Π = household
    r, w = prices
    m_size, z_size = len(m_grid), len(z_grid)
    
    # Initialize value function and consumption policy on m_grid
    # Initial guess: consume a fraction of cash-on-hand
    V = jnp.ones((m_size, z_size))
    c_policy = jnp.outer(m_grid, jnp.ones(z_size)) * 0.1 + 0.01
    
    # Ensure initial consumption is positive and less than m
    c_policy = jnp.maximum(c_policy, EPS)
    c_policy = jnp.minimum(c_policy, m_grid[:, None] - EPS)
    
    # Iteration
    for iteration in range(max_iter):
        V_new, c_new = ezegm_step(V, c_policy, household, prices, γ, ψ)
        
        # Check convergence
        error_V = jnp.max(jnp.abs(V_new - V))
        error_c = jnp.max(jnp.abs(c_new - c_policy))
        
        if verbose and iteration % 10 == 0:
            print(f"Iteration {iteration}: error_V = {error_V:.6e}, error_c = {error_c:.6e}")
        
        # Update
        V = V_new
        c_policy = c_new
        
        # Check convergence
        if error_V < tol and error_c < tol:
            if verbose:
                print(f"Converged in {iteration} iterations")
            break
    
    if iteration == max_iter - 1:
        print(f"Warning: Did not converge in {max_iter} iterations. Final error: {error_V:.6e}")
    
    return c_policy, V


@jax.jit
def get_policy_from_consumption(c_policy, household, prices):
    """
    Convert consumption policy on m_grid to asset policy indices on a_grid
    (Vectorized implementation for performance)
    
    Parameters:
    -----------
    c_policy : array (m_size, z_size)
        Consumption policy on (m_grid, z_grid)
    household : Household
        Household parameters (includes m_grid)
    prices : Prices
        Prices
    
    Returns:
    --------
    σ : array (a_size, z_size)
        Asset policy indices on (a_grid, z_grid)
    """
    β, a_grid, m_grid, z_grid, Π = household
    r, w = prices
    a_size, z_size = len(a_grid), len(z_grid)
    R = 1 + r
    
    # Compute cash-on-hand for all (a, z) combinations
    # m[i, j] = R * a_grid[i] + w * z_grid[j]
    m_all = R * a_grid[:, None] + w * z_grid[None, :]  # shape (a_size, z_size)
    
    # For each z state, interpolate consumption from m_grid
    def interp_for_z(j):
        """Interpolate consumption for income state z_j"""
        m_j = m_all[:, j]  # shape (a_size,)
        c_j = jnp.interp(m_j, m_grid, c_policy[:, j])  # shape (a_size,)
        return c_j
    
    # Vectorize over z states
    c_all = jax.vmap(interp_for_z)(jnp.arange(z_size))  # shape (z_size, a_size)
    c_all = c_all.T  # shape (a_size, z_size)
    
    # Ensure consumption bounds
    c_all = jnp.maximum(c_all, EPS)
    c_all = jnp.minimum(c_all, m_all - EPS)
    
    # Next period asset: a' = m - c
    ap_all = m_all - c_all  # shape (a_size, z_size)
    
    # Find closest index in a_grid for each a'
    # Use searchsorted for efficient index finding
    def find_closest_idx(ap_col):
        """Find closest a_grid index for each a' value"""
        # Binary search to find insertion point
        idx = jnp.searchsorted(a_grid, ap_col)
        # Clamp to valid range
        idx = jnp.clip(idx, 0, a_size - 1)
        # Check if previous index is closer
        idx_prev = jnp.maximum(idx - 1, 0)
        dist_curr = jnp.abs(a_grid[idx] - ap_col)
        dist_prev = jnp.abs(a_grid[idx_prev] - ap_col)
        # Choose the closer index
        idx = jnp.where(dist_prev < dist_curr, idx_prev, idx)
        return idx
    
    # Vectorize over z states
    σ = jax.vmap(find_closest_idx, in_axes=1, out_axes=1)(ap_all)  # shape (a_size, z_size)
    
    return σ.astype(jnp.int64)


# ------------------------------
# Stationary distribution computation (same as original)

@jax.jit
def compute_stationary(P):
    n = P.shape[0]
    I = jnp.identity(n)
    O = jnp.ones((n, n))
    A = I - jnp.transpose(P) + O
    return jnp.linalg.solve(A, jnp.ones(n))

@jax.jit
def compute_asset_stationary(σ, household):
    β, a_grid, m_grid, z_grid, Π = household
    a_size, z_size = len(a_grid), len(z_grid)
    
    # define the function to compute the transition row
    def compute_transition_row(a_idx, z_idx):
        ap_idx = σ[a_idx, z_idx]
        probs = jnp.zeros((a_size * z_size,))
        
        def body_fun(carry, zp_idx):
            probs = carry
            # Compute the index of the target state in the spreading array 
            target_idx = ap_idx + zp_idx * a_size
            # Setting the transfer probability   
            probs = probs.at[target_idx].set(Π[z_idx, zp_idx])
            return probs, None
        
        probs, _ = jax.lax.scan(body_fun, probs, jnp.arange(z_size))
        
        # normalisation
        probs = probs / jnp.sum(probs)
        return probs # size = (a_size * z_size,)
    
    # define the function to compute the transition matrix
    compute_transition_row_vmap = jax.vmap(lambda z_idx: jax.vmap(
            lambda a_idx: compute_transition_row(a_idx, z_idx)
        )(jnp.arange(a_size))
    ) # size = (a_size, a_size * z_size) 
    P_rows = compute_transition_row_vmap(jnp.arange(z_size)) # size = (z_size, a_size, a_size * z_size)
    
    # Reshape to the correct shape of the transfer matrix
    P_σ = jnp.reshape(P_rows, (z_size * a_size, a_size * z_size))
     
    # normalisation
    row_sums = jnp.sum(P_σ, axis=1)
    P_σ = P_σ / row_sums[:, None]
    
    # compute staitonary
    g = compute_stationary(P_σ)
    
    # Reshape to (z_size, a_size) to match state space alignment
    g = jnp.reshape(g, (z_size, a_size))
    # Transpose to get (a_size, z_size) shape
    g = jnp.transpose(g)
    # Calculating the marginal distribution of assets
    g_a = jnp.sum(g, axis=1)
    # normalisation
    g_a = g_a / jnp.sum(g_a)
    
    return g, g_a

def capital_supply(σ, household):
    """
    Compute the total capital supply given the policy σ.
    """
    β, a_grid, m_grid, z_grid, Π = household
    _, g_a = compute_asset_stationary(σ, household)
    return float(jnp.sum(g_a * a_grid))

# ------------------------------
# Equilibrium computation

def G(K, firm, household, γ, ψ, verbose=False):
    """
    Equilibrium mapping: given K, compute household's asset supply
    """
    # get the enterprise price given K
    r = r_given_k(K, firm)
    w = r_to_w(r, firm)
    prices = create_prices(r=r, w=w)
    
    # Solve household problem using EGM
    c_policy, V = solve_household_egm(household, prices, γ, ψ, verbose=verbose)
    
    # Convert to asset policy indices
    σ = get_policy_from_consumption(c_policy, household, prices)
    
    return capital_supply(σ, household)

def compute_equilibrium(firm, household, γ, ψ, a=1, b=20, xtol=1e-6):
    def objective(k):
        return k - G(k, firm, household, γ, ψ)
    
    # dynamically extend the search interval until a suitable interval is found
    left, right = a, b
    max_attempts = 10
    for _ in range(max_attempts):
        f_left = objective(left)
        f_right = objective(right)
        
        if f_left * f_right < 0:  # if a suitable interval is found
            break
            
        # extend the search interval
        left *= 0.5
        right *= 1.5
        
    if f_left * f_right >= 0:
        raise ValueError("No solution found in the given range")
                
    # solve using bisection
    K = bisect(objective, left, right, xtol=xtol)
    return K

# ------------------------------
# Main execution

if __name__ == '__main__':
    firm = create_firm()
    household = create_household()
    a_grid = household.a_grid
    z_grid = household.z_grid

    # =========================================================================
    # Benchmark: Measure single household problem solve time (EGM)
    # =========================================================================
    print("\n" + "=" * 60)
    print("BENCHMARK: Single Household Problem Solve Time (EGM)")
    print("=" * 60)
    
    # Use fixed test prices
    test_r = 0.03
    test_w = 1.0
    test_prices = create_prices(r=test_r, w=test_w)
    print(f"Test prices: r={test_r:.4f}, w={test_w:.4f}")
    print(f"Grid sizes: a_size={len(household.a_grid)}, m_size={len(household.m_grid)}")
    
    # Warm-up run (JIT compilation)
    print("\nWarm-up run (JIT compilation)...")
    _ = solve_household_egm(household, test_prices, γ=2.0, ψ=4.0, max_iter=5, verbose=False)
    
    # Timed run
    print("\nTimed run...")
    start_egm = time.time()
    c_test, V_test = solve_household_egm(household, test_prices, γ=2.0, ψ=4.0, tol=1e-5, verbose=True)
    time_egm = time.time() - start_egm
    print(f"\n>>> EGM single solve time: {time_egm*1000:.1f} ms ({time_egm:.3f} seconds)")
    
    # Also measure policy conversion and distribution calculation
    print("\nMeasuring policy conversion...")
    start_policy = time.time()
    σ_test = get_policy_from_consumption(c_test, household, test_prices)
    time_policy = time.time() - start_policy
    print(f">>> Policy conversion time: {time_policy*1000:.1f} ms")
    
    print("\nMeasuring stationary distribution calculation...")
    start_dist = time.time()
    g_test, g_a_test = compute_asset_stationary(σ_test, household)
    time_dist = time.time() - start_dist
    print(f">>> Distribution calculation time: {time_dist*1000:.1f} ms")
    
    print("\n" + "-" * 40)
    print("TIMING BREAKDOWN:")
    print(f"  EGM solve:           {time_egm*1000:8.1f} ms")
    print(f"  Policy conversion:   {time_policy*1000:8.1f} ms")
    print(f"  Distribution calc:   {time_dist*1000:8.1f} ms")
    print(f"  TOTAL:               {(time_egm+time_policy+time_dist)*1000:8.1f} ms")
    print("=" * 60)

    # =========================================================================
    # Equilibrium computation
    # =========================================================================
    # use bisect to compute the equilibrium capital
    print("\n\nCompute the equilibrium capital using EGM method")
    start = time.time()
    try:
        K_star = compute_equilibrium(firm, household, γ=2.0, ψ=4.0)
        elapsed = time.time() - start
        print(f"Compute the equilibrium capital {K_star:.5f}, time cost {elapsed:.2f} seconds")
    except ValueError as e:
        print(f"Error: {e}")
        
    # compute the equilibrium price
    r_star = r_given_k(K_star, firm)
    w_star = r_to_w(r_star, firm)
    prices_star = create_prices(r=r_star, w=w_star)
    
    # compute the optimal strategy using the equilibrium price
    c_policy_star, V_star = solve_household_egm(household, prices_star, γ=2.0, ψ=4.0, verbose=True)
    σ_star_eq = get_policy_from_consumption(c_policy_star, household, prices_star)
        
    # print the equilibrium price information
    print(f"Equilibrium interest rate: r_star = {r_star:.4f}")
    print(f"Equilibrium wage: w_star = {w_star:.4f}")
    
    # visualize the stationary distribution and asset distribution
    g, g_a = compute_asset_stationary(σ_star_eq, household)
    # compute the distribution g(a,z=0.1)
    g_a_z1 = g[:, 0]
    # compute the distribution g(a,z=1.0)
    g_a_z2 = g[:, 1]
    print(sum(g_a_z1))
    print(sum(g_a_z2))
    print("the sum of the stationary distribution：", g_a_z1.sum()+g_a_z2.sum())
        
    # visualize the marginal distribution of assets
    fig, ax = plt.subplots(figsize=(10, 8))
    n_bins = 75  
    ax.hist(a_grid, weights=g_a, density=True, bins=n_bins, alpha=0.7, color='#4169e1')
    plt.show()
    
    # visualize the distribution of assets g(a,z=0.1) and g(a,z=1.0) in one figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    hist_z1, bin_edges_z1 = jnp.histogram(a_grid, weights=g_a_z1, bins=n_bins)
    hist_z2, bin_edges_z2 = jnp.histogram(a_grid, weights=g_a_z2, bins=n_bins)
    
    # compute the center of the bins
    bin_centers_z1 = (bin_edges_z1[:-1] + bin_edges_z1[1:]) / 2
    bin_centers_z2 = (bin_edges_z2[:-1] + bin_edges_z2[1:]) / 2
    
    # visualize the two distributions g(a,z=0.1) and g(a,z=1.0)
    ax.hist(a_grid, weights=g_a_z1, bins=n_bins, alpha=0.3, color='blue', label=f'z = {z_grid[0]}')
    ax.hist(a_grid, weights=g_a_z2, bins=n_bins, alpha=0.3, color='red', label=f'z = {z_grid[1]}')
    
    # connect the top of the histogram to form a curve
    ax.plot(bin_centers_z1, hist_z1, color='blue', linewidth=2, linestyle='-')
    ax.plot(bin_centers_z2, hist_z2, color='red', linewidth=2, linestyle='--')
    
    # add a vertical dashed line at a_min
    a_min = household.a_grid[0]  # get the value of a_min
    ax.axvline(x=a_min, color='black', linestyle='--', linewidth=1.5)
    # add a underline a mark at the bottom, lower the y coordinate so it does not overlap with 0
    ax.text(a_min, -0.0015, r'$\underline{a}$', fontsize=12, horizontalalignment='center')
    
    ax.grid(False)
    ax.legend(fontsize=12)
    
    ax.set_xlim(left=-3, right=55)
    
    plt.tight_layout()
    plt.show()
    
    # visualize the consumption and saving functions under different ψ
    print("\nvisualize the consumption and saving functions under different ψ")
    
    gamma_fixed = 2.0
    psi_values = [0.2, 0.5, 0.8]

    # compute the consumption and saving functions for different ψ values
    fig_c_psi = plt.figure(figsize=(6, 4))
    ax_c_psi = fig_c_psi.add_subplot(111)
    
    fig_s_psi = plt.figure(figsize=(6, 4))
    ax_s_psi = fig_s_psi.add_subplot(111)
    
    colors = ['red', 'blue', 'green']
    
    ez_results = {}
    
    ez_results['psi_test'] = {}
    for psi in psi_values:
        try:
            print(f"\nTest γ = {gamma_fixed}, ψ = {psi}")
            K_star = compute_equilibrium(firm, household, γ=gamma_fixed, ψ=psi)
            r_star = r_given_k(K_star, firm)
            w_star = r_to_w(r_star, firm)
            
            ez_results['psi_test'][psi] = (K_star, r_star, w_star)
            print(f"γ = {gamma_fixed}, ψ = {psi}: Equilibrium capital = {K_star:.4f}, Equilibrium interest rate = {r_star:.4f}, Equilibrium wage = {w_star:.4f}")
        except Exception as e:
            print(f"γ = {gamma_fixed}, ψ = {psi} failed: {e}")
            ez_results['psi_test'][psi] = (None, None, None)
    
    K_star1 = ez_results['psi_test'][psi_values[0]][0]
    r_star1 = ez_results['psi_test'][psi_values[0]][1]
    w_star1 = ez_results['psi_test'][psi_values[0]][2]
    
    K_star2 = ez_results['psi_test'][psi_values[1]][0]
    r_star2 = ez_results['psi_test'][psi_values[1]][1]
    w_star2 = ez_results['psi_test'][psi_values[1]][2]
    
    K_star3 = ez_results['psi_test'][psi_values[2]][0]
    r_star3 = ez_results['psi_test'][psi_values[2]][1]
    w_star3 = ez_results['psi_test'][psi_values[2]][2]

    prices_star1 = create_prices(r=r_star1, w=w_star1)
    prices_star2 = create_prices(r=r_star2, w=w_star2)
    prices_star3 = create_prices(r=r_star3, w=w_star3)
    prices_psi = [prices_star1, prices_star2, prices_star3]
    
    c_policy_psi1, _ = solve_household_egm(household, prices_star1, γ=gamma_fixed, ψ=psi_values[0])
    c_policy_psi2, _ = solve_household_egm(household, prices_star2, γ=gamma_fixed, ψ=psi_values[1])
    c_policy_psi3, _ = solve_household_egm(household, prices_star3, γ=gamma_fixed, ψ=psi_values[2])
    c_policies_psi = [c_policy_psi1, c_policy_psi2, c_policy_psi3]
    
    # compute the consumption and saving functions
    def compute_with_params(a_idx, z_idx, c_policy, household, prices):
        β, a_grid, z_grid, Π = household
        r, w = prices
        a = a_grid[a_idx]
        z = z_grid[z_idx]
        c = c_policy[a_idx, z_idx]
        ap = w * z + (1 + r) * a - c
        s = ap - a
            
        return c, s
    
    for idx, psi in enumerate(psi_values):
        
        # compute the consumption and saving functions for low income state (z=0.1)
        compute_vmap_z1 = jax.vmap(
            lambda a_idx: compute_with_params(a_idx, 0, c_policies_psi[idx], household, prices_psi[idx])
        )
        c_psi_z1, s_psi_z1 = compute_vmap_z1(jnp.arange(len(a_grid)))
        
        # compute the consumption and saving functions for high income state (z=1.0)
        # Creating interrupted indexes: empty 35 dots after every 100 dots drawn
        a_indices_mask = jnp.zeros(len(a_grid), dtype=bool)
        for i in range(0, len(a_grid), 135):  # One cycle every 135 points (100 points displayed, 35 points not displayed)
            end_idx = min(i + 100, len(a_grid))  # Ensure not to exceed the array range
            a_indices_mask = a_indices_mask.at[i:end_idx].set(True)  # Set 100 points to True
        a_indices_z2 = jnp.where(a_indices_mask)[0]  # Get the indices of True
        
        compute_vmap_z2 = jax.vmap(
            lambda a_idx: compute_with_params(a_idx, 1, c_policies_psi[idx], household, prices_psi[idx])
        )
        c_psi_z2_all, s_psi_z2_all = compute_vmap_z2(jnp.arange(len(a_grid)))
        
        # Select the interrupted points for drawing
        c_psi_z2 = c_psi_z2_all[a_indices_z2]
        s_psi_z2 = s_psi_z2_all[a_indices_z2]
        a_grid_z2 = a_grid[a_indices_z2]
        
        # visualize the consumption function
        ax_c_psi.scatter(a_grid, c_psi_z1, s=0.1, alpha=0.6, color=colors[idx])
        ax_c_psi.scatter(a_grid_z2, c_psi_z2, s=0.1, alpha=0.6, color=colors[idx])
        
        # visualize the saving function
        ax_s_psi.scatter(a_grid, s_psi_z1, s=0.1, alpha=0.6, color=colors[idx])
        ax_s_psi.scatter(a_grid_z2, s_psi_z2, s=0.1, alpha=0.6, color=colors[idx])
        
        # visualize the legend
        ax_c_psi.plot([], [], color=colors[idx], linestyle='-', label=f'$\\psi$ = {psi}, z = {z_grid[0]}')
        ax_c_psi.plot([], [], color=colors[idx], linestyle='--', label=f'$\\psi$ = {psi}, z = {z_grid[1]}')
        
        ax_s_psi.plot([], [], color=colors[idx], linestyle='-', label=f'$\\psi$ = {psi}, z = {z_grid[0]}')
        ax_s_psi.plot([], [], color=colors[idx], linestyle='--', label=f'$\\psi$ = {psi}, z = {z_grid[1]}')
    
    ax_c_psi.grid(True, alpha=0.3)
    ax_c_psi.legend(fontsize=6, loc='best')

    ax_s_psi.grid(True, alpha=0.3)
    ax_s_psi.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax_s_psi.legend(fontsize=6, loc='best')
    
    ax_c_psi.set_xlim(left=0, right=40)
    ax_s_psi.set_xlim(left=0, right=40)

    c_psi_max = max([jnp.max(c_psi_z1[a_grid <= 40]), jnp.max(c_psi_z2[a_grid_z2 <= 40])])
    ax_c_psi.set_ylim(top=c_psi_max * 1.3)
    
    s_psi_min = min([jnp.min(s_psi_z1[a_grid <= 40]), jnp.min(s_psi_z2[a_grid_z2 <= 40])])
    ax_s_psi.set_ylim(bottom=s_psi_min * 1.3)

    fig_c_psi.tight_layout()
    fig_s_psi.tight_layout()
    plt.show()
    
    # visualize the consumption and saving functions under different γ
    print("\nvisualize the consumption and saving functions under different γ")
    
    gamma_values = [2.0, 6.0, 10.0]
    psi_fixed = 0.5

    # compute the consumption and saving functions for different γ values
    fig_c_gamma = plt.figure(figsize=(6, 4))
    ax_c_gamma = fig_c_gamma.add_subplot(111)
    
    fig_s_gamma = plt.figure(figsize=(6, 4))
    ax_s_gamma = fig_s_gamma.add_subplot(111)
    
    colors = ['red', 'blue', 'green']
    
    ez_results['gamma_test'] = {}
    for gamma in gamma_values:
        try:
            print(f"\nTest γ = {gamma}, ψ = {psi_fixed}")
            K_star = compute_equilibrium(firm, household, γ=gamma, ψ=psi_fixed)
            r_star = r_given_k(K_star, firm)
            w_star = r_to_w(r_star, firm)
            
            ez_results['gamma_test'][gamma] = (K_star, r_star, w_star)
            print(f"γ = {gamma}, ψ = {psi_fixed}: Equilibrium capital = {K_star:.4f}, Equilibrium interest rate = {r_star:.4f}, Equilibrium wage = {w_star:.4f}")
        except Exception as e:
            print(f"γ = {gamma}, ψ = {psi_fixed} failed: {e}")
            ez_results['gamma_test'][gamma] = (None, None, None)
    
    K_star1 = ez_results['gamma_test'][gamma_values[0]][0]
    r_star1 = ez_results['gamma_test'][gamma_values[0]][1]
    w_star1 = ez_results['gamma_test'][gamma_values[0]][2]
    
    K_star2 = ez_results['gamma_test'][gamma_values[1]][0]
    r_star2 = ez_results['gamma_test'][gamma_values[1]][1]
    w_star2 = ez_results['gamma_test'][gamma_values[1]][2]
    
    K_star3 = ez_results['gamma_test'][gamma_values[2]][0]
    r_star3 = ez_results['gamma_test'][gamma_values[2]][1]
    w_star3 = ez_results['gamma_test'][gamma_values[2]][2]

    prices_star1 = create_prices(r=r_star1, w=w_star1)
    prices_star2 = create_prices(r=r_star2, w=w_star2)
    prices_star3 = create_prices(r=r_star3, w=w_star3)
    prices_gamma = [prices_star1, prices_star2, prices_star3]
    
    c_policy_gamma1, _ = solve_household_egm(household, prices_star1, γ=gamma_values[0], ψ=psi_fixed)
    c_policy_gamma2, _ = solve_household_egm(household, prices_star2, γ=gamma_values[1], ψ=psi_fixed)
    c_policy_gamma3, _ = solve_household_egm(household, prices_star3, γ=gamma_values[2], ψ=psi_fixed)
    c_policies_gamma = [c_policy_gamma1, c_policy_gamma2, c_policy_gamma3]
        
    for idx, gamma in enumerate(gamma_values):
        
        # compute the consumption and saving functions for low income state (z=0.1)
        compute_vmap_z1 = jax.vmap(
            lambda a_idx: compute_with_params(a_idx, 0, c_policies_gamma[idx], household, prices_gamma[idx])
        )
        c_gamma_z1, s_gamma_z1 = compute_vmap_z1(jnp.arange(len(a_grid)))
        
        # compute the consumption and saving functions for high income state (z=1.0)
        # Creating interrupted indexes: empty 35 dots after every 100 dots drawn
        a_indices_mask = jnp.zeros(len(a_grid), dtype=bool)
        for i in range(0, len(a_grid), 135):  # One cycle every 135 points (100 points displayed, 35 points not displayed)
            end_idx = min(i + 100, len(a_grid))  # Ensure not to exceed the array range
            a_indices_mask = a_indices_mask.at[i:end_idx].set(True)  # Set 100 points to True
        a_indices_z2 = jnp.where(a_indices_mask)[0]  # Get the indices of True
        
        compute_vmap_z2 = jax.vmap(
            lambda a_idx: compute_with_params(a_idx, 1, c_policies_gamma[idx], household, prices_gamma[idx])
        )
        c_gamma_z2_all, s_gamma_z2_all = compute_vmap_z2(jnp.arange(len(a_grid)))
        
        # Select the interrupted points for drawing
        c_gamma_z2 = c_gamma_z2_all[a_indices_z2]
        s_gamma_z2 = s_gamma_z2_all[a_indices_z2]
        a_grid_z2 = a_grid[a_indices_z2]
        
        # visualize the consumption function
        ax_c_gamma.scatter(a_grid, c_gamma_z1, s=0.1, alpha=0.6, color=colors[idx])
        ax_c_gamma.scatter(a_grid_z2, c_gamma_z2, s=0.1, alpha=0.6, color=colors[idx])
        
        # visualize the saving function
        ax_s_gamma.scatter(a_grid, s_gamma_z1, s=0.1, alpha=0.6, color=colors[idx])
        ax_s_gamma.scatter(a_grid_z2, s_gamma_z2, s=0.1, alpha=0.6, color=colors[idx])
        
        # visualize the legend
        ax_c_gamma.plot([], [], color=colors[idx], linestyle='-', label=f'$\\gamma$ = {gamma}, z = {z_grid[0]}')
        ax_c_gamma.plot([], [], color=colors[idx], linestyle='--', label=f'$\\gamma$ = {gamma}, z = {z_grid[1]}')
        
        ax_s_gamma.plot([], [], color=colors[idx], linestyle='-', label=f'$\\gamma$ = {gamma}, z = {z_grid[0]}')
        ax_s_gamma.plot([], [], color=colors[idx], linestyle='--', label=f'$\\gamma$ = {gamma}, z = {z_grid[1]}')
    
    ax_c_gamma.grid(True, alpha=0.3)
    ax_c_gamma.legend(fontsize=6, loc='best')

    ax_s_gamma.grid(True, alpha=0.3)
    ax_s_gamma.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax_s_gamma.legend(fontsize=6, loc='best')

    fig_c_gamma.tight_layout()
    fig_s_gamma.tight_layout()
    plt.show()
