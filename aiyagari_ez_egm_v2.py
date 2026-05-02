"""
V2 experimental implementation for the Aiyagari model with Epstein-Zin
preferences solved by EGM.

Changes relative to the baseline script:
1. Use English parameter and field names only.
2. Replace nearest-grid asset policy with lottery interpolation.
3. Replace dense transition-matrix stationary distribution with distribution
   iteration.
4. Keep comparison plots in a separate block that is off by default.
"""

from collections import namedtuple
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

EPSILON = 1e-10
SOLVER_TOLERANCE = 1e-5
DISTRIBUTION_TOLERANCE = 1e-10
RUN_COMPARISON_PLOTS = False


# =============================================================================
# Model
# =============================================================================

Firm = namedtuple("Firm", ("productivity", "labor", "capital_share", "depreciation"))
Household = namedtuple(
    "Household",
    ("discount", "asset_grid", "cash_grid", "income_grid", "transition"),
)
Prices = namedtuple("Prices", ("interest_rate", "wage"))


def sync_jax(value):
    """Synchronize JAX work before measuring elapsed time."""
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
    elif isinstance(value, (tuple, list)):
        for item in value:
            sync_jax(item)
    elif isinstance(value, dict):
        for item in value.values():
            sync_jax(item)
    return value


def create_firm(
    productivity=1.0,
    labor=1.0,
    capital_share=0.33,
    depreciation=0.05,
):
    return Firm(
        productivity=productivity,
        labor=labor,
        capital_share=capital_share,
        depreciation=depreciation,
    )


def interest_rate_from_capital(capital, firm):
    productivity, labor, capital_share, depreciation = firm
    return productivity * capital_share * (labor / capital) ** (1 - capital_share) - depreciation


def wage_from_interest_rate(interest_rate, firm):
    productivity, labor, capital_share, depreciation = firm
    return (
        productivity
        * (1 - capital_share)
        * (productivity * capital_share / (interest_rate + depreciation))
        ** (capital_share / (1 - capital_share))
    )


def wage_from_capital(capital, firm):
    return wage_from_interest_rate(interest_rate_from_capital(capital, firm), firm)


def create_prices(interest_rate, wage):
    return Prices(interest_rate=interest_rate, wage=wage)


def create_household(
    discount=0.96,
    transition=((0.9, 0.1), (0.1, 0.9)),
    income_grid=(0.1, 1.0),
    asset_min=1e-10,
    asset_max=55.0,
    asset_size=2750,
    cash_size=2750,
    max_interest_rate=0.06,
    max_wage=1.5,
):
    income_grid = jnp.array(income_grid)
    transition = jnp.array(transition)

    asset_grid = jnp.linspace(asset_min, asset_max, asset_size)
    max_cash = (1 + max_interest_rate) * asset_max + max_wage * float(income_grid[-1])
    cash_grid = jnp.linspace(0.0, max_cash, cash_size)

    return Household(
        discount=discount,
        asset_grid=asset_grid,
        cash_grid=cash_grid,
        income_grid=income_grid,
        transition=transition,
    )


# =============================================================================
# Solver
# =============================================================================

@jax.jit
def ezegm_step(value, consumption, household, prices, gamma, psi):
    discount, asset_grid, cash_grid, income_grid, transition = household
    interest_rate, wage = prices
    asset_size, income_size = len(asset_grid), len(income_grid)
    gross_rate = 1 + interest_rate

    inverse_eis = 1.0 / psi
    ez_exponent = jnp.where(
        jnp.abs(inverse_eis - 1.0) > 1e-8,
        (1 - gamma) / (1 - inverse_eis),
        1 - gamma,
    )

    next_cash = gross_rate * asset_grid[:, None] + wage * income_grid[None, :]

    def interpolate_income(income_index):
        query_cash = next_cash[:, income_index]
        next_consumption = jnp.interp(query_cash, cash_grid, consumption[:, income_index])
        next_value = jnp.interp(query_cash, cash_grid, value[:, income_index])
        return next_consumption, next_value

    next_consumption, next_value = jax.vmap(interpolate_income)(jnp.arange(income_size))
    next_consumption = jnp.maximum(next_consumption.T, EPSILON)
    next_value = jnp.maximum(next_value.T, EPSILON)

    next_transformed_value = next_value ** (1 - inverse_eis)

    transformed_power = next_transformed_value**ez_exponent
    expected_transformed_power = jnp.maximum(transformed_power @ transition.T, EPSILON)
    certainty_equivalent = expected_transformed_power ** (1 / ez_exponent)

    euler_integrand = (next_transformed_value ** (ez_exponent - 1)) * (
        next_consumption ** (-inverse_eis)
    )
    euler_expectation = euler_integrand @ transition.T

    right_hand_side = jnp.maximum(
        discount
        * gross_rate
        * (certainty_equivalent ** (1 - ez_exponent))
        * euler_expectation,
        EPSILON,
    )
    endogenous_consumption = right_hand_side ** (-1 / inverse_eis)
    endogenous_cash = endogenous_consumption + asset_grid[:, None]

    endogenous_cash = jnp.vstack([jnp.zeros(income_size), endogenous_cash])
    endogenous_consumption = jnp.vstack([jnp.zeros(income_size), endogenous_consumption])
    endogenous_certainty = jnp.vstack([certainty_equivalent[0, :], certainty_equivalent])

    def interpolate_to_cash_grid(income_index):
        new_consumption = jnp.interp(
            cash_grid,
            endogenous_cash[:, income_index],
            endogenous_consumption[:, income_index],
        )
        new_certainty = jnp.interp(
            cash_grid,
            endogenous_cash[:, income_index],
            endogenous_certainty[:, income_index],
        )
        return new_consumption, new_certainty

    new_consumption, new_certainty = jax.vmap(interpolate_to_cash_grid)(
        jnp.arange(income_size)
    )
    new_consumption = jnp.maximum(new_consumption.T, EPSILON)
    new_certainty = jnp.maximum(new_certainty.T, EPSILON)

    new_consumption = jnp.minimum(new_consumption, cash_grid[:, None] - EPSILON)
    new_consumption = jnp.maximum(new_consumption, EPSILON)

    new_transformed_value = (
        (1 - discount) * (new_consumption ** (1 - inverse_eis))
        + discount * new_certainty
    )
    new_value = jnp.maximum(new_transformed_value, EPSILON) ** (1 / (1 - inverse_eis))

    return new_value, new_consumption


@jax.jit
def solve_household_loop(value_init, consumption_init, household, prices, gamma, psi, tolerance, max_iter):
    def condition(state):
        value, consumption, iteration, value_error, consumption_error = state
        return (jnp.maximum(value_error, consumption_error) >= tolerance) & (iteration < max_iter)

    def body(state):
        value, consumption, iteration, value_error, consumption_error = state
        new_value, new_consumption = ezegm_step(value, consumption, household, prices, gamma, psi)
        new_value_error = jnp.max(jnp.abs(new_value - value))
        new_consumption_error = jnp.max(jnp.abs(new_consumption - consumption))
        return new_value, new_consumption, iteration + 1, new_value_error, new_consumption_error

    initial_state = (
        value_init,
        consumption_init,
        jnp.array(0, dtype=jnp.int32),
        jnp.array(tolerance + 1.0),
        jnp.array(tolerance + 1.0),
    )
    return jax.lax.while_loop(condition, body, initial_state)


def solve_household(
    household,
    prices,
    gamma,
    psi,
    tolerance=SOLVER_TOLERANCE,
    max_iter=1000,
    initial_state=None,
    return_info=False,
):
    inverse_eis = 1.0 / psi
    if abs(inverse_eis - 1.0) < 1e-8:
        raise ValueError("psi = 1 is not supported by this power-transform implementation.")

    discount, asset_grid, cash_grid, income_grid, transition = household
    cash_size, income_size = len(cash_grid), len(income_grid)

    if initial_state is None:
        value_init = jnp.ones((cash_size, income_size))
        consumption_init = (0.1 * cash_grid[:, None] + 0.01) * jnp.ones(
            (1, income_size)
        )
    else:
        consumption_init, value_init = initial_state
        consumption_init = jnp.asarray(consumption_init)
        value_init = jnp.asarray(value_init)

    consumption_init = jnp.maximum(consumption_init, EPSILON)
    consumption_init = jnp.minimum(consumption_init, cash_grid[:, None] - EPSILON)
    consumption_init = jnp.maximum(consumption_init, EPSILON)

    value, consumption, iteration, value_error, consumption_error = solve_household_loop(
        value_init,
        consumption_init,
        household,
        prices,
        gamma,
        psi,
        tolerance,
        max_iter,
    )
    sync_jax((value, consumption, iteration, value_error, consumption_error))

    info = {
        "iterations": int(iteration),
        "value_error": float(value_error),
        "consumption_error": float(consumption_error),
    }
    if return_info:
        return consumption, value, info
    return consumption, value


@jax.jit
def consumption_to_lottery_policy(consumption, household, prices):
    discount, asset_grid, cash_grid, income_grid, transition = household
    interest_rate, wage = prices
    asset_size, income_size = len(asset_grid), len(income_grid)
    gross_rate = 1 + interest_rate

    current_cash = gross_rate * asset_grid[:, None] + wage * income_grid[None, :]

    def interpolate_income(income_index):
        return jnp.interp(current_cash[:, income_index], cash_grid, consumption[:, income_index])

    current_consumption = jax.vmap(interpolate_income)(jnp.arange(income_size)).T
    current_consumption = jnp.maximum(current_consumption, EPSILON)
    current_consumption = jnp.minimum(current_consumption, current_cash - EPSILON)

    next_asset = current_cash - current_consumption
    next_asset = jnp.clip(next_asset, asset_grid[0], asset_grid[-1])

    upper_index = jnp.searchsorted(asset_grid, next_asset)
    upper_index = jnp.clip(upper_index, 1, asset_size - 1)
    lower_index = upper_index - 1

    lower_asset = asset_grid[lower_index]
    upper_asset = asset_grid[upper_index]
    upper_weight = (next_asset - lower_asset) / jnp.maximum(upper_asset - lower_asset, EPSILON)
    upper_weight = jnp.clip(upper_weight, 0.0, 1.0)

    return lower_index.astype(jnp.int64), upper_index.astype(jnp.int64), upper_weight, next_asset


@jax.jit
def lottery_distribution_step(distribution, lower_index, upper_index, upper_weight, transition):
    asset_size, income_size = distribution.shape
    new_distribution = jnp.zeros_like(distribution)

    def current_income_body(income_index, carry):
        target_distribution = carry
        mass_by_asset = distribution[:, income_index]
        lower_target = lower_index[:, income_index]
        upper_target = upper_index[:, income_index]
        high_weight = upper_weight[:, income_index]

        def next_income_body(next_income_index, inner_carry):
            transition_prob = transition[income_index, next_income_index]
            moved_mass = mass_by_asset * transition_prob
            inner_carry = inner_carry.at[lower_target, next_income_index].add(
                moved_mass * (1 - high_weight)
            )
            inner_carry = inner_carry.at[upper_target, next_income_index].add(
                moved_mass * high_weight
            )
            return inner_carry

        return jax.lax.fori_loop(0, income_size, next_income_body, target_distribution)

    new_distribution = jax.lax.fori_loop(0, income_size, current_income_body, new_distribution)
    return new_distribution / jnp.sum(new_distribution)


@jax.jit
def solve_stationary_distribution_loop(
    lower_index,
    upper_index,
    upper_weight,
    household,
    tolerance,
    max_iter,
):
    discount, asset_grid, cash_grid, income_grid, transition = household
    asset_size, income_size = len(asset_grid), len(income_grid)
    initial_distribution = jnp.ones((asset_size, income_size)) / (asset_size * income_size)

    def condition(state):
        distribution, iteration, error = state
        return (error >= tolerance) & (iteration < max_iter)

    def body(state):
        distribution, iteration, error = state
        new_distribution = lottery_distribution_step(
            distribution,
            lower_index,
            upper_index,
            upper_weight,
            transition,
        )
        new_error = jnp.max(jnp.abs(new_distribution - distribution))
        return new_distribution, iteration + 1, new_error

    initial_state = (
        initial_distribution,
        jnp.array(0, dtype=jnp.int32),
        jnp.array(tolerance + 1.0),
    )
    return jax.lax.while_loop(condition, body, initial_state)


def solve_stationary_distribution(
    lower_index,
    upper_index,
    upper_weight,
    household,
    tolerance=DISTRIBUTION_TOLERANCE,
    max_iter=20_000,
    return_info=False,
):
    distribution, iteration, error = solve_stationary_distribution_loop(
        lower_index,
        upper_index,
        upper_weight,
        household,
        tolerance,
        max_iter,
    )
    sync_jax((distribution, iteration, error))

    asset_distribution = jnp.sum(distribution, axis=1)
    asset_distribution = asset_distribution / jnp.sum(asset_distribution)

    info = {
        "iterations": int(iteration),
        "error": float(error),
    }
    if return_info:
        return distribution, asset_distribution, info
    return distribution, asset_distribution


def capital_supply_from_distribution(asset_distribution, household):
    return float(jnp.sum(asset_distribution * household.asset_grid))


def solve_capital_supply(capital, firm, household, gamma, psi, verbose=False):
    interest_rate = interest_rate_from_capital(capital, firm)
    wage = wage_from_interest_rate(interest_rate, firm)
    prices = create_prices(interest_rate, wage)

    start_time = time.perf_counter()

    household_start = time.perf_counter()
    consumption, value, household_info = solve_household(
        household,
        prices,
        gamma,
        psi,
        return_info=True,
    )
    household_time = time.perf_counter() - household_start

    policy_start = time.perf_counter()
    lower_index, upper_index, upper_weight, next_asset = consumption_to_lottery_policy(
        consumption,
        household,
        prices,
    )
    sync_jax((lower_index, upper_index, upper_weight, next_asset))
    policy_time = time.perf_counter() - policy_start

    distribution_start = time.perf_counter()
    distribution, asset_distribution, distribution_info = solve_stationary_distribution(
        lower_index,
        upper_index,
        upper_weight,
        household,
        return_info=True,
    )
    supply = capital_supply_from_distribution(asset_distribution, household)
    distribution_time = time.perf_counter() - distribution_start
    total_time = time.perf_counter() - start_time

    if verbose:
        print(
            f"K={capital:.6f}, r={interest_rate:.6f}, w={wage:.6f}, "
            f"supply={supply:.6f}, household_iters={household_info['iterations']}, "
            f"dist_iters={distribution_info['iterations']}, "
            f"times=[household {household_time:.3f}s, policy {policy_time:.3f}s, "
            f"distribution {distribution_time:.3f}s, total {total_time:.3f}s]"
        )

    info = {
        "interest_rate": interest_rate,
        "wage": wage,
        "household": household_info,
        "distribution": distribution_info,
        "times": {
            "household": household_time,
            "policy": policy_time,
            "distribution": distribution_time,
            "total": total_time,
        },
        "consumption": consumption,
        "value": value,
        "distribution_array": distribution,
        "asset_distribution": asset_distribution,
    }
    return supply, info


def compute_equilibrium(
    firm,
    household,
    gamma,
    psi,
    lower_capital=1.0,
    upper_capital=20.0,
    tolerance=1e-4,
    max_bracket_attempts=10,
    verbose=True,
):
    def objective(capital):
        supply, info = solve_capital_supply(capital, firm, household, gamma, psi, verbose=verbose)
        return capital - supply, info

    left, right = lower_capital, upper_capital
    left_value, left_info = objective(left)
    right_value, right_info = objective(right)

    for _ in range(max_bracket_attempts):
        if left_value * right_value < 0:
            break
        left *= 0.5
        right *= 1.5
        left_value, left_info = objective(left)
        right_value, right_info = objective(right)
    else:
        raise ValueError("Could not bracket equilibrium capital.")

    last_info = None
    eval_count = 2
    while right - left > tolerance:
        midpoint = 0.5 * (left + right)
        midpoint_value, last_info = objective(midpoint)
        eval_count += 1
        if left_value * midpoint_value <= 0:
            right = midpoint
            right_value = midpoint_value
        else:
            left = midpoint
            left_value = midpoint_value

    equilibrium_capital = 0.5 * (left + right)
    interest_rate = interest_rate_from_capital(equilibrium_capital, firm)
    wage = wage_from_interest_rate(interest_rate, firm)
    if verbose:
        print(
            f"Equilibrium: K={equilibrium_capital:.6f}, r={interest_rate:.6f}, "
            f"w={wage:.6f}, G evaluations={eval_count}"
        )
    return equilibrium_capital, interest_rate, wage, last_info


# =============================================================================
# Comparison plots, default off
# =============================================================================

def consumption_and_saving_on_asset_grid(consumption, household, prices):
    asset_grid = household.asset_grid
    cash_grid = household.cash_grid
    income_grid = household.income_grid
    interest_rate, wage = prices
    gross_rate = 1 + interest_rate
    income_size = len(income_grid)

    current_cash = gross_rate * asset_grid[:, None] + wage * income_grid[None, :]

    values = []
    for income_index in range(income_size):
        values.append(
            jnp.interp(current_cash[:, income_index], cash_grid, consumption[:, income_index])
        )
    asset_consumption = jnp.stack(values, axis=1)
    next_asset = current_cash - asset_consumption
    saving = next_asset - asset_grid[:, None]
    return asset_consumption, saving


def plot_consumption_saving_comparisons(
    firm,
    household,
    fixed_gamma=2.0,
    psi_values=(0.2, 0.5, 0.8),
    fixed_psi=0.5,
    gamma_values=(2.0, 6.0, 10.0),
    max_asset_plot=40.0,
    enabled=RUN_COMPARISON_PLOTS,
):
    if not enabled:
        return

    def solve_for_params(gamma, psi):
        capital, interest_rate, wage, _ = compute_equilibrium(
            firm,
            household,
            gamma,
            psi,
            verbose=False,
        )
        prices = create_prices(interest_rate, wage)
        consumption, value = solve_household(household, prices, gamma, psi)
        asset_consumption, saving = consumption_and_saving_on_asset_grid(
            consumption,
            household,
            prices,
        )
        return capital, prices, asset_consumption, saving

    asset_grid = household.asset_grid
    mask = asset_grid <= max_asset_plot

    for title, values, fixed_name, fixed_value, varied_name in [
        ("Varying psi", psi_values, "gamma", fixed_gamma, "psi"),
        ("Varying gamma", gamma_values, "psi", fixed_psi, "gamma"),
    ]:
        fig_consumption, ax_consumption = plt.subplots(figsize=(7, 4))
        fig_saving, ax_saving = plt.subplots(figsize=(7, 4))

        for value in values:
            gamma = fixed_gamma if varied_name == "psi" else value
            psi = value if varied_name == "psi" else fixed_psi
            capital, prices, asset_consumption, saving = solve_for_params(gamma, psi)
            label_base = f"{varied_name}={value}, K={capital:.2f}"

            for income_index, income in enumerate(household.income_grid):
                linestyle = "-" if income_index == 0 else "--"
                ax_consumption.plot(
                    asset_grid[mask],
                    asset_consumption[mask, income_index],
                    linestyle=linestyle,
                    label=f"{label_base}, z={float(income):.1f}",
                )
                ax_saving.plot(
                    asset_grid[mask],
                    saving[mask, income_index],
                    linestyle=linestyle,
                    label=f"{label_base}, z={float(income):.1f}",
                )

        ax_consumption.set_title(f"{title}: consumption ({fixed_name}={fixed_value})")
        ax_consumption.set_xlabel("asset")
        ax_consumption.set_ylabel("consumption")
        ax_consumption.legend(fontsize=7)
        ax_consumption.grid(alpha=0.3)

        ax_saving.set_title(f"{title}: saving ({fixed_name}={fixed_value})")
        ax_saving.set_xlabel("asset")
        ax_saving.set_ylabel("saving")
        ax_saving.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax_saving.legend(fontsize=7)
        ax_saving.grid(alpha=0.3)

    plt.show()


# =============================================================================
# Main
# =============================================================================

def main():
    firm = create_firm()
    household = create_household()

    test_prices = create_prices(interest_rate=0.03, wage=1.0)
    print("V2 benchmark: EGM + lottery policy + distribution iteration")
    print(f"Grid sizes: asset_size={len(household.asset_grid)}, cash_size={len(household.cash_grid)}")

    print("\nWarm-up...")
    sync_jax(solve_household(household, test_prices, gamma=2.0, psi=4.0, max_iter=5))

    print("\nTimed fixed-price solve...")
    start_time = time.perf_counter()
    consumption, value, household_info = solve_household(
        household,
        test_prices,
        gamma=2.0,
        psi=4.0,
        return_info=True,
    )
    household_time = time.perf_counter() - start_time

    start_time = time.perf_counter()
    lower_index, upper_index, upper_weight, next_asset = consumption_to_lottery_policy(
        consumption,
        household,
        test_prices,
    )
    sync_jax((lower_index, upper_index, upper_weight, next_asset))
    policy_time = time.perf_counter() - start_time

    start_time = time.perf_counter()
    distribution, asset_distribution, distribution_info = solve_stationary_distribution(
        lower_index,
        upper_index,
        upper_weight,
        household,
        return_info=True,
    )
    supply = capital_supply_from_distribution(asset_distribution, household)
    distribution_time = time.perf_counter() - start_time

    print(f"Household solve: {household_time * 1000:.1f} ms, iterations={household_info['iterations']}")
    print(f"Lottery policy:  {policy_time * 1000:.1f} ms")
    print(
        f"Distribution:    {distribution_time * 1000:.1f} ms, "
        f"iterations={distribution_info['iterations']}, error={distribution_info['error']:.3e}"
    )
    print(f"Capital supply at test prices: {supply:.6f}")
    print(f"Total fixed-price time: {(household_time + policy_time + distribution_time) * 1000:.1f} ms")

    print("\nComputing equilibrium...")
    start_time = time.perf_counter()
    capital, interest_rate, wage, _ = compute_equilibrium(
        firm,
        household,
        gamma=2.0,
        psi=4.0,
        verbose=True,
    )
    elapsed = time.perf_counter() - start_time
    print(
        f"Equilibrium result: K={capital:.6f}, r={interest_rate:.6f}, "
        f"w={wage:.6f}, elapsed={elapsed:.2f}s"
    )

    plot_consumption_saving_comparisons(firm, household, enabled=RUN_COMPARISON_PLOTS)


if __name__ == "__main__":
    main()
