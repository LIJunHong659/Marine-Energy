"""Machine-readable objective and constraint descriptions."""

from __future__ import annotations


def objective_terms() -> list[str]:
    """Return objective terms used by the current evaluator."""

    return [
        "Maximize operating margin: R_power + R_compute + R_h2 + V_marine"
        " - C_power_var - C_compute_var - C_h2_var - Penalty_marine_unserved.",
        "R_power,t = P_grid_del,t * Delta_t * 1000 * price_power.",
        "R_compute,t = X_compute,t * price_compute.",
        "R_h2,t = H_delivered,t * price_h2.",
        "Current code reports operating value only; annualized CAPEX is kept as"
        " a separate report-layer calculation.",
    ]


def constraint_terms() -> list[str]:
    """Return the full Day1-Day7 constraint checklist."""

    return [
        "Power export capacity: 0 <= P_grid,t <= min(P_grid_max, P_grid_accept_max).",
        "Power export delivery: P_grid_del,t = P_grid,t * (1 - loss_cable).",
        "Compute online bounds: P_compute,t = 0 or P_compute_min <= P_compute,t <= P_compute_max.",
        "Compute PUE: P_compute,t = P_it,t * PUE.",
        "Fiber service capacity: 0 <= X_compute,t <= B_fiber_service_max * Delta_t.",
        "Electrolyzer bound: 0 <= P_h2_el,t <= P_h2_el_max.",
        "Hydrogen production: H_prod,t = 1000 * P_h2_el,t * Delta_t / SEC_H2.",
        "Pipeline output: 0 <= H_pipe,t <= H_pipe_max * Delta_t.",
        "Shipping output: 0 <= H_ship,t <= H_ship_max * Delta_t.",
        "Hydrogen inventory: H_storage,t+1 = H_storage,t + H_prod,t - H_pipe,t - H_ship,t.",
        "Hydrogen storage capacity: 0 <= H_storage,t <= H_storage_max.",
        "Marine load service: 0 <= P_marine_served,t <= P_marine_request,t.",
        "Integrated balance: P_available,t + P_storage_dis,t = P_grid,t"
        " + P_compute,t + P_h2_el,t + P_marine,t + P_storage_ch,t + P_curt,t.",
        "Curtailment non-negativity: P_curt,t >= 0.",
        "All physical flow variables are non-negative.",
    ]


def markdown_objectives_and_constraints() -> str:
    """Render objective and constraint descriptions as Markdown."""

    objective_lines = "\n".join(f"- {item}" for item in objective_terms())
    constraint_lines = "\n".join(f"- {item}" for item in constraint_terms())
    return (
        "# 目标函数与约束条件\n\n"
        "## 目标函数\n\n"
        f"{objective_lines}\n\n"
        "## 约束条件\n\n"
        f"{constraint_lines}\n"
    )

