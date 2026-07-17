"""Machine-readable objective and constraint descriptions."""

from __future__ import annotations


def objective_terms() -> list[str]:
    """Return objective terms used by the current evaluator."""

    return [
        # 最大化经营毛利：把电力、算力、制氢和海洋业务收入相加，再减去各项可变成本以及海运未满足需求的惩罚。
        "Maximize operating margin / 最大化经营毛利: R_power + R_compute + R_h2 + V_marine"
        " - C_power_var - C_compute_var - C_h2_var - Penalty_marine_unserved.",
        # 电力收入：并网后实际送出的电量，乘以时间步长和电价。
        "R_power,t = P_grid_del,t * Delta_t * 1000 * price_power. / 电力收入",
        # 算力收入：算力负荷对应的服务量乘以算力单价。
        "R_compute,t = X_compute,t * price_compute. / 算力收入",
        # 制氢收入：交付的氢气量乘以氢气单价。
        "R_h2,t = H_delivered,t * price_h2. / 制氢收入",
        # 目前代码只报告运营价值，年化 CAPEX 另在报告层单独计算。
        "Current code reports operating value only; annualized CAPEX is kept as"
        " a separate report-layer calculation. / 当前代码只报告运营价值，年化 CAPEX"
        " 另在报告层单独计算。",
    ]


def constraint_terms() -> list[str]:
    """Return the full Day1-Day7 constraint checklist."""

    return [
        # 电力外送容量：并网功率不能超过电缆容量和接入侧可接受容量中的较小值。
        "Power export capacity / 电力外送容量: 0 <= P_grid,t <= min(P_grid_max, P_grid_accept_max).",
        # 电力外送交付：考虑电缆损耗后，实际送出的功率等于并网功率乘以剩余比例。
        "Power export delivery / 电力外送交付: P_grid_del,t = P_grid,t * (1 - loss_cable).",
        # 算力在线边界：算力要么关闭，要么落在最小到最大功率区间内。
        "Compute online bounds / 算力在线边界: P_compute,t = 0 or P_compute_min <= P_compute,t <= P_compute_max.",
        # 算力 PUE：IT 侧功率乘以 PUE 得到总算力用电功率。
        "Compute PUE / 算力 PUE: P_compute,t = P_it,t * PUE.",
        # 光纤服务容量：服务量不能超过单位时间内可提供的最大服务能力。
        "Fiber service capacity / 光纤服务容量: 0 <= X_compute,t <= B_fiber_service_max * Delta_t.",
        # 电解槽功率边界：制氢电解槽输入功率不能超过额定上限。
        "Electrolyzer bound / 电解槽功率边界: 0 <= P_h2_el,t <= P_h2_el_max.",
        # 氢气产量：电解槽功率、时间步长和制氢比能耗共同决定产氢量。
        "Hydrogen production / 氢气产量: H_prod,t = 1000 * P_h2_el,t * Delta_t / SEC_H2.",
        # 管道输出：通过管道外送的氢气量有速率上限。
        "Pipeline output / 管道输出: 0 <= H_pipe,t <= H_pipe_max * Delta_t.",
        # 船运输出：通过船运外送的氢气量有速率上限。
        "Shipping output / 船运输出: 0 <= H_ship,t <= H_ship_max * Delta_t.",
        # 氢气库存守恒：期末库存等于期初库存加生产量，减去管道和船运外送量。
        "Hydrogen inventory / 氢气库存守恒: H_storage,t+1 = H_storage,t + H_prod,t - H_pipe,t - H_ship,t.",
        # 氢气储量容量：库存不能为负，也不能超过储罐容量。
        "Hydrogen storage capacity / 氢气储量容量: 0 <= H_storage,t <= H_storage_max.",
        # 海洋负载服务：实际服务量不能超过请求量。
        "Marine load service / 海洋负载服务: 0 <= P_marine_served,t <= P_marine_request,t.",
        # 综合功率平衡：可用功率与放电功率之和，等于外送、算力、制氢、海洋负载、充电和弃电之和。
        "Integrated balance / 综合功率平衡: P_available,t + P_storage_dis,t = P_grid,t"
        " + P_compute,t + P_h2_el,t + P_marine,t + P_storage_ch,t + P_curt,t.",
        # 弃电非负：弃电功率不能小于零。
        "Curtailment non-negativity / 弃电非负: P_curt,t >= 0.",
        # 所有物理流量变量都必须非负。
        "All physical flow variables are non-negative. / 所有物理流量变量都必须为非负。",
    ]


def markdown_objectives_and_constraints() -> str:
    """Render objective and constraint descriptions as Markdown."""

    objective_lines = "\n".join(f"- {item}" for item in objective_terms())
    constraint_lines = "\n".join(f"- {item}" for item in constraint_terms())
    return (
        "# 目标函数与约束条件 / Objectives and Constraints\n\n"
        "## 目标函数 / Objective Terms\n\n"
        f"{objective_lines}\n\n"
        "## 约束条件 / Constraint Terms\n\n"
        f"{constraint_lines}\n"
    )
