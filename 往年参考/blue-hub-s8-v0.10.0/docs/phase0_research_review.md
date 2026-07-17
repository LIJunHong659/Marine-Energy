# 蓝海枢纽模型：Phase 0 建模审查与开发报告

## 1. 本轮结论

项目总说明的总体架构是合理的，尤其是逐时能量模型与秒级动态模型分离、物理守恒优先、固定容量调度与外层容量规划分离、确定性优化优先于遗传算法等原则。当前最大的风险并不在求解算法，而在数据边界与价值口径：若这些问题未先锁定，模型很容易得到数值上更优、物理上却不可解释的混合方案。

因此，本轮严格完成 Phase 0，并将优化求解暂缓到 S0 数据契约通过之后。已经形成可运行的配置 schema、合成时序、单位换算、输入校验和测试体系。

## 2. 对总说明的关键调整

### 2.1 建立物理电量与绿色属性两套账本

同一 MWh 可再生电量只能沿外送、制氢、算力、海洋负荷、充电或弃电中的一条物理路径流动；对应的绿证、碳减排和绿色溢价还需要独立的属性账本。后续模型应先完成物理分配，再约束属性分配不超过 eligible renewable energy，防止电力收入、绿氢减排和绿色算力减排重复认领。

### 2.2 输电模型不预设单一距离效率函数

HVAC 与 HVDC 的终端成本、海缆成本、无功补偿、转换损耗和部分负荷损耗结构不同。Phase 0 不给出未经校准的统一 `eta(D)`。S0 可以先使用明确标注为情景假设的常数效率，随后升级为技术、距离、容量与负载率共同决定的分段线性模型。

### 2.3 电池效率必须从往返效率正确拆分

NREL ATB 的参考往返效率为 85%。若后续假定充放电效率对称，应使用

\[
\eta_{ch}=\eta_{dis}=\sqrt{0.85}\approx0.922,
\]

而不能把 0.85 同时赋给充电和放电，否则实际往返效率会被错误降至 72.25%。

### 2.4 制氢只能选择综合法或分项法之一

DOE 2024 PEM 分析使用的平均系统耗电为 57.5 kWh/kg，其中已经包含 stack 与 balance-of-plant。若模型使用这一综合 SEC，不能再叠加水处理、功率电子和常规辅助功耗。若研究压缩压力、淡化工艺或部分负荷效率，则应切换到分项法，并显式关闭综合 SEC。

### 2.5 算力价值以服务完成量计量

数据中心设施功率应由 IT 任务完成量和 PUE 推导。刚性、柔性和可中断任务需要不同的到达、时限、违约惩罚和中断恢复成本。Token 仅能在具体 GPU、模型、精度、batch、利用率和实测能耗确定后做结果映射，不能作为当前优化变量。

### 2.6 周期边界与极端事件边界分开

正常典型年可采用电池 SOC 和储氢首尾相等的周期边界；海缆故障或台风事件窗口不应强制事件结束时立即回到初态，否则会夸大备用容量需求。事件模型应使用给定初态和终端剩余价值或恢复期约束。

## 3. 最新资料对参数口径的约束

- DOE 的 PEM 成本情景分析给出 57.5 kWh/kg 的寿命平均系统耗电，并说明其包含栈体和 BOP；这支持总说明中 58 kWh/kg 的基准，但要求消除辅助能耗重复计算。
- NREL ATB 给出的 85% 是往返效率，不是单程效率；后续必须在 schema 中区分 round-trip、charge 和 discharge 三种字段。
- LBNL 的数据中心研究将 IT 用电乘以由冷却系统、气候、UPS、配电、风机和泵等共同决定的 PUE。海洋环境会改变冷却和可靠性设计，因此 1.08、1.15、1.25 只能作为情景值。
- IEA 2026 的最新跟踪仍强调低排放氢的成本、基础设施与需求不确定性，说明项目不应以单一氢价或电解槽成本得出确定性产业结论。
- 国家能源局 2026 年公开信息确认海上风电向深远海发展以及风电与制氢、制氨、制醇等多元利用方向，但政策方向不能替代设备成本和市场价格的工程校准。

## 4. 当前数据契约

| 字段 | 物理含义 | 单位 |
|---|---|---|
| `wind_cf`, `pv_cf` | 时段平均容量因子 | fraction |
| `electricity_price` | 陆侧结算电价 | CNY/MWh-land |
| `grid_carbon_intensity` | 被替代电网边际或情景碳强度 | tCO2/MWh |
| `critical_load` | 能源岛关键负荷平均功率 | MW |
| `rigid_compute_arrival` | 本时段到达的刚性 IT 服务量 | MWh-IT |
| `flex_compute_arrival` | 本时段到达的柔性 IT 服务量 | MWh-IT |
| `hydrogen_demand` | 本时段可交付氢需求上限 | kg/h |
| `tx_availability`, `wind_availability` | 聚合设备可用比例 | fraction |

其中 `electricity_price` 明确对应陆侧交付电量，因此输电损耗不能在收入端被忽略。合成数据中的所有数值均为契约测试数据，不能用于项目经济结论。

## 5. 已完成验证

- 24 h 数据可重复生成并保存；
- 168 h 与 8760 h 数据可重复生成并通过完整校验；
- 缺列、缺失值、非数值、无穷值、重复时间戳、时间缺口、顺序错误、容量因子越界和负物理负荷均会报出明确问题；
- 负电价被保留为合法市场输入；
- 未知配置字段被拒绝；
- 电池功率与能量容量不成对时被拒绝；
- 参数来源等级和 low/base/high 关系由 schema 强制；
- ruff、mypy、compileall 和 12 项 pytest 测试均通过。

## 6. S0 的下一步实现设计

S0 只包含风电、关键负荷、海缆外送与弃电。建议按以下顺序建立每小时闭合：

\[
P_t^{wind}=C^{wind}CF_t^{wind}A_t^{wind},
\]

\[
P_t^{export,send}=\min\left(A_t^{tx}C^{tx},\,\max(0,P_t^{wind}-P_t^{critical})\right),
\]

\[
P_t^{export,land}=\eta_t^{tx}P_t^{export,send},
\]

\[
P_t^{curt}=P_t^{wind}-P_t^{critical,served}-P_t^{export,send},
\]

并单独报告海缆损耗：

\[
P_t^{tx,loss}=P_t^{export,send}-P_t^{export,land}.
\]

这里的海上功率平衡使用 send-side 电量，电力收入使用 land-side 电量。若风电不足，增加 `unmet_critical_load`，而不是让弃电变成负值。S0 的关键回归场景包括无风、海缆停运、海缆容量拥塞、负电价、关键负荷高于风电和距离/效率单调性。

负电价还引出一个控制边界：若没有最低发电、合同或弃电惩罚，理性调度可能停止外送并弃电。S0 不能简单写成“只要有剩余风电就满额外送”，而应把电价、外送变动成本和合同约束显式化。

## 7. 在进入容量优化前仍需补齐的数据

1. 广东目标海域至少 1–3 年逐时风速或容量因子，以及台风切出与可用率记录；
2. 目标登陆点的现货、年度合约或项目可实现结算电价口径；
3. HVAC/HVDC 候选电压、回路数、换流站与海缆分项造价、损耗和可用率；
4. 电解槽技术路线、出口压力、部分负荷曲线、启停与退化参数；
5. 氢的离岸储存、装卸、运输和岸上交付价格；
6. 算力硬件、任务类型、SLA、网络时延、海上冷却、UPS 与备件策略；
7. 所有成本的币值年份、税口径、汇率与融资假设。

这些数据缺失时仍可做阈值图，但结论必须写成“何时成立”的边界判断，不能写成工程项目的确定收益预测。

## 8. 权威参考

- U.S. DOE, *Clean Hydrogen Production Cost Scenarios with PEM Electrolyzer Technology* (2024): https://www.hydrogen.energy.gov/docs/hydrogenprogramlibraries/pdfs/24005-clean-hydrogen-production-cost-pem-electrolyzer.pdf
- NREL, *Annual Technology Baseline: Utility-Scale Battery Storage*: https://atb.nrel.gov/electricity/2023/technologies/utility-scale_battery_storage
- LBNL, *2024 United States Data Center Energy Usage Report*: https://eta-publications.lbl.gov/sites/default/files/2024-12/lbnl-2024-united-states-data-center-energy-usage-report.pdf
- U.S. DOE/NREL, *Best Practices Guide for Energy-Efficient Data Center Design* (2024): https://www.energy.gov/sites/default/files/2024-07/best-practice-guide-data-center-design_0.pdf
- IEA, *Global Hydrogen Review 2026*: https://www.iea.org/reports/global-hydrogen-review-2026
- 国家能源局，《海上风电转向“深蓝”》(2026): https://www.nea.gov.cn/20260710/2706be51a7f1446a90658328d1c245d4/c.html

