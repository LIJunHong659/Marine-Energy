# Parameter source audit

| Parameter | Initial value/range | Grade | Status |
|---|---:|:---:|---|
| PEM system electricity use | 57.5 kWh/kg; scenario 52–65 | A | DOE 2024 analysis supports 57.5 average system use. Range retained for sensitivity. |
| Battery round-trip efficiency | 85% | A | NREL ATB reference assumption. Offshore environment and grid-forming reserve effects remain uncalibrated. |
| Data-centre PUE | 1.15; scenario 1.08–1.25 | D | Scenario only. LBNL/DOE sources support physics-based PUE accounting, not this offshore value. |
| Offshore transmission efficiency/cost | Not fixed in Phase 0 | — | Must be technology-, voltage-, capacity-, distance- and loading-specific; a single distance multiplier is not accepted. |
| Electricity, hydrogen and compute prices | Not fixed for inference | D | Require China/Greater Bay Area market evidence and common price-year conversion. |

Primary references are recorded as URLs in `configs/technology_parameters.csv`. The next calibration round should add a frozen evidence snapshot with title, publication year, table/page locator, currency year and conversion method.

## Phase 1 transmission placeholders

The executable S0 base case currently uses a 3% combined HVDC terminal-loss fraction and a 1.5%/100 km full-load cable-loss proxy. Both are grade-D engineering scenarios. They provide a transparent nonlinear test model and do not claim to represent a selected Chinese cable or converter system. Transmission CAPEX remains intentionally absent; capacity results are reported as break-even annualized cost ceilings rather than economic optima.

## Phase 2 battery parameters

| Parameter | Base | Grade | Interpretation |
|---|---:|:---:|---|
| Round-trip efficiency | 85% | A | NREL ATB reference value; split symmetrically in S1. |
| SOC range | 10%–90% | D | Operational scenario pending vendor and safety calibration. |
| Self-discharge | 0.002%/h | D | Aggregate standing-loss proxy. |
| Throughput degradation cost | 80 CNY/MWh | D | Numerical and economic scenario; sensitivity spans 40–160. |
| Transmission linearization | 8 segments | D | Numerical setting; final revenue uses exact quadratic loss. |

NREL's analysis of storage beyond four hours indicates that additional duration can have diminishing marginal market value. The project therefore reports both total value and value per added kWh instead of treating longer duration as automatically superior: https://docs.nrel.gov/docs/fy23osti/85878.pdf

## Phase 6 flexibility and hydrogen-return parameters

| Parameter | Base | Grade | Interpretation |
|---|---:|:---:|---|
| Nationwide spot-compute service price | 420 CNY/MWh-IT in parameter table; 360 CNY/MWh-IT in main case | D | Illustrative workload bid, not a market quotation. |
| Mainland absorption factor | 0.32/0.55/1.00 | D | Stress profile derived from synthetic price and wind states. |
| Hydrogen lower heating value | 33.33 kWh/kg | A | Standard thermochemical conversion basis; no compression energy is inferred from it. |
| Fuel-cell electrical efficiency | 55% LHV | A/D | DOE supports high-efficiency fuel-cell conversion; 55% is a project scenario within the published technical range. |
| Fuel-cell variable cost | 35 CNY/MWh-electric | D | Non-fuel operating proxy pending vendor data. |

The uploaded compute-storage study is used for one methodological lesson: flexible-resource value must be evaluated under both loose and binding interconnection constraints. Its U.S. market revenues, DVFS details and numerical penalties are not transferred to this offshore case. The S5 adaptation uses mainland absorption scarcity, nationwide optional workloads and seasonal hydrogen return instead: https://arxiv.org/abs/2605.16190

The 2025 NREL utility-scale battery report separates power-related and energy-related capital costs and documents wide cost uncertainty. S5 therefore reports operating-value ceilings per kWh-year rather than treating the 100 MW/400 MWh battery as economically justified: https://docs.nrel.gov/docs/fy25osti/93281.pdf

Fuel cells are represented as hydrogen-to-electricity converters capable of long-duration grid support. The model uses a conservative 55% LHV electrical-efficiency scenario inside DOE's stated high-efficiency range: https://www.energy.gov/cmei/fuels/fuel-cells

## Phase 7 investment parameters

S6的成本参数见 `configs/s6_investment_cost_cases.csv`。美元参数统一按7.2元/美元的工程换算情景转换，仅用于保持量级和比例一致，不代表2026年7月13日即期汇率。国外陆上设备成本也不代表中国海上交钥匙价格。

| 项目 | 低成本 | 参照成本 | 高成本 | 依据与等级 |
|---|---:|---:|---:|---|
| 电池功率投资（元/kW） | 1,179 | 2,678 | 3,214 | 低值按NREL 2035低成本投影缩放；参照值取2024年功率分项372美元/kW换算；高值加入工程溢价，A/D |
| 电池能量投资（元/kWh） | 764 | 1,735 | 2,082 | NREL 2024年能量分项241美元/kWh及2035低成本投影，A/D |
| PEM电解槽投资（元/kW） | 10,800 | 14,400 | 18,000 | DOE 1,500、2,000、2,500美元/kW区间按工程汇率换算，A/D |
| 储氢投资（元/kg） | 50 | 200 | 600 | 海上储罐、压力与安全设计尚未确定的筛选区间，D |
| 燃料电池投资（元/kW） | 6,000 | 10,000 | 15,000 | 项目筛选区间，待设备商报价，D |
| 算力—光纤综合投资（元/kW-IT） | 20,000 | 60,000 | 120,000 | 服务器、供配电、冷却、海上建造和光纤分摊尚未分项的宽区间，D |

NREL 2025年报告给出的2024年电池功率与能量分项分别为372美元/kW和241美元/kWh，四小时系统合计334美元/kWh；2035年四小时系统低、中、高投影分别为147、243和339美元/kWh。报告选择15年寿命、85%往返效率，并指出含扩容的固定运维可达到四小时系统功率成本的4%。S6保留15年寿命，固定运维率按成本情景取2%至3%，因此仍需以中国海上项目全寿命维护方案校准：https://docs.nrel.gov/docs/fy25osti/93281.pdf

DOE 2024年PEM成本记录给出57 kWh/kg系统耗电、30年系统寿命、1,500至2,500美元/kW安装成本、每40,000运行小时一次和初始投资11%的更换成本。报告以初始投资5%作为不含栈体更换的年度固定运维。S6将栈体更换折成元/MWh制氢电量，避免与固定运维重复：https://www.hydrogen.energy.gov/docs/hydrogenprogramlibraries/pdfs/24005-clean-hydrogen-production-cost-pem-electrolyzer.pdf

算力投资缺少能够覆盖服务器代际、加速器型号、海上冷却、供配电和光缆分摊的权威统一口径。S6因此把2万至12万元/kW-IT列为D级筛选范围，并通过服务报价前沿反推合同门槛。该区间不得替代设备清单和EPC报价。上传文献只用于验证互联约束收紧会提高灵活工作量的运行价值，没有用于确定本项目算力造价或收入：https://arxiv.org/abs/2605.16190
