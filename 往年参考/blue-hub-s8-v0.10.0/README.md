# Blue Hub

面向深远海能源岛的透明、分层、可审计计算模型。本版本完成 Phase 9 / S8：将S7形成的固定容量方案放入统一的资源、登陆能力和客户合同情景中逐时回放，并从期望收益、下行风险、后悔值、登陆能力替代和低风保供五个方面检验其稳健性。

## 当前能力

- 校验逐时时序的列、时间连续性、重复时间戳、缺失值、范围和非负约束；
- 校验技术参数的单位、来源等级及 low/base/high 一致性；
- 校验系统容量和情景定义；
- 生成可复现的 24、168 或 8760 小时合成数据；
- 显式验证 `1 MWh = 1000 kWh` 等核心单位换算。
- 分别验证海上母线和陆侧交付端的逐时功率守恒；
- 采用可手算的解析优化器处理经济外送与强制外送策略；
- 分离换流/终端损耗、海缆二次损耗、网络弃电和经济弃电；
- 批量运行海缆容量、距离和电价敏感性以及24小时故障情景；
- 导出逐时结果、KPI、配置哈希、文件哈希和诊断图。
- 采用HiGHS线性规划联合优化充电、放电、外送、弃电和关键负荷保障；
- 通过8段增量线性化保留海缆二次损耗，并用精确损耗式复核结果；
- 独立审计同时充放电、SOC边界、首尾状态和储备功率/能量裕度；
- 比较电池功率、时长、效率、退化成本和故障初始SOC。
- 以57.5 kWh/kg综合系统SEC将电解槽功率转换为氢气产量，避免重复计入常规BOP耗电；
- 逐时约束氢需求、销售量、储罐容量、库存损耗和年末库存闭合；
- 输出氢气收入、运输费、变动运维、水耗及水成本，并将其与电力收益分账；
- 比较氢价、需求、制氢功率、储氢时长和24/72小时海缆故障下的增量运行价值。
- 以MWh-IT计量算力服务，并以情景化PUE换算设施功率；Token和GPU小时等待实测硬件能耗后再映射；
- 以刚性任务未完成量、柔性任务队列、最大延迟、IT爬坡和年度队列闭合表达服务质量；
- 以海底光缆的服务可用率和MWh-IT/h交付能力约束算力服务输出；光缆不进入电力损耗模型；
- 比较算力价格、需求、PUE、IT容量及海缆/光缆故障下的服务与运行价值。
- 在同一海上功率平衡中联合分配外送、电池、制氢和算力设施耗电，避免同一MWh重复计入不同价值路径；
- 以S2和S3精确退化路径验证联合模型，并在统一资源、价格与合同边界下比较S0—S4五种运行模式；
- 输出氢价—算力价分配图、海缆/光缆故障响应和三条路径的逐时能量分配。
- 将海缆物理容量与大陆逐时接纳能力分开，允许供需错配时段成为可观测、可计价的稀缺约束；
- 新增全国弹性算力池，算力任务可按逐时报价在IT容量、PUE、光缆能力和需求上限内即时接入；
- 新增燃料电池回发，使储氢既能销售，也能在长时间低风期恢复为岛上电力；
- 支持电池、氢能和算力任意开关，输出八组合交互项、Shapley分摊、年化成本上限、海缆容量影子价格和等效海缆容量；
- 支持风电与海上光伏共同进入可再生能源母线，完成大陆接纳受限、72小时海缆故障和90日连续低风案例。
- 将电池功率与能量、电解槽、储氢、燃料电池和算力—光纤容量作为连续规划变量，允许求解结果为零投资；
- 按设备寿命和折现率计算资本回收，分别核算固定运维、PEM栈体更换、初始投资、运行边际和增量净年值；
- 以低、参照和高成本对照三档大陆接纳条件，输出算力报价与氢价的容量进入区间；
- 将规划容量转换成S5固定配置，在8个扰动风况、电价、接纳能力、算力需求和通道故障的样本外年份中回放；
- 在90日低风压力年中，依据关键负荷缺供价值内生选择制氢、储氢和燃料电池容量，区分商品收益与保供价值。
- 将大陆登陆点的绝对接纳上限与海缆物理容量分开；当登陆电网不扩容时，海缆增容不再被误认为可以提高消纳；
- 将输氢管道的小时交付能力和随距离变化的投资年化成本纳入容量选择，避免把制氢视为无限弃电出口；
- 将风电、光伏、波浪能、换流终端、海缆、登陆侧电网和岛上公共设施计入完整项目年化筛选，并与灵活资产成本分账；
- 完成全国利用率校准、局部登陆受限、海缆扩容、电网加固、综合能源岛、算力—氢价前沿、任务可移动性、多资源和故障/低风压力案例；
- 导出中国情境证据等级、结果清单与 48 个固定随机时点的独立复核表。
- 将直接送电、算力主导、制氢主导、联合合同和容量冗余五个固定方案置于六个加权情景中回放，容量不随情景事后改变；
- 计算期望完整项目年值、最差20%概率质量均值、风险调整年值、正年值概率与情景后悔值；
- 在300—700 MW登陆能力下量化综合方案的弃电缓解和等效登陆能力，并保持海缆物理容量不变；
- 以固定候选容量检验7档算力价格和4档到岸氢价，分开报告相对直接送电与完整项目的盈亏门槛；
- 对7、15、30和60日低风期比较直接送电、商品型联合方案与专用储氢—燃料电池保供方案。

## 快速运行

```bash
cd blue-hub
PYTHONPATH=src python scripts/prepare_data.py --hours 24
PYTHONPATH=src:../.deps python -m pytest
PYTHONPATH=src python scripts/run_baseline.py --hours 8760 --output outputs/baseline_8760
MPLCONFIGDIR=/tmp/blue-hub-mpl PYTHONPATH=src python scripts/run_s0_sensitivity.py
PYTHONPATH=src python scripts/run_battery_baseline.py --hours 8760
MPLCONFIGDIR=/tmp/blue-hub-mpl PYTHONPATH=src python scripts/run_battery_scenarios.py
PYTHONPATH=src python scripts/run_hydrogen_baseline.py --hours 8760
MPLCONFIGDIR=/tmp/blue-hub-mpl PYTHONPATH=src python scripts/run_hydrogen_scenarios.py
PYTHONPATH=src python scripts/run_compute_baseline.py --hours 8760
MPLCONFIGDIR=/tmp/blue-hub-mpl PYTHONPATH=src python scripts/run_compute_scenarios.py
PYTHONPATH=src python scripts/run_integrated_baseline.py --hours 8760
MPLCONFIGDIR=/tmp/blue-hub-mpl PYTHONPATH=src python scripts/run_integrated_scenarios.py
MPLCONFIGDIR=/tmp/blue-hub-mpl PYTHONPATH=src python scripts/run_s5_flexibility_analysis.py
MPLCONFIGDIR=/tmp/blue-hub-mpl PYTHONPATH=src python scripts/run_s6_investment_analysis.py
MPLCONFIGDIR=/tmp/blue-hub-mpl PYTHONPATH=src python scripts/run_s6_reliability_analysis.py
MPLCONFIGDIR=/tmp/blue-hub-mpl PYTHONPATH=src python scripts/run_s7_china_value_analysis.py
MPLCONFIGDIR=/tmp/blue-hub-mpl PYTHONPATH=src python scripts/run_s7_china_supporting_scenarios.py
PYTHONPATH=src python scripts/audit_s7_results.py
MPLCONFIGDIR=/tmp/blue-hub-mpl PYTHONPATH=src python scripts/run_s8_robust_value_analysis.py --task all
PYTHONPATH=src python scripts/audit_s8_results.py
```

若 pytest 安装在仓库外部，可省略 `.deps`，或使用当前环境中 pytest 的实际路径。

## 当前边界

S8已经检验固定容量在多种经营条件下的表现，但六个情景的权重仍是透明的规划判断，并非由历史样本估计的客观概率；情景内调度仍具有完美预见，候选容量也尚未构成连续的两阶段随机最优解。波浪能继续使用由风况构造的筛选代理曲线，所有8760小时中国情境输入仍用于识别机制。全国算力池、服务价格、登陆点接纳曲线以及多数海工、氢能和设备成本属于C/D级资料或工程筛选范围，不能直接视为项目收益预测或工程报价。真实项目仍需要同址多年度气象与海况、登陆点调度和电价、接网方案、离岸制氢储运、数据中心设备和客户合同数据。S8的方法、结果和适用范围见 [固定容量多情景风险报告](docs/phase9_s8_robust_value_report.md)。
