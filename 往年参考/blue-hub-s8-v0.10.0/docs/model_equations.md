# Phase 1 / S0 model equations

## Renewable power and critical-load priority

\[
P_t^{wind}=C^{wind}CF_t^{wind}A_t^{wind},
\]

\[
P_t^{critical,served}=\min(P_t^{wind},P_t^{critical}),
\qquad
P_t^{unmet}=P_t^{critical}-P_t^{critical,served},
\]

\[
S_t=P_t^{wind}-P_t^{critical,served}.
\]

S0不允许从陆网反向购电，因此关键负荷不足量显式计入EENS。

## Transmission feasibility and loss

\[
0\le P_t^{send}\le
\min(S_t,A_t^{tx}C^{tx}).
\]

终端设备采用线性损耗，海缆采用额定功率归一化的二次损耗代理：

\[
P_t^{term,loss}=\lambda_{term}P_t^{send},
\]

\[
P_t^{cable,loss}=
\lambda_{100}\frac{D}{100}
\frac{(P_t^{send})^2}{C^{tx}},
\]

\[
P_t^{land}=(1-\lambda_{term})P_t^{send}
-\lambda_{100}\frac{D}{100}
\frac{(P_t^{send})^2}{C^{tx}}.
\]

该模型保留了I²R损耗随负载增大的主要非线性，同时仍可获得解析调度解。参数目前属于D级工程情景，不能代替具体电压、导体、回路和换流站设计。

## Analytical hourly optimization

经济外送策略求解：

\[
\max_{P_t^{send}}
\left[
\pi_tP_t^{land}
-c_{tx}^{var}P_t^{send}
-c^{curt}(S_t-P_t^{send})
\right].
\]

代入损耗后，每个时段均为一元二次函数。正电价时比较零送出、可行上限和区间内驻点；负电价时目标函数可能转为凸函数，最优解位于零送出或可行上限。代码同时保留 `must_take` 策略，用于模拟PPA、强制消纳或弃电惩罚情景。

这一解析解将作为后续LP/MILP调度器的独立回归基准，避免复杂模型通过放宽约束获得虚假收益。

## Separate balances

海上母线：

\[
P_t^{wind}=P_t^{critical,served}+P_t^{send}+P_t^{curt}.
\]

输电链路：

\[
P_t^{send}=P_t^{land}+P_t^{term,loss}+P_t^{cable,loss}.
\]

弃电发生在送端，海缆损耗发生在输电链路，两者不得合并。

# Phase 2 / S1 battery extension

## Battery state

\[
E_{t+1}=(1-\sigma)E_t
+\eta_{ch}P_t^{ch}\Delta t
-\frac{P_t^{dis}\Delta t}{\eta_{dis}},
\]

\[
SOC_{min}C_E\le E_t\le SOC_{max}C_E,
\qquad E_T=E_0.
\]

85%的往返效率在对称假设下拆分为

\[
\eta_{ch}=\eta_{dis}=\sqrt{0.85}\approx0.922.
\]

吞吐退化成本为

\[
C_t^{deg}=c^{deg}(P_t^{ch}+P_t^{dis})\Delta t.
\]

正退化成本在当前场景中消除了LP的同时充放电退化解，求解后仍执行显式排他审计。

## Reserve proxy

给定储备功率 \(R\) 和持续时间 \(\tau\)：

\[
E_t\ge E_{min}+\frac{R\tau}{\eta_{dis}},
\]

\[
P_t^{dis}-P_t^{ch}\le C_P-R.
\]

该约束只证明功率和能量裕度，不代表完成频率稳定验证。

## Joint power balance

\[
P_t^{wind}+P_t^{dis}+P_t^{unmet}
=P_t^{critical}+P_t^{send}+P_t^{ch}+P_t^{curt}.
\]

输电的凹二次交付曲线采用等宽增量分段线性化。优化完成后使用原二次式重新计算陆侧电量、收入和线性化误差，防止近似误差被写入最终KPI。

# Phase 3 / S2 hydrogen extension

电解槽功率按综合系统SEC转换为氢气产量：

\[
m_t^{H2}=\frac{1000P_t^{H2}\Delta t}{SEC^{H2}}.
\]

氢库存满足：

\[
H_{t+1}=(1-\lambda^H)H_t+m_t^{H2}-S_t,
\qquad H_T=H_0.
\]

销售量受每小时需求上限和储罐容量共同约束。电解槽的综合SEC只在海上功率平衡中扣除一次；水耗作为物料成本另行计量，未重复叠加常规BOP用电。

# Phase 4 / S3 green-compute extension

算力服务以MWh-IT计量。已完成的IT工作量为：

\[
X_t^{IT}=X_t^{rigid}+X_t^{flex},
\qquad
P_t^{DC}=PUE_t\frac{X_t^{IT}}{\Delta t}.
\]

刚性任务满足：

\[
X_t^{rigid}+U_t^{rigid}=A_t^{rigid} \Delta t.
\]

柔性队列满足：

\[
Q_{t+1}=Q_t+A_t^{flex}\Delta t-X_t^{flex},
\qquad Q_T=Q_0=0.
\]

最大等待时间通过队列上界实现。若最大延迟为L小时，则在时段t完成调度后，队列中只能保留最近L小时内到达的柔性工作量。该形式与累计服务约束等价，同时保持8760小时LP稀疏。

\[
X_t^{rigid}+X_t^{flex}
\le C^{IT}\Delta t,
\]

\[
X_t^{rigid}+X_t^{flex}
\le A_t^{fiber}C^{fiber}\Delta t.
\]

第二式是海底光缆的服务交付代理，单位为MWh-IT/h，不表示光通信带宽或电力损耗。S3海上功率平衡为：

\[
P_t^{wind}+P_t^{unmet}
=P_t^{critical}+P_t^{send}+P_t^{DC}+P_t^{curt}.
\]

# Phase 5 / S4 integrated allocation

S4将S1、S2和S3放到一个功率平衡中：

\[
P_t^{wind}+P_t^{dis}+P_t^{unmet}
=P_t^{critical}+P_t^{send}+P_t^{ch}+P_t^{H2}+P_t^{DC}+P_t^{curt}.
\]

电池、氢库存和柔性算力队列仍分别遵守其S1、S2和S3状态方程。联合目标函数为电力交付、氢销售和算力服务收入之和，扣除输电变动成本、电池吞吐成本、电解槽和水成本、算力变动运维、刚性任务SLA惩罚及失供惩罚。

在单一功率平衡下，所有电量都有唯一去向：直接外送、进入电池、驱动电解槽、支撑数据中心设施负荷、保障关键负荷或弃电。S4通过将算力服务量保留在MWh-IT、氢销售量保留在kg、电力交付量保留在MWh，避免跨价值路径的物理量混用和收入重复。

# Phase 6 / S5 scarcity-aware flexible hub

S5允许风电与海上光伏共同进入海上母线：

\[
P_t^{RE}=C^{wind}CF_t^{wind}A_t^{wind}+C^{PV}CF_t^{PV}.
\]

海缆的物理可用容量与大陆接纳能力分别记录。若大陆接纳系数为
\(g_t\in[0,1]\)，送端功率满足：

\[
0\le P_t^{send}\le C^{tx}A_t^{tx}g_t.
\]

全国弹性算力是可选服务量，不形成必须清偿的任务队列。其完成量满足：

\[
0\le X_t^{spot}\le D_t^{national}\Delta t,
\]

\[
X_t^{rigid}+X_t^{flex}+X_t^{spot}
\le \min(C^{IT},A_t^{fiber}C^{fiber})\Delta t.
\]

算力设施功率改写为：

\[
P_t^{DC}=PUE_t\frac{X_t^{rigid}+X_t^{flex}+X_t^{spot}}{\Delta t}.
\]

燃料电池的耗氢量依据氢低位热值与电效率计算：

\[
m_t^{FC}=\frac{1000P_t^{FC}\Delta t}{LHV^{H2}\eta^{FC}}.
\]

氢库存同时受到制取、销售、储存损耗和回发耗氢影响：

\[
H_{t+1}=(1-\lambda^H)H_t+m_t^{H2}-S_t-m_t^{FC},
\qquad H_T=H_0.
\]

最终海上功率平衡为：

\[
P_t^{RE}+P_t^{dis}+P_t^{FC}+P_t^{unmet}
=P_t^{critical}+P_t^{send}+P_t^{ch}+P_t^{H2}+P_t^{DC}+P_t^{curt}.
\]

目标函数以运行边际最大为准，收入包括陆侧交付电力、实际销售氢气和实际完成算力服务；成本包括输电变动成本、电池吞吐退化、电解制氢与水耗、氢运输、燃料电池变动运维、算力变动运维、任务违约、弃电和关键负荷失供。资产投资尚未进入目标函数，因此S5输出的是年化固定成本上限和容量影子价值，不是净现值。

# Phase 7 / S6 endogenous investment planning

S6固定风电、光伏与海缆公共配置，将六项灵活容量放入与逐时调度相同的线性规划：

\[
K=\{C_B^P,C_B^E,C_{EL},C_{H2},C_{FC},C_{IT}\}.
\]

其中算力容量同时代表本轮成本口径下配套的光纤服务能力。逐时运行变量受容量决策约束，例如：

\[
0\le P_t^{ch},P_t^{dis}\le C_B^P,
\qquad
s_{min}C_B^E\le E_t\le s_{max}C_B^E,
\]

\[
0\le P_t^{H2}\le C_{EL},
\quad
0\le H_t\le C_{H2},
\quad
0\le P_t^{FC}\le C_{FC},
\]

\[
0\le X_t^{spot}\le C_{IT}A_t^{fiber}\Delta t.
\]

折现率为 \(r\)、寿命为 \(n\) 时，资本回收因子为：

\[
CRF(r,n)=\frac{r(1+r)^n}{(1+r)^n-1}.
\]

某项容量的年化固定成本由初始投资与固定运维组成：

\[
C_i^{annual}=CAPEX_i\left[CRF(r,n_i)+f_i^{FOM}\right].
\]

PEM栈体更换以满负荷电量代理计入变动成本。若更换比例为 \(q\)，间隔为 \(L\) 满负荷小时，初始投资为 \(c_{EL}\) 元/kW，则：

\[
c_{stack}^{rep}=\frac{1000q c_{EL}}{L}\quad \text{元/MWh}.
\]

S6目标函数最大化年度运行边际与灵活资产年化固定成本之差：

\[
\max\;V^{op}(K,x)-\sum_i C_i^{annual}(K_i).
\]

所有容量变量均有非负下界，因此当运行价值不足以覆盖资本回收和固定运维时，最优解可以为零。对不足一年的测试时段，运行结果按8760小时等比例年化，正式案例均使用连续8760小时。

S6规划层只使用全国可选算力池，不处理S5中的刚性和可延期合同队列。输出容量可以转换为固定 `SystemConfiguration`，再由S5完整模型回放，以复核算力爬坡、光纤可用率、功率平衡和库存闭合。90日低风案例仍采用相同目标函数，其中关键负荷缺供成本用于表示保供价值；它属于压力测试价值，不是已经存在的市场收入。
