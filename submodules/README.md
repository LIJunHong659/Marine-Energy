# Blue Hub 子模型代码包

这个目录是根据 `Day1-Day7任务说明.md` 生成的最小可运行模型包，覆盖：

- 电力-海缆外送模型
- 算力-光缆输出模型
- 氢气-管道/船运输出模型
- 海洋综合用能模型
- 综合功率平衡
- 目标函数与约束条件清单
- 单元测试和 24h 示例

当前版本是“可审计子模型”，不是完整优化求解器。它用于把公式、参数、约束和收益口径先跑通，后续可以接 Pyomo、CVXPY 或其他优化器。

联合运行优先读取根目录的公共案例参数包：

```text
4.1边界与口径/common_case_v1/common_case_v1.yaml
```

`submodules/configs/parameters.yaml` 仍作为兼容入口保留；公共案例参数包缺失时，示例程序才会回退到该文件。

## 目录结构

```text
submodules/
  pyproject.toml
  README.md
  目标函数与约束条件.md
  examples/
    run_example.py
  src/
    bluehub_submodules/
      power_export.py
      compute_load.py
      hydrogen_output.py
      marine_load.py
      balance.py
      objectives.py
      constraints.py
      parameters.py
      scenario.py
  tests/
    test_power_export.py
    test_compute_load.py
    test_hydrogen_output.py
    test_marine_load.py
    test_integrated_balance.py
    test_constraints.py
```

## 使用 uv 运行

在项目根目录执行：

```powershell
cd D:\project\submodules
$env:UV_CACHE_DIR='D:\project\.uv-cache'
uv run python -m unittest discover -s tests
```

运行 24h 示例：

```powershell
cd D:\project\submodules
$env:UV_CACHE_DIR='D:\project\.uv-cache'
uv run python examples\run_example.py
```

如果你后续安装了 pytest，也可以执行：

```powershell
cd D:\project\submodules
$env:UV_CACHE_DIR='D:\project\.uv-cache'
uv run pytest
```

## Python 调用示例

```python
from bluehub_submodules import default_parameters, simple_greedy_dispatch, summarize_results

params = default_parameters()
results = simple_greedy_dispatch(params)
summary = summarize_results(results)
print(summary)
```

单独调用电力外送模型：

```python
from bluehub_submodules.power_export import PowerExportParams, evaluate_power_export

params = PowerExportParams(
    cable_capacity_mw=700,
    grid_accept_max_mw=450,
    cable_loss_fraction=0.08,
    price_power_cny_per_kwh=0.35,
)
result = evaluate_power_export(500, params)
print(result.delivered_power_mw, result.revenue_cny, result.violations)
```

单独调用算力模型：

```python
from bluehub_submodules.compute_load import ComputeLoadParams, evaluate_compute_load

params = ComputeLoadParams(
    compute_power_max_mw=120,
    compute_power_min_mw=10,
    pue=1.15,
    fiber_service_capacity_mw_it=100,
    price_compute_cny_per_mwh_it=1500,
)
result = evaluate_compute_load(80, params)
print(result.it_power_mw, result.service_mwh_it, result.revenue_cny)
```

单独调用氢气模型：

```python
from bluehub_submodules.hydrogen_output import HydrogenParams, evaluate_hydrogen_output

params = HydrogenParams(
    electrolyzer_power_max_mw=150,
    sec_kwh_per_kg=57.5,
    pipe_capacity_kg_per_h=1800,
    ship_capacity_kg_per_h=1200,
    storage_max_kg=30000,
    price_h2_cny_per_kg=30,
)
result = evaluate_hydrogen_output(
    requested_electrolyzer_power_mw=100,
    requested_pipe_output_kg=1000,
    requested_ship_output_kg=500,
    storage_start_kg=0,
    params=params,
)
print(result.produced_kg, result.delivered_kg, result.storage_end_kg)
```

## 当前模型边界

- 不做详细电网潮流，只做海缆容量、陆上接纳能力和损耗约束。
- 不做真实光通信网络路由，光缆只作为算力服务交付能力约束。
- 不做船舶逐航次调度，船运用等效 kg/h 能力表达。
- 不做完整投资优化，当前目标函数输出运行毛利。
- 所有默认参数都是筛选假设，写入报告前必须替换为有来源的参数表。

## 后续接优化器的方式

当前函数都可以作为优化模型的公式原型：

- `evaluate_power_export` 对应电力外送链路。
- `evaluate_compute_load` 对应算力负荷和光缆交付。
- `evaluate_hydrogen_output` 对应制氢、储氢、管道和船运。
- `evaluate_marine_load` 对应海洋综合用能。
- `evaluate_integrated_hour` 对应综合功率平衡。

如果接 Pyomo/CVXPY，建议保持同样变量名和单位，不要改变收益和成本口径。
