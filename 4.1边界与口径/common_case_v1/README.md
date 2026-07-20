# common_case_v1

这是 4.3--4.7 联合运行的公共参数包骨架。

- `common_case_v1.yaml` 是 Python 当前可读取的静态参数入口。
- `source_timeseries.csv` 是接口联调用 24 小时源侧出力。
- `availability.csv` 是逐时设备可用率和状态标签。
- `demand_and_price.csv` 是逐时需求和价格口径。
- `initial_state.yaml` 是 SOC、储氢和算力队列等初始状态。

当前数据均为 `interface_smoke` 假设值，只用于接口联调，不形成工程经济结论。
