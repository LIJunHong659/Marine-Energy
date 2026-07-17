# Phase 0 validation report

## Scope

The validation target is the data and configuration contract, not the dispatch or capacity optimum.

## Acceptance checks

- Required columns and missing values are checked explicitly.
- Timestamps must be parseable, unique, increasing and exactly hourly.
- Capacity factors and availability lie in [0, 1].
- Physical loads, demand and carbon intensity are non-negative.
- Negative electricity prices remain valid.
- Parameter uncertainty bounds and evidence grades are typed.
- Configuration fields reject unknown keys and physically incomplete battery capacity pairs.
- 24 h, 168 h and 8760 h synthetic datasets are deterministic and valid.
- `1 MWh = 1000 kWh` is covered by an executable test.

## Execution record

- Python: 3.12.13
- pytest: `13 passed in 0.33s`
- ruff: all checks passed
- mypy: no issues in 7 source files
- compileall: passed
- 8760 h contract validation: passed
- deterministic 48-hour manual sample audit: no missing values, bound violations or negative physical demands
- 24 h baseline SHA-256: `a8f783fb7a7b79830f1e0a05c1644618e7bf589f3be98b101211e3e7e4ebcb6e`

Phase 1 has added power balance, transmission availability, energy monotonicity, revenue and regression checks.

## Phase 1 / S0 execution record

- Version: 0.2.0
- pytest: `27 passed in 0.50s`
- ruff: all checks passed
- mypy: no issues in 12 source files
- 24 h and 8760 h baseline: passed
- 60-case capacity-distance-price sensitivity: passed
- 24 h cable-outage and negative-price events: passed
- randomized optimizer audit: 50 cases, each checked against 20,001-point enumeration
- maximum 8760 h offshore-bus residual: `0 MW`
- maximum 8760 h land-side residual: `5.684e-14 MW`

## Phase 2 / S1 execution record

- Version: 0.3.0
- pytest: `39 passed in 1.80s`
- ruff: all checks passed
- mypy: no issues in 15 source files
- compileall: passed
- 8760 h S1 optimization: passed
- S0 zero-battery ablation: exact match
- simultaneous charge/discharge audit: passed
- SOC bounds and annual terminal state: passed
- fixed reserve power and energy headroom: passed
- 5 operating scenarios, 9 power-duration cases and 18 efficiency-degradation cases: passed
- exact transmission recomputation error: below 0.01% of annual land delivery
- deterministic 48-hour S1 output audit: SOC bounds, both power balances, state equation and exclusivity passed
- exported hourly and KPI SHA-256 checks: passed

## Phase 3 / S2 execution record

- Version: 0.4.0
- pytest: `44 passed`
- ruff: all checks passed
- mypy: no issues in 24 source and runner files with third-party imports ignored
- compileall: passed
- 8760 h hydrogen-only and battery-hydrogen optimization: passed
- zero-hydrogen S2 to S1 ablation: exact operating-margin and delivered-energy match
- hydrogen inventory, battery SOC and both power balances: closed within their configured tolerances
- maximum hydrogen state residual: `3.638e-12 kg`; terminal inventory error: `0 kg`
- 7 annual operating scenarios, 9 power-storage cases and 6 hydrogen-price cases: all solved and exported with hashes
- zero-demand case: no hydrogen production, sales or inventory-revenue artefact

## Phase 4 / S3 execution record

- Version: 0.5.0
- pytest: `49 passed`
- ruff: all checks passed
- mypy: no issues in 28 source and runner files with third-party imports ignored
- compileall: passed
- 8760 h green-compute baseline: passed in approximately 2.4 s
- S0 zero-compute ablation: exact operating-margin and delivered-energy match
- maximum flexible-queue residual: `2.842e-14 MWh-IT`; maximum offshore-bus residual: `7.105e-14 MW`
- PUE, rigid SLA, flexible maximum delay, IT capacity, IT ramp, subsea-fibre availability and terminal queue: passed
- 9 annual operating scenarios, 3 IT-capacity cases and 6 compute-price cases: all solved and exported with hashes

## Phase 5 / S4 execution record

- Version: 0.6.0
- pytest: `53 passed`
- ruff: all checks passed
- mypy: no issues in 31 source and runner files with third-party imports ignored
- compileall: passed
- 8760 h joint battery-hydrogen-compute dispatch: passed
- S4 to S2 and S4 to S3 ablations: exact operating-margin and service-output match
- maximum offshore residual: `1.137e-13 MW`; hydrogen residual: `3.638e-12 kg`; flexible-queue residual: `2.842e-14 MWh-IT`
- five shared-boundary operating modes, 8 representative cases and 9 hydrogen-price/compute-price cases: all solved and exported with hashes

## Phase 6 / S5 execution record

- Version: 0.7.0
- pytest: `58 passed`
- ruff: all S5 source, runner and test checks passed
- compileall: passed
- 8760 h loose-export and demand-mismatch factorial dispatch: 16 cases passed
- mainland absorption, physical cable availability and exact cable-loss ledgers: passed
- nationwide spot-compute demand, IT capacity, fibre capacity, PUE and ramp limits: passed
- hydrogen production, sale, fuel-cell use, inventory and terminal state: passed
- maximum annual S5 offshore residual: `1.137e-13 MW`; maximum land residual: `2.842e-14 MW`
- demand-mismatch direct curtailment: `1,257,281.249 MWh`; full-hub curtailment: `0 MWh`
- 72 h cable-outage, 3 seasonal hydrogen-storage cases, 9 price cases, 11 equivalent-cable iterations and 6 cable-marginal cases: solved and exported with hashes
- continuous 90-day wind-lull direct EENS: `11,465.404 MWh`; 1,500 t hydrogen-storage EENS: `0 MWh`

## Phase 7 / S6 execution record

- Version: 0.8.0
- pytest: `63 passed`
- ruff: all source, runner and test checks passed
- compileall: passed
- zero investment limits to S5 direct-export ablation: operating margin and curtailment match within configured tolerance
- endogenous compute-capacity, electrolyzer entry, capacity upper bound and S6-to-S5 configuration replay tests: passed
- 9 cost-scarcity annual investment cases, 6 compute-price cases and 6 hydrogen-price cases: solved to HiGHS optimality
- strategic annual design: 300 MW-IT compute and 51.134 MW electrolyzer; all planning capacities remained inside engineering upper bounds
- strategic maximum offshore residual: `3.212e-11 MW`; land residual: `1.421e-14 MW`; hydrogen residual: `0 kg`
- 8 fixed-capacity out-of-sample S5 replays: all solved; positive incremental-net-value share `100%` in the defined synthetic stress set
- 90-day low-wind reliability planning at 3 interruption values: all solved; direct EENS `11,465.404 MWh`
- 10,000 CNY/MWh reliability case maximum offshore residual: `1.723e-13 MW`; hydrogen state residual: `9.153e-09 kg`
- S6 investment and reliability analysis manifests, hourly outputs and figure hashes: verified
