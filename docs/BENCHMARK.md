# Benchmark & reproducing paper numbers

The `benchmark/` folder contains everything needed to reproduce the
pass-rate and continuous-quality numbers reported in the paper.

## Layout

```
benchmark/
├── verifier.py                    # 3-layer verifier (L1 artifact, L2 executable, L3 oracle)
├── tasks/
│   └── _sources/                  # Task specs (100+ scene-gen, RT, PHY, opt, system tasks)
├── oracles/                       # Reference answers per task family
│   ├── n1/, n2_edit/, n2_freq/, n3_multi_ap/
│   ├── n4_phy/, p1_optimize/, p2_pareto/, s1_s4/
├── trial/                         # Per-trial harness (writes result.json)
├── compute_metrics.py             # Extract continuous metrics from trial output
├── aggregate_metrics.py           # Roll up per-trial into per-condition summaries
├── paper_appendix_table.py        # Generate the paper's appendix table
├── metrics_per_trial.csv          # 2094 trials' quality metrics (checked in)
├── metrics_summary_by_condition.csv
├── metrics_summary_by_tier.csv
└── paper_appendix_table.md        # Failure taxonomy + wall-clock table (paper appendix)
```

## Reproduce paper Tables II / IV / VI

The three pass-rate tables are computed from `benchmark/results/<study>/
<condition>/<task>/tN/result.json` files. Those raw trial outputs are
kept locally but excluded from git (600+ MB). To regenerate:

```bash
# Run one condition on one task, k=1 trial
# (output goes to benchmark/results/<label>/)
PYTHONPATH=. python benchmark/run_benchmark.py \
    --label n1_demo \
    --tasks-file benchmark/tasks/_sources/n1_v2.json \
    --task-ids N1_cov_box_one_screen \
    --conditions with_skill \
    --k 1

# Or run a full sweep over the default task set (takes hours; requires
# Sionna + GPU). --split defaults to train; the held-out test split
# additionally needs --i-understand-held-out.
PYTHONPATH=. python benchmark/run_benchmark.py \
    --label full_sweep \
    --conditions with_skill no_skill self_gen \
    --k 5
```

Each trial writes `result.json` (verifier verdict + per-check details)
and `simulation_result.json` (the agent's numeric output).

## Compute continuous quality metrics

Once you have trial output, the metrics pipeline is:

```bash
# 1. Extract per-trial metrics (path-gain MAE, RSS grid MAE, SINR err,
#    BER log-err, throughput RE) → metrics_per_trial.csv
python benchmark/compute_metrics.py

# 2. Aggregate into per-condition summary
python benchmark/aggregate_metrics.py

# 3. Generate paper-aligned appendix table (with layered verification
#    breakdown + failure taxonomy)
python benchmark/paper_appendix_table.py
less benchmark/paper_appendix_table.md
```

## What's tracked in git

- ✅ `verifier.py`, `compute_metrics.py`, `aggregate_metrics.py`,
  `paper_appendix_table.py` — the logic
- ✅ `tasks/`, `oracles/` — task specs and reference answers
- ✅ `metrics_per_trial.csv`, `metrics_summary_*.csv`,
  `paper_appendix_table.md` — the aggregated results reviewers see
- ❌ `results/<study>/<condition>/<task>/tN/` — raw trial dumps
  (regenerate with `run_benchmark.py`)

The tracked summary CSVs are enough to inspect every trial's numeric
result without rerunning; the raw per-trial `simulation_result.json`
files are only needed if you want to look at generated Sionna code.

## Verifier layers

`verifier.py` composes three sub-checks:

| Layer | Check | Failure mode |
|---|---|---|
| **L1** | required `.py` / `.json` / `.npy` artifacts present, JSON parseable, no `placeholder_pre_shipped_by_harness` | agent never wrote output |
| **L2** | `sionna.rt.*` (or `sionna.phy.*`) actually called; not `fspl_analytical` etc. | analytical shortcut instead of ray tracing |
| **L3** | numeric output falls in audited tolerance windows around the oracle (per-family: path-gain MAE ≤ 3 dB, BER log err ≤ 0.5 dec, SINR err ≤ 0.15, etc.) | wrong numbers |

Only trials passing all three layers count as passes. See
`paper_appendix_table.md` for the exact per-family bucket counts of
each failure type.
