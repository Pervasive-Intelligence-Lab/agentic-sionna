# Sionna Skill Benchmark

Infrastructure for evaluating the `rf-simulator` skill. Not runtime skill
content — the agent running a task only reads `.claude/skills/rf-simulator/`.

## Layout

```
benchmark/
├── README.md                       # This file
├── docs/
│   ├── BENCHMARK_METHODOLOGY.md
│   └── SKILL_UPDATE_PROCESS.md
│
├── run_benchmark.py                # Orchestrator (entry point)
├── trial.py                        # Per-trial worker (called by orchestrator)
├── verifier.py                     # Verification dispatcher (9 check types)
│
├── build_tasks.py                  # Build tasks.json from _sources/*
├── generate_scenes.py              # Build scenes/ fixtures (one-time setup)
├── generate_baseline_skill.py      # Stage-1 of self_gen condition
├── audit_oracles.py                # Methodology check: would oracle pass verifier?
├── audit_leakage.py                # Methodology check: skill leakage
│
├── tasks/
│   ├── tasks.json                  # ← active benchmark (134 tasks)
│   ├── _sources/
│   │   ├── capability_grid.json    # 60 hand-authored capability tasks
│   │   └── tutorial_variants.json  # 99-task corpus (Sionna tutorials + variants)
│   └── _audits/
│       ├── oracle_audit.json
│       └── leakage_audit.json
│
├── scenes/                         # 27 deterministic scene fixtures (tracked)
├── results/                        # Per-run outputs (gitignored)
└── plots/                          # Generated figures (gitignored)
```

## Run it

### Full 134-task benchmark (crash-protected, resumable)
```bash
python benchmark/run_benchmark.py --label run_v1 --workers 4
```

### Paired eval (with_skill / no_skill / self_gen)
```bash
python benchmark/run_benchmark.py --label paired \
    --conditions with_skill no_skill self_gen --workers 6
```

### Targeted rerun (e.g., T1 PHY tasks on opus, 3 trials each)
```bash
python benchmark/run_benchmark.py --label opus_phy \
    --tiers T1 --model opus --k 3 --workers 4
```

### Resume after a crash / SIGTERM / reboot
```bash
python benchmark/run_benchmark.py --label run_v1 --resume --workers 4
```
Already-completed trials are skipped; partial workdirs (killed mid-trial) are
cleaned and re-run.

### Single task via the lower-level worker
```bash
python benchmark/trial.py \
    --task-ids U001 --output-root benchmark/results/smoke \
    --model sonnet --max-turns 15 --timeout 300 --k 1 \
    --condition with_skill
```

### Re-verify existing results with an updated verifier
```bash
python benchmark/verifier.py \
    --task-id U001 \
    --output-dir benchmark/results/run_v1/with_skill/U001/t1
```

### Rebuild tasks.json from sources
```bash
python benchmark/build_tasks.py
# -> benchmark/tasks/tasks.json
```

### Generate the 27 scene fixtures (one-time)
```bash
python benchmark/generate_scenes.py
# -> benchmark/scenes/{easy,medium,hard}/scene_*/scene_state.json
```

### Stage-1 self_gen skill (run once before paired eval)
```bash
python benchmark/generate_baseline_skill.py --model sonnet
# -> benchmark/self_gen_skill/SKILL.md
```

## Result schema

Each trial writes `benchmark/results/<run>/<condition>/<task_id>/t<k>/`:
- `result.json`     — task_id, condition, exec_success, wall_sec, verification
- `stdout.txt`      — agent CLI stdout (stream-json)
- `stderr.txt`      — agent CLI stderr
- `prompt.txt`      — the prompt the agent received
- `simulation.py`   — code the agent wrote (if any)
- `simulation_result.json` — agent's output (if any)
- plus any artifacts: `*.npy`, `*.png`, `scene_state.json`, etc.
