# Sionna RF Skill Benchmark — Dataset Card

**Version:** v1.8 (2026-05-01)
**License:** project-internal; tasks reference Sionna v2.0 (Apache 2.0)
**Repository:** Pervasive-Intelligence-Lab/agentic-sionna

## Summary

134 wireless-simulation tasks for evaluating LLM agents augmented with the
`rf-simulator` skill. Tasks span Sionna's three modules (PHY, RT, SYS),
plus end-to-end ML/neural integration and emerging research areas (ISAC,
STAR-RIS, channel charting, etc.).

Each task has a deterministic verifier producing pass/fail. Designed for
paired evaluation: same task run with skill / without skill / self-generated
skill, on the same model, in the same harness.

## Task distribution

### By split (60/40 train/test)

| Split | n | Use |
|---|---|---|
| train | 81 | Iteration / skill tuning |
| test  | 53 | Held-out final evaluation (run **once** per published skill version) |

The test split is held out by `--split` flag in `run_benchmark.py`.
`run_heldout.sh` refuses to re-execute on the same skill version
(directory-existence guard) to prevent test-set leakage.

### By tier (master-guide research-area mapping)

| Tier | Capability area | n | Examples |
|---|---|---|---|
| T0 | Scene generation | 20 | Office layouts, floor plans, furniture placement |
| T1 | PHY link-level | 18 | BER/BLER curves, OFDM, MIMO, FEC |
| T2 | Ray tracing | 22 | Coverage maps, CIR/CFR, RT-to-PHY pipelines |
| T3 | ML/neural integration | 15 | Neural demapper/estimator, end-to-end autoencoder |
| T4 | System level | 43 | Multi-cell, scheduling, link adaptation, AP placement |
| T5 | Emerging research | 10 | ISAC, STAR-RIS, OTFS, channel charting, near-field, semantic |
| T6 | Anchor / tutorial | 6 | Compositional tasks combining multiple tiers |

### By difficulty

| Difficulty | n |
|---|---|
| Easy | 26 |
| Medium | 51 |
| Hard | 57 |

### By verifier type

| Verifier | n | What it checks |
|---|---|---|
| `file_exists` | 33 | Required artifact present + non-trivial size |
| `code_contains` | 32 | Generated code references required identifiers |
| `metric_threshold` | 24 | Scalar metric ≤/≥ threshold |
| `composite` | 20 | AND-composition of multiple subchecks |
| `execution_ok` | 8 | Code runs without errors |
| `metric_monotone` | 7 | Array values monotone in expected direction |
| `count` | 6 | Object count matches expected |
| `metric_range` | 3 | Scalar in [min, max] |
| `value_exact` | 1 | Scalar within tolerance of expected |

Every task additionally runs a battery of plausibility checks (BER physics,
coverage range, training evidence, NMSE floor, etc.) regardless of its
primary verifier — see `benchmark/verifier.check_plausibility()`.

## Provenance

- **Authored by:** project team + adapted from Sionna v2.0 tutorials,
  3GPP TR 38.901 specifications, and the master-guide T1-T60 spec
  (master guide Part 10, partial alignment — see "Reconciliation" below)
- **Sionna version:** all reference values target Sionna 2.0.x (PyTorch
  backend, March 2026 release)
- **Last updated:** 2026-05-01 (v1.8 skill release)
- **Verification:** every task has an oracle solution that the verifier was
  built to accept; oracles are not shipped with the task definitions to
  prevent agents copying

## Known biases / limitations

1. **Sionna API training-data coverage** — many tasks reproduce patterns
   in published Sionna tutorials. Frontier-model agents may have
   memorized solutions. Mitigation: tasks use parameter combinations not
   found verbatim in the Sionna repo (verified by oracle-leakage audit
   in `2026-05-01_phase1_integrity_report.md`).

2. **Tier 4 over-representation** — system-level tasks are 32% of the
   suite (43/134). Skill performance on T4 dominates the headline number.
   Consider domain-level reporting (paper Table 1 mandates this).

3. **Easy-mode bias on T4** — system-level threshold checks (`coverage_pct ≥ X`)
   reward parameter-tuning agents over physically-realistic ones. The
   physics-realism failure mode is documented in
   `2026-04-29_why_skill_hurts.md`.

4. **Single-author bias** — tasks not yet cross-reviewed with an external
   wireless-engineering panel. Multi-author validation deferred to v2.1.

5. **No T1-T60 master-guide alignment yet** — our 134 U-tasks were
   authored from the master guide's 60-task taxonomy but use different
   IDs. Reconciliation table at `benchmark/tasks/U_to_T_mapping.json`
   (Phase 5.1 — pending).

6. **Verifier permissiveness** — to recover Gemma4-style descriptive-naming
   variants, `verifier.py` accepts many field-name aliases. This may
   accept superficially-correct outputs that fail deeper inspection;
   plausibility checks are the second line of defense.

## How to use

```bash
# Train-split run (paired with/without skill):
python3 benchmark/run_benchmark.py \
    --label paired_v18 \
    --split train \
    --conditions with_skill no_skill \
    --k 5 \
    --model meta-llama/Llama-3.1-70B-Instruct

# Held-out evaluation (run ONCE per skill version):
bash benchmark/run_heldout.sh v2.0 meta-llama/Llama-3.1-70B-Instruct
```

## File layout

```
benchmark/tasks/
├── tasks.json                    # 134 task definitions (split-tagged)
├── DATASET_CARD.md               # this file
└── U_to_T_mapping.json           # (planned) master-guide T1-T60 cross-ref
```

Each task has fields: `id` (U-prefixed), `tier`, `capability`, `difficulty`,
`split`, `prompt`, `required_artifacts`, `verifier` (type + spec).

## Reproducibility

- Tasks are versioned with the skill (see git tags `v1.0`, `v1.4`, `v1.8`).
- Verifier scripts are **read-only to the agent** (separate dir).
- Each archived run includes the full task list it ran against
  (`benchmark/results/<label>/progress.json`).
- Oracle-leakage audit script: `python3 -c "..."` in
  `2026-05-01_phase1_integrity_report.md`.

## Citation

If you use this benchmark in research, please cite (placeholder):

> RF-Skill Benchmark v1.8. Sionna-based RF simulation tasks for skill-augmented
> LLM evaluation. 2026.
