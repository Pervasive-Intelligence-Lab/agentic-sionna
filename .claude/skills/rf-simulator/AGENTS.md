# Sionna RF Skill — AGENTS.md

This file is for coding-agent harnesses (Claude Code, Codex, etc.) that
execute against the rf-simulator skill. It provides the operational
glue: env setup, version detection, eval running, plausibility ranges,
the most common pre-submit failure checks, and an importable scene
generation library at `lib/scene_gen/` so a single skill covers BOTH
Sionna coding tasks AND room/floor-plan generation.

For research-direction guidance, see `SKILL.md` in this directory.
For the scene-gen library API, see `references/scene-gen-library.md`.

---

## Environment setup

```bash
pip install sionna           # v2.0+ uses PyTorch backend (default since Mar 2026)
python -c "import sionna; print(sionna.__version__)"   # confirm version
```

GPU recommended for ray-tracing (Sionna RT). CPU fallback works but is
~10× slower; for CPU-only environments load `references/cpu-fallback.md`
and switch to the analytical model.

---

## Version detection

Always confirm or infer Sionna's version BEFORE generating code. Mixing
versions causes immediate `ImportError`.

```python
v2.0+ : from sionna.phy.channel.tr38901 import UMi      # works
v1.x  : from sionna.phy.channel.tr38901 import UMi      # works (TF backend)
v0.x  : import sionna.channel                            # works (legacy)
```

Mixed imports across versions → ImportError. Never mix.
See `references/sionna-version-guide.md` for the full migration table.

---

## Update governance per file (matches SKILL.md tags)

| File | Class | Update rule |
|---|---|---|
| `SKILL.md` (Modules 1-3) | `[ACTIVE]` | Auto-rewritten by eval loop after each cycle |
| `SKILL.md` Task Baselines table | `[STABLE]` | Updated only on Sionna major releases |
| `SKILL.md` Domain Constants | `[FROZEN]` | Domain-expert review only |
| `references/static-knowledge.md` | `[FROZEN]` | Domain-expert review only |
| `references/sionna-version-guide.md` | `[STABLE]` | Updated on Sionna major releases |
| `references/3gpp-models.md` | `[STABLE]` | Updated on 3GPP standard revisions |
| `references/task-baselines.md` | `[STABLE]` | Updated when published baselines move |
| `references/error-patterns.md` | `[ACTIVE]` | Distilled from failure pipeline; auto-updated |
| `references/{neural-receivers,differentiable-optimization,system-level,sionna-diffraction-ris}.md` | `[ACTIVE]` | Eval-loop eligible |
| All other `references/*.md` | `[STABLE]` | Eval-loop touches only with two confirmations |
| `templates/*.py` | `[ACTIVE]` | Replaced wholesale by validated regenerations |
| `templates/result_schema_*.json` | `[STABLE]` | Schema additions only; never break canonical names |

---

## On Sionna release: tag `[REVIEW_NEEDED]`

When Sionna ships a new minor/major version, run the version-review script
to flag stale guidance per master-guide §5 + §7 (SWE-Skills-Bench pattern):

```bash
# Dry-run: report blocks that mention the old version
python3 .claude/skills/rf-simulator/scripts/version_review.py

# Apply tags inline:
python3 .claude/skills/rf-simulator/scripts/version_review.py --apply

# After re-eval pass, strip the tags from blocks that still pass:
python3 .claude/skills/rf-simulator/scripts/version_review.py --clear
```

Blocks tagged `[REVIEW_NEEDED]` are excluded from auto-improvement-loop
edits until cleared. Implements the SWEBench++ rule that version-mismatched
guidance is the dominant failure cause for documentation-style skills.

---

## Vector store: bulk-seed, bulk-index, serve, tunnel

```bash
# Seed failure principles from references/failure_library.md:
python3 .claude/skills/rf-simulator/scripts/seed_memory.py

# Index Sionna v2.0 API into the store (limit 20 for testing):
python3 .claude/skills/rf-simulator/scripts/index_sionna_docs.py --limit 20

# Start chroma HTTP server (background, loopback-only):
bash .claude/skills/rf-simulator/scripts/start_chroma_server.sh --bg
# stop:    --stop
# w/auth:  CHROMA_TOKEN=$(openssl rand -hex 32) bash ... --bg

# Open reverse SSH tunnel sunlab → vast.ai (for GPU rentals to reach the DB):
bash .claude/skills/rf-simulator/scripts/start_reverse_tunnel.sh \
    --vast-host ssh1.vast.ai --vast-port 23456 --vast-user root
# stop:   --stop;  status: --status
```

ChromaDB collection path resolution (in order, per `memory/store.py`):
1. `SIONNA_SKILL_CHROMA_PATH` env var
2. `~/.local/share/sionna-skill/chroma_db` (XDG default — survives
   working-tree resets, branch switches, multi-clone)
3. Legacy `.claude/skills/rf-simulator/memory/chroma_db/` (used only if a
   non-empty dir exists there)

Both seed scripts upsert by stable id with cosine-similarity dedup at
write time (skipped if cosine ≥ 0.85 to an existing chunk). Idempotent.

---

## Running the benchmark

```bash
# From repo root
python benchmark/run_benchmark.py \
    --label paired_run \
    --conditions with_skill no_skill \
    --workers 6 \
    --model meta-llama/Llama-3.1-70B-Instruct
```

Results land in `benchmark/results/<label>/<cond>/<task_id>/t<n>/`:
- `prompt.txt`           — task prompt sent to agent
- `simulation.py`         — agent-generated code
- `simulation_result.json`— canonical output (pre-shipped by harness; agent overwrites)
- `result.json`           — verification result + token usage
- `stdout.txt`/`stderr.txt`— full transcript

Aggregate via `benchmark/_studies_archive/2026-04-28_v4_summary.md` patterns
or peek per-task through `result.json`'s `verification.checks[]`.

---

## Physical plausibility thresholds

Use these to sanity-check before submitting. Out-of-range values almost
always indicate a unit error or wrong API.

| Quantity | Valid range | Common error if violated |
|---|---|---|
| BER (any modulation) | 0 ≤ BER ≤ 1 at all SNR points | wrong noise formula, missing `ebnodb2no()` |
| BER monotonicity | non-increasing with SNR | sign flip on LLR or hard-decision before decoder |
| Path loss | 40-160 dB for typical wireless | frequency in Hz vs GHz unit mismatch |
| Received power | ≤ TX power (energy conservation) | path loss applied with wrong sign |
| NMSE (channel est.) | -25 to -5 dB useful range | below -25 dB suggests overfitting or evaluation on training set |
| NVE | LS baseline ≈ 94; novel target < 70 | "94" exact = neural net not training |
| RIS gain | up to ~10 dB over random in NLoS | negative → RIS hurting; phase init wrong |
| Permittivity (RT materials) | [1, 80] | unphysical values crash ray tracer; clamp during gradients |
| Coverage % | [0, 100] | values outside indicate unit/threshold bug |
| BLER (coded) | should drop > 5 dB faster than uncoded | flat curve = code not connected |

---

## Most common pre-submit failures (CHECK BEFORE ENDING)

1. **CDL used with multi-user** → `RuntimeError: CDL channel does not support multiple transmitters`. Use UMi/UMa/RMa.
2. **Wrong noise formula** → manual conversion instead of `ebnodb2no(ebno_db, num_bits_per_symbol, code_rate)`.
3. **Missing `normalize_delays=True`** in `paths.cir()` → CIR taps misaligned with OFDM grid.
4. **`num_bits_per_symbol` mismatch** between `Mapper` and `Demapper`.
5. **TF ops in PyTorch environment (v2.0)** → `tf.GradientTape` in v2.0 raises AttributeError; use `torch.autograd`.
6. **`out_type="tensorflow"` in v2.0** → TypeError; use `out_type="torch"`.
7. **Permittivity unclamped during gradient learning** → ray tracer crashes (clamp [1, 80]).
8. **Equal-weight FedAvg with equal-size datasets** → collapses to centralized; defeats federated purpose.
9. **Continuous RIS phases without quantization note** → real hardware can't do continuous phase shifts.
10. **Same channel realization across batch items** → autoencoder overfits to one channel; introduce per-sample randomness.
11. **`scene.frequency` not set before solver** → wrong material permittivity → wrong path loss.
12. **`sionna.rt` not imported before `load_scene()`** → ITU material plugins not registered.
13. **Eb/N0 vs Es/N0 confusion** → curve shifts by `10·log10(num_bits_per_symbol)` dB; for coded systems use Eb/N0.
14. **CP length < channel delay spread** → ISI corrupts BER even at high SNR.

---

## Code style conventions

- Variable names match Sionna conventions: `rg=ResourceGrid`, `sm=StreamManagement`, `ll=LDPC5GEncoder`, etc.
- Channel models in `ALL_CAPS` matching 3GPP notation: `UMi`, `UMa`, `RMa`, `CDL_A`, `TDL_C`.
- One runnable script: imports → parameter block → model setup → simulation loop → output → visualization. Top-to-bottom.
- Always include the visualization step: BER curve, radio map, constellation, training-loss plot. Visualizations get implicit credit on subjective evaluations and explicit credit on `artifact:*.png` checks.
- For RTX 5090 / Mitsuba destructor segfault: end scripts with `os._exit(0)`. Not needed on H100/H200/A100.

---

## Project structure

```
.claude/skills/rf-simulator/
├── SKILL.md                          # Technical guidance [ACTIVE]
├── AGENTS.md                         # This file
├── agents/                           # Role-specific procedures
│   ├── qa-validator.md
│   └── rf-researcher.md
├── references/                       # Loaded conditionally per Module 1 routing
│   ├── failure_library.md            #   [ACTIVE] auto-distilled principles
│   ├── static-knowledge.md           #   [FROZEN]
│   ├── sionna-version-guide.md       #   [STABLE]
│   ├── 3gpp-models.md                #   [STABLE]
│   ├── task-baselines.md             #   [STABLE]
│   ├── error-patterns.md             #   [ACTIVE]
│   └── ... 22 more files
├── templates/                        # Simulation templates + canonical schemas
│   ├── template_{ber,rt_coverage,mimo_ofdm,rt_to_phy,scene,neural_train,system_level}.py
│   └── result_schema_{ber,rt_coverage,mimo_ofdm,neural,scene}.json
├── scripts/                          # Agent-callable helper scripts
│   ├── _verifier_core.py             #   Task-agnostic plausibility helpers (imported by verify_output.py + benchmark/verifier.py)
│   ├── run_ber_analytical.py         #   BER analytical fallback (Q-function)
│   ├── verify_output.py              #   Self-verification gate (5 plausibility checks)
│   ├── nve_metric.py                 #   Normalized Validation Error computation
│   ├── seed_memory.py                #   Bulk-seed failure principles → vector store
│   ├── index_sionna_docs.py          #   Bulk-index Sionna v2 API → vector store
│   ├── version_review.py             #   Tag [REVIEW_NEEDED] on Sionna release
│   ├── start_chroma_server.sh        #   Run chroma HTTP server (loopback only)
│   └── start_reverse_tunnel.sh       #   autossh -R from sunlab → vast.ai
├── memory/                           # Vector store
│   ├── store.py                      #   ChromaDB wrappers (lazy-init, dedup)
│   └── README.md                     #   Activation instructions
│   #  chroma_db/ lives at ~/.local/share/sionna-skill/chroma_db (XDG)
├── tools/                            # Online lookups against live sources
│   └── online_apis.py                #   fetch_sionna_docs, search_arxiv, search_github_issues
└── lib/                              # Importable Python utilities
    └── scene_gen/                    #   Scene generation library (room/furniture/exporter pipeline)
        ├── models.py                 #     Pydantic v2 Scene/Room/Furniture/TX/RX (frozen)
        ├── geometry.py               #     AABB overlap, in-bounds, rotated-rect corners
        ├── constraints.py            #     wall_affinity / collision / pathway costs + validate_scene()
        ├── optimizer.py              #     LayoutOptimizer + place_furniture() + place_tx()
        ├── exporters/                #     PNG / XML (Sionna RT) / GLTF / materials / validator
        └── tests/                    #     Round-trip tests (pytest)

benchmark/                            # Evaluation infrastructure (NOT runtime skill)
├── tasks/                            # 134-task definitions, 81 train + 53 test
├── trial.py                          # Per-trial worker (pre-ships skeleton, retry hook)
├── verifier.py                       # Deterministic check dispatcher
├── run_benchmark.py                  # Top-level runner (parallel pool)
├── run_heldout.sh                    # Run-once held-out test evaluation
├── improvement_loop.py               # Auto-improvement orchestrator (scaffold)
├── tool_call_proxy.py                # OpenAI-compat repair proxy
└── _studies_archive/                 # Per-version archives + iteration_log.md
```
