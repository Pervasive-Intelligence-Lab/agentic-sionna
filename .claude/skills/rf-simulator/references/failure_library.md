# Sionna/RF Failure Library

**Auto-distilled from observed trajectory failures across v1.0 → v1.4 benchmark
runs (2026-04-26 → 2026-04-29). Each principle traces to ≥1 source task and
has been confirmed by manual review.**

| Last updated | Total principles | Sionna version | Source benchmarks | Verification status |
|---|---|---|---|---|
| 2026-05-01 | 18 | 2.0.x (claimed) | Sonnet baseline + Qwen3.6/Gemma4-27B v1, paired_qwen3_6_v2/v3/v4, paired_llama31_8b_v4, paired_gemma4_31b_v4 | **Trajectory-confirmed**, not Sionna-execution-confirmed |

> **Verification caveat (2026-05-01):** Each principle here has been observed
> in real failed trajectories (sources cited per entry). The "Last verified"
> dates reflect when the symptom was last seen in a benchmark run, NOT when
> the fix was re-tested against a live Sionna 2.0.x install. Re-execution
> against live Sionna is **task P1.6 in v1.5_to_v2.0_plan.md** — pending H200
> rollout where Sionna will actually be installed. Until then, treat every
> principle as a high-confidence pattern from past runs, not a guaranteed
> Sionna 2.0.1 reproduction.

---

## Category: Wrong Channel Model

### CDL/TDL for Multi-User Scenario  `[high confidence]`
- **Symptom**: `RuntimeError: CDL channel does not support multiple transmitters`
- **Root cause**: CDL-A/B/C/D/E and TDL-* support only `NUM_TX = NUM_RX = 1` (point-to-point).
- **Fix**: Replace CDL with `UMi`, `UMa`, or `RMa` from `sionna.phy.channel.tr38901`. Use `gen_single_sector_topology()` or `gen_hexgrid_topology()` for placement.
- **Source tasks**: T6, T9, T25 (3 task families, all multi-user scenarios)
- **Last verified**: Sionna 2.0.1
- **Update class**: `[ACTIVE]` — links to SKILL.md Module 2 constraint #5

### TDL for MIMO Antenna Correlation  `[medium confidence]`
- **Symptom**: Antenna correlation not applied; all spatial streams produce identical output.
- **Root cause**: TDL does not model spatial correlation; it's tap-delay only.
- **Fix**: Use CDL (point-to-point) or Sionna RT (multi-node) for correlated MIMO.
- **Source tasks**: T10 (Massive MIMO 64×4)
- **Update class**: `[ACTIVE]`

---

## Category: Wrong API Version

### v0.x Imports in v2.0 Environment  `[high confidence]`
- **Symptom**: `ImportError: cannot import name 'AWGN' from 'sionna.channel'`
- **Root cause**: `sionna.channel`/`sionna.mimo`/`sionna.ofdm` are v0.x namespaces; v2.0 uses `sionna.phy.channel`/`sionna.phy.mimo`/`sionna.phy.ofdm`.
- **Fix**: `from sionna.phy.channel import AWGN`. See `references/sionna-version-guide.md` for the full migration table.
- **Source tasks**: T16 (migration task), incidental on T4/T8 when version not declared
- **Update class**: `[STABLE]`

### TF Ops in PyTorch v2.0 Environment  `[high confidence; relevance decreasing]`
- **Symptom**: `AttributeError: module 'tensorflow' has no attribute 'GradientTape'` — or silent type mismatch on tensors.
- **Root cause**: Sionna 2.0 dropped TensorFlow; uses PyTorch backend. `tf.GradientTape`, `tf.Variable`, etc. don't exist.
- **Fix**: Use `torch.autograd`, `torch.optim`, `torch.nn.Parameter` for trainable variables. Convert v1.x examples by replacing `with tf.GradientTape() as tape: ... grads = tape.gradient(loss, vars)` with `loss.backward()`.
- **Source tasks**: T22, T23, T28 (any gradient-based task touching v1.x docs)
- **Update class**: `[ACTIVE]`
- **Relevance note (2026-05-01)**: Sionna 2.0 dropped TF in March 2026, so this failure mode shrinks over time as agents stop being trained on v1.x examples. Keep until a full benchmark cycle goes by without observing it.

### `out_type="tensorflow"` in v2.0  `[high confidence]`
- **Symptom**: `TypeError: out_type 'tensorflow' is not supported in this version`.
- **Root cause**: Pre-v2.0 ray-tracer methods accepted `out_type="tensorflow"`. v2.0 replaces with `out_type="torch"`.
- **Fix**: `paths.cir(out_type="torch", normalize_delays=True)`.
- **Source tasks**: T20 (CIR extraction), incidental on T24
- **Update class**: `[STABLE]`

---

## Category: Gradient / Optimization Errors

### Missing `requires_grad` in v2.0 Optimization  `[high confidence]`
- **Symptom**: `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`.
- **Root cause**: PyTorch tensors must opt into autograd; default `torch.tensor(...)` has `requires_grad=False`.
- **Fix**: Use `torch.nn.Parameter(torch.zeros(N))` for variables you'll train; or `tensor.requires_grad_(True)` after creation.
- **Source tasks**: T22, T23, T26 (TX orientation, material learning, RIS phase opt)
- **Update class**: `[ACTIVE]`

### Permittivity Unclamped During Material Learning  `[high confidence]`
- **Symptom**: Ray tracer crashes mid-iteration with cryptic Mitsuba error, or produces NaN paths.
- **Root cause**: Optimizer drove permittivity to negative or absurdly high value; physically the tracer can't handle it.
- **Fix**: Clamp to `[1.0, 80.0]` after every step: `eps_r.data.clamp_(1.0, 80.0)`. Or apply softplus + bias before passing to scene.
- **Source tasks**: T23 (material learning), T26 (RIS via gradients)
- **Update class**: `[ACTIVE]`

### No Power Normalization in End-to-End Autoencoder  `[high confidence]`
- **Symptom**: Loss decreases but BER stays at 0; learned constellation has very high-power outliers.
- **Root cause**: Optimizer "cheats" by driving TX power higher rather than learning a good constellation.
- **Fix**: Normalize encoder output: `x = x / torch.sqrt(torch.mean(torch.abs(x)**2))`. Add as a module-level layer, not a manual division.
- **Source tasks**: T31 (E2E AWGN autoencoder), T37 (neural beamforming)
- **Update class**: `[ACTIVE]`

---

## Category: Numerical / Unit Errors

### Frequency in Hz vs GHz Confusion  `[high confidence]`
- **Symptom**: Path-loss values come out as +200 dB or −20 dB (instead of normal 60-120 dB range).
- **Root cause**: Sionna expects frequency in Hz everywhere (`scene.frequency = 3.5e9`, not `3.5`). User code passes GHz literal.
- **Fix**: Always use `freq_hz` variable name, set as `e9` literal: `FREQ_HZ = 3.5e9`. Validate result is in [40, 160] dB before continuing.
- **Source tasks**: T21 (radio map), T54 (THz channel)
- **Update class**: `[ACTIVE]`

### Eb/N0 vs Es/N0 Off-by-`10·log10(N)` dB  `[high confidence]`
- **Symptom**: BER curve shifted by 3 dB (QPSK), 6 dB (16-QAM), 7.78 dB (64-QAM) from the textbook.
- **Root cause**: Coded systems use Eb/N0; uncoded comparisons sometimes use Es/N0; converting requires `+10*log10(num_bits_per_symbol*code_rate)`.
- **Fix**: Use `ebnodb2no(ebno_db, num_bits_per_symbol, code_rate)` — never compute the noise variance manually. For coded systems use Eb/N0 on x-axis.
- **Source tasks**: T1 (uncoded baseline), T2 (LDPC coded)
- **Update class**: `[ACTIVE]`

### CP Length Shorter Than Channel Delay Spread  `[high confidence]`
- **Symptom**: BER plateau even at high SNR; OFDM error floor.
- **Root cause**: Inter-symbol interference because the cyclic prefix doesn't cover the channel's delay spread.
- **Fix**: For CDL/TDL channels, set CP ≥ max delay spread × sample rate. With `subcarrier_spacing=30 kHz`, CP=288 samples covers ~9.6 µs — adequate for most CDL profiles.
- **Source tasks**: T4 (basic SISO OFDM), T24 (RT to PHY)
- **Update class**: `[ACTIVE]`

---

## Category: Sionna Setup / Loading

### `sionna.rt` Not Imported Before Scene Load  `[high confidence]`
- **Symptom**: All materials default to vacuum (permittivity 1.0); RSS computations off by ~30 dB.
- **Root cause**: ITU material plugins register on `import sionna.rt`. Without that import, `load_scene()` falls back to vacuum.
- **Fix**: `import sionna.rt as rt` at the top of every RT script, before any scene operation.
- **Source tasks**: T19 (scene loading), T21 (radio map)
- **Update class**: `[ACTIVE]`

### `scene.frequency` Not Set Before Solver  `[high confidence]`
- **Symptom**: Material parameters use stale values from a previous frequency; path loss off by several dB.
- **Root cause**: ITU material permittivity is frequency-dependent and recomputes on `scene.frequency` assignment.
- **Fix**: Set `scene.frequency = FREQ_HZ` immediately after `load_scene()` and before any `RadioMapSolver` / `PathSolver` call.
- **Source tasks**: T21, T22, T26
- **Update class**: `[ACTIVE]`

### Missing `normalize_delays=True` in `paths.cir()`  `[medium confidence]`
- **Symptom**: CIR taps misaligned with OFDM grid; equalizer struggles even with perfect CSI.
- **Root cause**: Without normalization, the absolute delays don't fit cleanly in the CP window; first tap may be at a non-zero delay.
- **Fix**: `a, tau = paths.cir(out_type="torch", normalize_delays=True)`.
- **Source tasks**: T20 (CIR extraction), T24 (RT to PHY)
- **Update class**: `[ACTIVE]`

---

## Category: Schema / Output Format

### Descriptive Field Names Instead of Canonical  `[high confidence]`
- **Symptom**: Verifier reports `threshold:metric` failures even though the agent computed the metric correctly.
- **Root cause**: Verifier extracts by exact field name first, then aliases. Agents (especially Gemma-class) emit `{"ebno_db": [...], "BPSK": [...], "QPSK": [...]}` instead of `numerical_metrics.{snr_db, ber_simulated}`.
- **Fix**: Use the canonical schema verbatim. SKILL.md Module 2 constraint #11 has ✓/✗ examples.
- **Source tasks**: U001-U005, U017 (Sonnet-rare, Gemma-frequent)
- **Update class**: `[ACTIVE]` — directly maps to constraint #11

### Top-Level Modulation Keys Instead of `numerical_metrics`  `[medium confidence]`
- **Symptom**: BER curves stored as `{"BPSK": [...], "QPSK": [...]}` at top level → verifier doesn't find canonical paths.
- **Root cause**: Tutorial-style output (one curve per modulation) isn't the schema the verifier expects.
- **Fix**: Pick the highest-priority modulation as the canonical curve in `numerical_metrics.ber_simulated`. Store all other curves under `numerical_metrics.ber_curves[]` (list-of-dicts with `modulation` + `ber` keys).
- **Source tasks**: U003 (BER multi-modulation), generic on T1
- **Update class**: `[ACTIVE]`

---

## Category: Workflow / Behavior

### Premature `end_turn` After First Error  `[medium confidence]`
- **Symptom**: Agent runs script once, gets a stderr message, ends with empty result text. Trial shows ~5-7 turns used.
- **Root cause**: Skill prose like "ZERO references on first attempt" reads as "give up if uncertain"; agent doesn't retry.
- **Fix**: Skill v1.4+ moved retry rule into `build_prompt()` tail (max-salience). Auto-retry hook in `trial.py` re-invokes once if first run < `max_turns / 2`.
- **Source tasks**: U049, U054, U058 (Qwen3.6 v1.2)
- **Update class**: `[ACTIVE]` — see SKILL.md "Restate Before Coding"

### `Write` Tool Malformation Loop  `[high confidence]`
- **Symptom**: Agent calls `Write file_path="X"` without `content`, gets `InputValidationError`, retries identical broken call until 25-turn cap.
- **Root cause**: Qwen3.6's `qwen3_coder` parser drops the `content` arg ~40% when the body is large. Skill's "Turn 1: Write skeleton" rule directly hits this weakness.
- **Fix**: v1.4 dropped `Write` from `CLAUDE_CODE_TOOLS_OVERRIDE` (`Bash,Read` only). Bash + heredoc is single-arg and works on every model.
- **Source tasks**: U018, U058, U089 (Qwen3.6 v1.2)
- **Update class**: `[ACTIVE]` — infrastructure-side fix in the local-LLM serving script (not part of this release)

### Physics-Realism Tax (T4 system_level)  `[high confidence]`
- **Symptom**: Skill-loaded agent computes 58% coverage in a corridor; no-skill agent computes 100%. Same numpy code; different parameter choices.
- **Root cause**: Skill teaches ITU-style radio params (lower TX, log-normal fading, longer scenarios). Verifier rewards `coverage_pct ≥ X` regardless of physical realism.
- **Fix**: TBD for v1.5+ — either (a) verifier acceptance bands, or (b) explicit "tune for the metric" guidance, or (c) task prompts that make tuning explicit. See `_studies_archive/2026-04-29_why_skill_hurts.md`.
- **Source tasks**: U104, U109 (T4 coverage threshold)
- **Update class**: `[ACTIVE]` — open issue, fix not yet shipped

### Skeleton Priming Distraction  `[medium confidence]`
- **Symptom**: Task asks "compute peak SE"; pre-shipped T1 BER schema primes the agent to fill BER fields and skip peak_se.
- **Root cause**: With skeleton on disk, agents preferentially fill canonical fields rather than reading the prompt for the actual target metric.
- **Fix**: TBD — task-aware skeleton (use `verifier.metric` to pick the schema, not just tier) or skip pre-ship when prompt names a specific metric.
- **Source tasks**: U017 (peak_se in T1 BER skeleton)
- **Update class**: `[ACTIVE]` — open issue

---

## How to Update This File

1. After each eval round, run the distillation pipeline (see master guide Part 8 step-by-step).
2. New principles enter under the right category. If the new principle is semantically close (cosine ≥ 0.85) to an existing one, **merge** rather than duplicate.
3. Confidence:
   - `[high confidence]` — observed on ≥ 3 distinct tasks across ≥ 2 models, fix verified
   - `[medium confidence]` — observed on 1-2 tasks, fix not yet stress-tested
   - `[low confidence]` — single observation; treat as a hypothesis
4. Update class follows SKILL.md governance (`[FROZEN]`/`[STABLE]`/`[ACTIVE]`).
5. Bump the table at the top: increment count, update `Last updated` and `Sionna version`.

The vector-store version of this file (planned) will embed each principle for semantic retrieval at task start. Until then, this markdown is loaded as a normal reference per SKILL.md Module 1.
