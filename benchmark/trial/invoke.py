"""Claude CLI subprocess invocation."""
from __future__ import annotations
import json
import os
import sys
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

from benchmark.trial.skeletons import pre_ship_skeleton

if TYPE_CHECKING:
    from benchmark.trial.run import TrialConfig

# benchmark/trial/invoke.py is 2 levels below repo root
ROOT = Path(__file__).resolve().parents[2]
CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")


def _agent_gave_up_empty(stdout: str, workdir: Path) -> bool:
    """Detect the 'Qwen3.6-style premature end_turn' pattern: agent
    end_turn'd with no result text AND no real simulation_result.json
    (only the harness-pre-shipped placeholder on disk).

    A real run overwrites simulation_result.json with status != the
    pre-ship sentinel and at least one non-null numerical field.
    """
    sim = workdir / "simulation_result.json"
    if not sim.exists():
        return True  # not even pre-ship survived → definitely broken
    try:
        j = json.loads(sim.read_text())
    except Exception:
        return False  # malformed but present — agent at least tried
    if j.get("status") != "placeholder_pre_shipped_by_harness":
        return False  # agent overwrote with real output
    # Last result line in stdout: empty result string?
    has_text_result = False
    for line in (stdout or "").splitlines()[::-1]:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("type") == "result":
            text = (ev.get("result") or "").strip()
            has_text_result = bool(text)
            break
    return not has_text_result


def parse_stream_json_usage(stdout: str) -> dict:
    """Aggregate token usage from Claude CLI --output-format stream-json NDJSON.
    Returns totals + per-turn list. Robust to partial / malformed lines.
    """
    totals = {
        "input_tokens": 0, "output_tokens": 0,
        "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
        "num_turns": 0,
    }
    per_turn = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        # Events with usage live under message.usage (per CLI schema)
        msg = ev.get("message") or ev
        u = msg.get("usage") if isinstance(msg, dict) else None
        if not isinstance(u, dict):
            continue
        turn = {k: int(u.get(k, 0) or 0)
                for k in ("input_tokens", "output_tokens",
                          "cache_read_input_tokens",
                          "cache_creation_input_tokens")}
        per_turn.append(turn)
        for k, v in turn.items():
            totals[k] += v
    totals["num_turns"] = len(per_turn)
    totals["total_input_including_cache"] = (
        totals["input_tokens"]
        + totals["cache_read_input_tokens"]
        + totals["cache_creation_input_tokens"]
    )
    return {"totals": totals, "per_turn": per_turn}


def _build_invoke_env(task: dict, condition: str) -> dict:
    """Build the environment dict passed to the Claude CLI subprocess.

    Extracted from invoke_claude() so tests can call it directly without
    launching a real subprocess.
    """
    env = dict(os.environ)

    # scripts/ and tools/ are at the SKILL dir but the agent runs
    # in a deep trial workdir. Export RF_SKILL_DIR so SKILL.md references
    # like `$RF_SKILL_DIR/scripts/run_ber_analytical.py` resolve correctly.
    env["RF_SKILL_DIR"] = str(ROOT / ".claude" / "skills" / "rf-simulator")
    # tell the skill not to prompt the user (would deadlock the harness).
    env["RF_NO_PROMPT"] = "1"

    # Sionna conda env: `python3` in the default user shell does NOT have
    # sionna installed on this host. Export the absolute path of the sionna
    # interpreter so the agent (instructed by SKILL.md / task prompt) can
    # invoke it as `$RF_SIONNA_PY simulation.py` instead of the default
    # `python3` (which would fall back to FSPL on ImportError).
    # Default to the interpreter running the benchmark (activate the
    # `sionna` conda env first), overridable via RF_SIONNA_PY.
    _sionna_py = os.environ.get("RF_SIONNA_PY") or sys.executable
    if Path(_sionna_py).exists():
        env["RF_SIONNA_PY"] = _sionna_py

    # expose the 3D-FUTURE catalog if present on disk. The
    # scene-gen library checks $FURNITURE_CATALOG_PATH; absent on local
    # dev machines, present on sunlab at ~/3D-FUTURE-model/.
    # Order: explicit FURNITURE_CATALOG_PATH wins; otherwise probe known paths.
    _explicit = os.environ.get("FURNITURE_CATALOG_PATH")
    if _explicit and Path(_explicit).is_dir() and (Path(_explicit) / "model_info.json").exists():
        env["FURNITURE_CATALOG_PATH"] = _explicit
    else:
        for _candidate in (
            Path.home() / "3D-FUTURE-model",
            Path("/data/3D-FUTURE-model"),
            Path.home() / "PycharmProjects/new-sionna-skill/3D-FUTURE-model",
        ):
            if _candidate.is_dir() and (_candidate / "model_info.json").exists():
                env["FURNITURE_CATALOG_PATH"] = str(_candidate)
                break

    return env


def invoke_claude(prompt: str, workdir: Path,
                  config: "TrialConfig",
                  task: dict | None = None) -> tuple[bool, str, str, float, dict]:
    """Run Claude CLI; return (exec_success, stdout, stderr, wall_seconds, usage).

    Uses --output-format stream-json so we can parse per-turn token usage.
    """
    cmd = [CLAUDE_BIN, "-p", prompt,
           "--model", config.model,
           "--max-turns", str(config.max_turns),
           "--permission-mode", "bypassPermissions",
           "--output-format", "stream-json",
           "--verbose"]

    # Optional tool subset — local LLMs via vLLM have small context windows
    # (~24K). Default Claude Code tool defs add ~10K input tokens. Setting
    # CLAUDE_CODE_TOOLS_OVERRIDE='Bash,Edit,Read,Write,Glob,Grep' keeps the
    # prompt under budget for those models. Sonnet on Anthropic API ignores.
    tools_override = os.environ.get("CLAUDE_CODE_TOOLS_OVERRIDE")
    if tools_override:
        cmd.extend(["--tools", tools_override])

    run_cwd = str(workdir)
    env = _build_invoke_env(task or {}, config.condition)

    # no_skill / self_gen run in an isolated tmpdir so the project's
    # auto-discovered SKILL.md doesn't leak into the agent's context.
    # with_skill always uses the project root so Claude Code finds the
    # full SKILL.md.
    use_isolated = config.condition in ("no_skill", "self_gen")

    if use_isolated:
        import tempfile, shutil
        prefix = f"claude_{config.condition}_"
        isolated = Path(tempfile.mkdtemp(prefix=prefix))
        if config.condition == "self_gen":
            # Populate the isolated dir with the pre-generated self_gen skill
            # so Claude Code's discovery finds it exactly as it would
            # discover the curated skill in with_skill.
            #
            # Per-model self_gen skill (preferred) or legacy single-file
            # (fallback). The bootstrap generates these via the model's own
            # backend so each model tests its own self-gen capability.
            model_basename = (os.environ.get("MODEL_HF")
                              or os.environ.get("VAST_MODEL_HF")
                              or "").split("/")[-1].lower()
            candidates = []
            if model_basename:
                candidates.append(
                    ROOT / f"benchmark/self_gen_skill/{model_basename}/SKILL.md")
            candidates.append(ROOT / "benchmark/self_gen_skill/SKILL.md")  # legacy fallback
            src = next((c for c in candidates if c.exists()), None)
            if src is None:
                raise FileNotFoundError(
                    f"self_gen skill not found at any of: {candidates}. "
                    "Run `python benchmark/generate_baseline_skill.py "
                    "--backend openai-compat --out-file benchmark/self_gen_skill/<model>/SKILL.md` "
                    "before launching self_gen trials.")
            dest_dir = isolated / ".claude/skills/sionna-self-generated"
            dest_dir.mkdir(parents=True)
            shutil.copy2(src, dest_dir / "SKILL.md")
        run_cwd = str(isolated)
        # After the run we copy the agent's outputs back into workdir.

    # pre-ship skeleton to whatever directory the agent
    # actually runs in (workdir for with_skill; isolated tmpdir for
    # no_skill/self_gen). The agent overwrites it with json.dump if it
    # produces real output; otherwise the verifier finds the placeholder.
    if task is not None:
        pre_ship_skeleton(task, Path(run_cwd))

    def _copy_back():
        """Copy isolated tmpdir contents back to workdir + clean up.
        MUST run on every exit path (success, error, timeout) so the
        pre-shipped skeleton + any partial agent output survive."""
        if not use_isolated:
            return
        import shutil
        isolated_path = Path(run_cwd)
        if not isolated_path.exists():
            return
        for item in isolated_path.iterdir():
            # Skip the .claude tree we populated for skill discovery.
            if item.name == ".claude":
                continue
            dest = workdir / item.name
            if dest.exists():
                continue
            try:
                if item.is_file():
                    shutil.copy2(item, dest)
                elif item.is_dir():
                    shutil.copytree(item, dest)
            except Exception:
                pass  # best-effort; don't crash the trial on a copy error
        shutil.rmtree(isolated_path, ignore_errors=True)

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=run_cwd, env=env, timeout=config.timeout,
            capture_output=True, text=True,
        )
        wall = time.time() - t0
        ok = proc.returncode == 0
        _copy_back()
        usage = parse_stream_json_usage(proc.stdout)
        return ok, proc.stdout, proc.stderr, wall, usage
    except subprocess.TimeoutExpired as e:
        wall = time.time() - t0
        _copy_back()
        def _decode(x):
            if x is None:
                return ""
            return x.decode(errors="replace") if isinstance(x, bytes) else x
        out = _decode(e.stdout)
        usage = parse_stream_json_usage(out) if out else {"totals": {}}
        return (False, out,
                _decode(e.stderr) + "\n[TIMEOUT]", wall, usage)
