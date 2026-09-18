"""Chat blueprint — OpenAI-compatible call to any Claude / GPT endpoint.

Users bring their own API key via environment variables. The endpoint may be
Anthropic direct, an OpenAI-compat relay (exchangetoken, MG6, PackyCode,
ZetaAPI, ...), or a local model server.

Configure via env (NO defaults — the app fails loudly if these are missing,
so no accidental fallback to a shared account):
  DASHBOARD_CHAT_BASE_URL  e.g. https://api.anthropic.com/v1
                                https://api.exchangetoken.ai/v1
                                http://localhost:11434/v1  (Ollama)
  DASHBOARD_CHAT_API_KEY   your own API key (sk-...)
  DASHBOARD_CHAT_MODEL     e.g. claude-sonnet-4-6 / gpt-4o / kimi-k2
  DASHBOARD_CHAT_TIMEOUT   per-call HTTP timeout in seconds (default 300)

See docs/DASHBOARD.md for provider-specific setup.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
import urllib.error
from pathlib import Path

from flask import Blueprint, jsonify, request

chat_bp = Blueprint("chat", __name__)


# NO hard-coded defaults — users must supply their own key. This is
# deliberate to prevent shared-account leakage in an open-source release.
_DEFAULT_BASE_URL = os.environ.get("DASHBOARD_CHAT_BASE_URL", "").rstrip("/")
_DEFAULT_API_KEY = os.environ.get("DASHBOARD_CHAT_API_KEY", "")
_DEFAULT_MODEL = os.environ.get("DASHBOARD_CHAT_MODEL", "claude-sonnet-4-6")


# ─── Skill context: full SKILL.md + Sionna 2.0 API reference ─────
# Turns this dashboard's /api/chat into the "with_skill" AutoNetSim
# condition (rather than the naive baseline) by loading the same
# procedural skill file the benchmark agent consumes.
#
# We inject:
#   1. SKILL.md                    — routing tree, templates, workflow
#   2. references/sionna-v2-api.md — API signatures + gotchas
#
# This is ~20-25 K tokens which fits comfortably in Sonnet's 200K
# window. Other references (channel-models, materials, etc.) are
# skipped by default — activate them by name via the env override.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SKILL_ROOT = _REPO_ROOT / ".claude" / "skills" / "rf-simulator"

# Files loaded into the chat's system prompt. Absolute paths OR paths
# relative to _SKILL_ROOT. Task-tail is loaded from benchmark/ because it
# holds the exact signature block (samples_per_src=…, paths.a tuple, etc.)
# that the benchmark uses as an authoritative Sionna 2.0 contract.
_DEFAULT_SKILL_FILES = [
    "SKILL.md",
    "references/sionna-v2-api.md",
    str(_REPO_ROOT / "benchmark" / "prompts" / "task_tail.txt"),
]


def _read(rel: str) -> str:
    p = Path(rel)
    if not p.is_absolute():
        p = _SKILL_ROOT / rel
    try:
        return p.read_text()
    except Exception:
        return ""


def _load_skill_context() -> str:
    """Assemble the skill payload injected into the chat system prompt."""
    parts: list[str] = []
    files = os.environ.get("DASHBOARD_CHAT_SKILL_FILES", "").strip()
    files = files.split(",") if files else _DEFAULT_SKILL_FILES
    for rel in files:
        rel = rel.strip()
        if not rel:
            continue
        body = _read(rel)
        if body:
            label = Path(rel).name
            parts.append(f"===== BEGIN {label} =====\n{body}\n===== END {label} =====")
    return "\n\n".join(parts)


_SKILL_CONTEXT = _load_skill_context()


def _cfg():
    """Read chat provider config from environment. Raises a helpful error
    when either the base URL or the API key is missing — no silent fallback
    to a shared account, so users always bring their own credentials."""
    base_url = os.environ.get("DASHBOARD_CHAT_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")
    api_key  = os.environ.get("DASHBOARD_CHAT_API_KEY", _DEFAULT_API_KEY)
    model    = os.environ.get("DASHBOARD_CHAT_MODEL", _DEFAULT_MODEL)
    return base_url, api_key, model


def _cfg_error_or_none():
    """Return a user-facing error string when config is incomplete, else None."""
    base_url, api_key, _ = _cfg()
    missing = []
    if not base_url: missing.append("DASHBOARD_CHAT_BASE_URL")
    if not api_key:  missing.append("DASHBOARD_CHAT_API_KEY")
    if missing:
        return (f"Chat is disabled — please set: {', '.join(missing)}. "
                f"See docs/DASHBOARD.md for provider-specific setup "
                f"(Anthropic direct, OpenAI-compat relay, or a local model server).")
    return None


def _chat_timeout() -> float:
    """Per-call HTTP timeout. The worker prompt carries the full SKILL.md
    (~35k tokens) and may emit up to 4k tokens, so 60 s is too tight for
    large models behind a gateway."""
    try:
        return float(os.environ.get("DASHBOARD_CHAT_TIMEOUT", "300"))
    except ValueError:
        return 300.0


def _openai_chat(messages, base_url, api_key, model, max_tokens=8192):
    """One-shot call to an OpenAI-compat /v1/chat/completions endpoint."""
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens}
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_chat_timeout()) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read().decode('utf-8', 'ignore')[:200]}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    if "error" in body:
        return None, str(body["error"])[:200]
    try:
        return body["choices"][0]["message"]["content"], None
    except (KeyError, IndexError) as e:
        return None, f"unexpected response shape: {e}"


def _scene_context(data):
    """Compact scene state into a system-style preface. Includes the
    concrete furniture inventory so the worker can refuse move/rotate/
    remove requests for pieces that don't exist and pick sensible
    positions when the user says 'move it to the corner'."""
    scene = data.get("scene") or {}
    lines = []
    if scene.get("type") == "indoor" and scene.get("room"):
        r = scene["room"]
        w, l, h = r.get("width", "?"), r.get("length", "?"), r.get("height", 3.0)
        furn = r.get("furniture") or []
        lines.append(f"current scene: indoor {w} m × {l} m × {h} m, "
                     f"{len(furn)} furniture items")
        if furn:
            # List actual placed pieces with position + rotation so Sonnet
            # can reason about "the chair near the window" or "rotate the
            # desk to face north".
            lines.append("furniture inventory:")
            for i, f in enumerate(furn):
                cat = f.get("category", "?")
                fx, fy = f.get("x", 0), f.get("y", 0)
                fw, fd = f.get("width", "?"), f.get("depth", "?")
                theta = f.get("theta", 0)
                lines.append(
                    f"  [{i}] {cat} at ({fx:.2f}, {fy:.2f}), "
                    f"{fw}×{fd} m, rotated {theta:.0f}°")
    elif scene.get("type") == "outdoor":
        lines.append(
            f"current scene: outdoor {scene.get('width','?')}m x"
            f" {scene.get('length','?')}m,"
            f" {len(scene.get('buildings', []))} buildings"
        )
    else:
        lines.append("no scene loaded yet — the user must build a room "
                     "first before you can move / rotate / remove furniture. "
                     "For a 'build ...' request you must emit set_room_size "
                     "and add_furniture actions.")
    # Measured coverage results, newest last. This is the ONLY source of
    # truth for "how did the signal change" questions.
    hist = data.get("coverage_history") or []
    if hist:
        lines.append("coverage results so far (measured by the ray tracer, "
                     "newest last — quote these numbers; never estimate "
                     "them when a measurement exists):")
        for i, c in enumerate(hist[-4:]):
            tx = c.get("tx") or {}
            lines.append(
                f"  run {i+1}: {c.get('freq_ghz', '?')} GHz, AP at "
                f"({tx.get('x', '?')}, {tx.get('y', '?')}, {tx.get('z', '?')}) m, "
                f"engine {c.get('engine', '?')}: mean {c.get('mean_dbm', '?')} dBm, "
                f"min {c.get('min_dbm', '?')} dBm, max {c.get('max_dbm', '?')} dBm, "
                f"{c.get('coverage_pct', '?')}% of cells above -80 dBm"
                + (("; per-room mean: " + ", ".join(
                        f"{k} {v} dBm" for k, v in c["rooms"].items()))
                   if c.get("rooms") else ""))
        if len(hist) >= 2:
            a, b = hist[-2], hist[-1]
            try:
                lines.append(f"  latest change in mean RSS: "
                             f"{float(b['mean_dbm']) - float(a['mean_dbm']):+.1f} dB "
                             f"(run {min(len(hist),4)-1} -> run {min(len(hist),4)})")
            except (KeyError, TypeError, ValueError):
                pass
    else:
        lines.append("no coverage has been computed yet in this session.")
    ib = data.get("inside_building")
    sb = data.get("selected_building")
    if ib:
        lines.append(f"user is INSIDE building '{ib.get('name') or ib.get('id','?')}'")
    elif sb:
        lines.append(f"user has selected building '{sb.get('name') or sb.get('id','?')}'")
    return "\n".join(lines)


_WORKER_SYSTEM = (
    "You are the WORKER agent for the AutoNetSim wireless simulation "
    "dashboard. You have full access to the rf-simulator procedural "
    "skill below. Read the user's request and produce a technical "
    "response: Sionna 2.0 code that will actually run, key numerical "
    "results, and short reasoning. Any code you write MUST conform "
    "exactly to the Sionna 2.0 API in the reference — v1 signatures "
    "(sionna.fec.*, sionna.channel.*, tf.math.*, `num_samples`) are "
    "gone. Quote a signature verbatim rather than inventing one. "
    "Your output is NOT shown to the end user directly — a general "
    "agent will summarize it — so favor completeness and precision "
    "over friendliness.\n\n"
    "**IMPORTANT — UI ACTIONS.**\n"
    "If (and only if) the user's request implies operations the "
    "dashboard UI can perform, emit one or more JSON action blocks in "
    "your reply so the frontend can execute them. Wrap each block "
    "exactly like this — no extra text inside the fence — and use "
    "the action types below:\n\n"
    "```action\n"
    '{"type": "set_room_size", "width": 6.0, "length": 5.0}\n'
    "```\n\n"
    "Supported action types:\n"
    "- `set_room_size` → `{width, length}` (meters)\n"
    "- `add_furniture` → `{items: [{quantity, category}, ...], "
    "room_width, room_length}` — categories: bed, desk, chair, sofa, "
    "wardrobe, nightstand, bookcase, cabinet, table, coffee_table, "
    "tv_stand, lamp\n"
    "- `set_tx_position` → `{x, y, z}` in meters. USE THIS whenever "
    "the user asks to move / relocate / reposition the AP, "
    "transmitter, or antenna. Coordinates are in the room frame "
    "(x=width axis, y=length axis, z=height above floor).\n"
    "- `set_tx_power` → `{power_dbm}`\n"
    "- `set_frequency` → `{ghz}` (e.g. 2.4, 5, 28, 60)\n"
    "- `move_furniture` → `{category | index, x, y}` — pick furniture "
    "by category (\"desk\", \"chair\", \"sofa\") or list index. "
    "Coordinates in room frame.\n"
    "- `rotate_furniture` → `{category | index, absolute_deg | delta_deg}` "
    "— set a new heading, or add a delta (positive = counterclockwise).\n"
    "- `remove_furniture` → `{category | index}` — delete one piece.\n"
    "- `set_room_height` → `{height}` (meters) — updates the ceiling / "
    "coverage layer height.\n"
    "- `set_material` → `{surface: interior_wall|exterior_wall|wall|floor|"
    "ceiling, material: plasterboard|brick|concrete|wood|glass|marble|"
    "metal}`. On a loaded apartment this rewrites the scene's ray-tracing "
    "materials immediately and coverage is recomputed automatically; use "
    "`interior_wall` for the partitions between rooms. (plasterboard = "
    "drywall.)\n"
    "- `configure_antenna` → `{pattern: iso|tr38901, polarization: "
    "V|H|VH, rows, cols, azimuth, elevation}` — changes the WHOLE "
    "antenna. Use this only when the user asks to change the pattern "
    "or the array size.\n"
    "- `set_ap_orientation` → `{azimuth, elevation}` in degrees. USE "
    "THIS when the user asks only to rotate / re-aim / re-point the "
    "AP. Azimuth 0° = +X axis (east), 90° = +Y (north), 180° = west, "
    "270° = south. Elevation 0° = horizontal, negative = downtilt "
    "(typical -15° to -30° for ceiling APs), positive = uptilt.\n"
    "- `compute_coverage` → `{}` (triggers a fresh coverage run)\n"
    "- `load_scene` → `{scene_id}`\n"
    "- `create_apartment` → `{rooms: [{id, type, width, depth}, ...], "
    "name?, height?}` — builds a MULTI-ROOM apartment with interior "
    "walls, door openings and room-appropriate 3D furniture, then loads "
    "it. USE THIS whenever the user describes a home / flat / apartment "
    "with more than one room (e.g. 'two bedrooms, a living room, a "
    "kitchen and a bathroom'). type must be one of living_room, kitchen, "
    "bedroom, bathroom, balcony. Sizes in meters; if the user gives none, "
    "use typical ones (living_room 5x4, kitchen 3x4, bedroom 4x3.5, "
    "bathroom 2x3.5, balcony 3x2; height 3.0 m unless the user says "
    "otherwise). Rooms are laid out in a 2-column grid "
    "in list order, so list living_room first. Give distinct ids "
    "(bedroom1, bedroom2). Do NOT combine with set_room_size / "
    "add_furniture — create_apartment already builds and loads "
    "everything; follow it with compute_coverage only.\n"
    "- `create_outdoor` → `{}`\n"
    "- `fetch_osm` → `{name, lat, lon, radius}`\n\n"
    "Emit action blocks eagerly when the user asks to build / "
    "modify / view a scene, add furniture, MOVE THE AP, tune "
    "antennas, change TX power / frequency, or run a coverage / "
    "BER simulation. If the user is only asking a conceptual "
    "question ('what is path gain'), emit NO action blocks.\n\n"
    "**Coordinates must respect the current room bounds.** If the "
    "room is W×L×H meters, x∈[0, W], y∈[0, L], z∈[0.1, H]. Furniture "
    "must sit inside the walls (leave half its footprint clearance "
    "from every edge). Coordinates outside these bounds will be "
    "silently clamped by the dashboard and the user will be shown a "
    "'clamped' notice — so pick sensible in-bounds values from the "
    "start. When the user only gives a category (e.g. 'move the "
    "desk to the corner'), interpret 'corner' concretely: (0.5, "
    "0.5), (W-0.5, 0.5), etc. Never emit negative coordinates.\n\n"
    "**Default room height is 3.0 m** when the user does not specify "
    "one. If the user asks to mount the AP higher than 3.0 m, emit a "
    "`set_room_height` action FIRST so the ceiling is raised before "
    "the AP is placed. Same rule for `add_furniture` with tall items "
    "(wardrobe = 2.1 m — fits in 3 m by default, no action needed).\n\n"
    "**Only act on furniture that actually exists.** The scene "
    "context below lists the current furniture inventory with exact "
    "categories and positions. If the user asks to move / rotate / "
    "remove a piece that is NOT in the inventory (e.g. 'move the "
    "piano' when there is no piano), DO NOT emit the corresponding "
    "action — instead, in your prose reply, tell the user which "
    "pieces are actually available and ask whether they want to add "
    "the missing piece first. Same rule applies when no scene is "
    "loaded: refuse to emit move/rotate/remove actions until "
    "add_furniture has run.\n\n"
    "**Action ordering rules** — when emitting multiple action "
    "blocks in one reply, order them so downstream actions see the "
    "correct scene state:\n"
    "  1. set_material (before any scene creation, so materials apply)\n"
    "  2. set_room_size or set_room_height (define the box)\n"
    "  3. add_furniture (creates or rebuilds the scene)\n"
    "  4. set_tx_position / set_tx_power / set_frequency / configure_antenna\n"
    "  5. move_furniture / rotate_furniture / remove_furniture "
    "(tweak the just-built scene)\n"
    "  6. compute_coverage (always LAST — needs everything above)\n"
    "Deviating from this order will cause materials to be silently "
    "queued for later, or downstream actions to reference "
    "non-existent furniture."
)


_GENERAL_SYSTEM = (
    "You are the GENERAL assistant for the AutoNetSim wireless "
    "simulation dashboard — the friendly face users interact with.\n\n"
    "A technical worker agent has just prepared a detailed answer to "
    "the user's request. Your job is to read the worker's output and "
    "produce a SHORT, conversational reply for the user.\n\n"
    "Rules:\n"
    "- 2 to 4 sentences maximum, plus at most one small markdown bullet "
    "list if it clarifies things.\n"
    "- Confirm what will be done (or has been done) in plain language.\n"
    "- Highlight the key numerical result if one is present (e.g. "
    "'path gain came out to **-68.3 dB**'). Use bold for numbers.\n"
    "- NEVER show code, PARAMS blocks, capability tags, or the "
    "worker's internal reasoning steps.\n"
    "- If the request is under-specified, ask ONE precise follow-up "
    "question at the end.\n"
    "- Any action blocks the worker emitted are executed automatically by "
    "the dashboard (including recomputing coverage). NEVER ask 'want me "
    "to run it?' — state what was done.\n"
    "- When the worker's output contains measured coverage numbers "
    "(mean / min / max dBm, coverage %, or a before-vs-after delta), "
    "report them verbatim. If the user asks how something changed and no "
    "measurement exists yet, say the new run is in progress and the "
    "numbers will appear in the stats panel — do not invent values.\n"
    "- Match the language the user wrote in (English → English, "
    "Chinese → Chinese).\n"
    "- Never mention 'worker agent' or 'orchestrator' — those are "
    "implementation details.\n"
    "- Describe only what the dashboard actually does: ONE ray-traced "
    "coverage run per request, reporting whole-scene mean/min/max RSS, "
    "coverage %, and per-room mean RSS when rooms exist. Do NOT invent "
    "procedures such as 'two-pass runs', 'bedroom-only slicing', or "
    "'before/after comparisons being set up' — if a comparison is wanted, "
    "quote the previous measured run from the context.\n"
    "- Do NOT claim the AP was placed or coverage was started unless the "
    "worker actually emitted set_tx_position / compute_coverage."
)


def _build_worker_system(data: dict) -> str:
    parts = [_WORKER_SYSTEM]
    if _SKILL_CONTEXT:
        parts.append(_SKILL_CONTEXT)
    ctx = _scene_context(data)
    if ctx:
        parts.append("Current dashboard state:\n" + ctx)
    return "\n\n".join(parts)


def _messages_for_llm(system: str, user_msgs: list) -> list:
    out = [{"role": "system", "content": system}]
    for m in user_msgs:
        role = m.get("role", "user")
        content = m.get("content", "")
        if not content:
            continue
        if role not in ("user", "assistant", "system"):
            role = "user"
        out.append({"role": role, "content": content})
    return out


def _summarize(user_msg: str, worker_output: str, base_url: str,
                api_key: str, model: str) -> tuple[str | None, str | None]:
    messages = [
        {"role": "system", "content": _GENERAL_SYSTEM},
        {"role": "user", "content": (
            f"User request:\n{user_msg}\n\n"
            f"Worker's technical output (do not repeat verbatim):\n"
            f"{worker_output}"
        )},
    ]
    # Summaries are short — cap output to keep latency down.
    return _openai_chat(messages, base_url, api_key, model, max_tokens=512)


_ACTION_FENCE_RE = re.compile(r"```action\s*\n(.*?)```", re.DOTALL)


def _extract_actions(worker_out: str) -> list[dict]:
    """Parse ```action …``` fences into a list of action dicts.

    Silently drop malformed JSON blocks so a partially bad worker
    reply still yields any well-formed actions.
    """
    actions: list[dict] = []
    for m in _ACTION_FENCE_RE.finditer(worker_out or ""):
        raw = m.group(1).strip()
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        if isinstance(obj, dict) and obj.get("type"):
            actions.append(obj)
        elif isinstance(obj, list):
            for a in obj:
                if isinstance(a, dict) and a.get("type"):
                    actions.append(a)
    return actions


def _strip_action_fences(text: str) -> str:
    """Remove all ```action …``` fences from text before summarizing."""
    return _ACTION_FENCE_RE.sub("", text or "").strip()


@chat_bp.route("/api/chat", methods=["POST"])
def chat():
    """Two-agent + action-block architecture:
      worker (skill-loaded) → emits action blocks + technical prose
        → general agent summarizes prose for the user
        → chat.py returns {reply, actions} for the frontend to execute.
    """
    data = request.json or {}
    user_msgs = data.get("messages") or []
    if not user_msgs:
        return jsonify({"error": "No messages provided"}), 400

    # Guard: refuse to run if the user hasn't configured a provider yet.
    cfg_err = _cfg_error_or_none()
    if cfg_err:
        return jsonify({
            "reply": cfg_err, "actions": None, "panes": [],
            "files_produced": [], "error": "not_configured",
        }), 200

    base_url, api_key, model = _cfg()
    # Diagnostic: what the browser believes the measured history is.
    logger.info("chat: scene=%s coverage_history=%s",
                (data.get("scene") or {}).get("scene_id"),
                json.dumps(data.get("coverage_history") or [])[:2000])
    last_user = ""
    for m in reversed(user_msgs):
        if m.get("role") == "user" and m.get("content"):
            last_user = m["content"]
            break

    # Step 1 — worker agent produces technical answer + action blocks.
    worker_msgs = _messages_for_llm(_build_worker_system(data), user_msgs)
    # 2048 is plenty for action blocks + reasoning; the general agent
    # summarizes anyway, and halving the cap halves worst-case latency.
    worker_out, err = _openai_chat(worker_msgs, base_url, api_key, model,
                                    max_tokens=2048)
    if err:
        return jsonify({
            "reply": f"[worker error] {err}",
            "actions": None, "panes": [], "files_produced": [], "error": err,
        }), 200

    # Step 2 — extract action blocks; strip them from the text sent to
    # the general agent so the summary reads naturally.
    actions = _extract_actions(worker_out or "")
    prose_for_summary = _strip_action_fences(worker_out or "")

    # Step 3 — general agent summarizes for the user.
    reply, err = _summarize(last_user, prose_for_summary, base_url,
                            api_key, model)
    if err or not reply:
        reply = prose_for_summary or f"[general agent error] {err or 'empty reply'}"

    return jsonify({
        "reply": reply,
        "actions": actions or None,
        "panes": [],
        "layout": {"auto": True},
        "files_produced": [],
    })
