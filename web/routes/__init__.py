"""Dashboard route blueprints.

Chatbox policy (per references/viewer-spec.md): the chat blueprint is
registered ONLY when ANTHROPIC_API_KEY is set in the environment. With no
key, the dashboard renders without a chat sidebar (the Jinja2
`{% if chat_enabled %}` guard in templates/dashboard.html omits the
chat HTML, so users never see a dead input field).

Other route imports (coverage/catalog/creation/rays/ase/segmentation) are
guarded with try/except so a missing module doesn't crash the whole app —
the restored web/ is a partial port from older commits and not all
blueprints have been brought forward yet.
"""
from __future__ import annotations
import os
import sys


def chat_enabled() -> bool:
    """True iff a chat backend can serve /api/chat. Currently only
    Anthropic is supported; later this can fall through to a local LLM
    if OPENAI_BASE_URL is set."""
    return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())


ALL_BLUEPRINTS = []


def _try_register(import_path: str, attr: str) -> None:
    try:
        mod = __import__(import_path, fromlist=[attr])
        ALL_BLUEPRINTS.append(getattr(mod, attr))
    except Exception as e:
        print(f"[routes] {import_path} not registered: {e}", file=sys.stderr)


# Always-on routes (page rendering + scene CRUD).
_try_register("routes.pages", "pages_bp")
_try_register("routes.scenes", "scenes_bp")

# Agent-step endpoint — always registered so the chatbox can forward
# prompts even when the Anthropic Haiku key is not set (the agent uses the
# `claude` CLI rather than the Anthropic SDK directly).
_try_register("routes.agent", "agent_bp")

# Chat blueprint forwards to the agent endpoint (which uses the `claude`
# CLI, not the Anthropic SDK directly), so it works without an
# ANTHROPIC_API_KEY too. Keep it always-registered.
_try_register("routes.chat", "chat_bp")

# Optional / not-yet-ported routes — silently skip if missing.
for path, attr in [
    ("routes.coverage", "coverage_bp"),
    ("routes.catalog", "catalog_bp"),
    ("routes.creation", "creation_bp"),
    ("routes.rays", "rays_bp"),
]:
    _try_register(path, attr)
