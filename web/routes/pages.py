"""Page routes, SSE progress endpoint, and job management."""

import json
import os
import time
from pathlib import Path

from flask import Blueprint, Response, jsonify, render_template, request

from routes.shared import _get_job, _cancel_job


def _chat_enabled() -> bool:
    """Chat is always enabled — /api/chat now uses the exchangetoken
    OpenAI-compat gateway with a bundled default key (K3 in api-key.txt).
    Override via env DASHBOARD_CHAT_{BASE_URL,API_KEY,MODEL} in chat.py."""
    return True


pages_bp = Blueprint("pages", __name__)


@pages_bp.route("/")
def index():
    # Chat is gated by ANTHROPIC_API_KEY (see references/viewer-spec.md).
    # When the key is absent, the chatbox HTML is omitted entirely so
    # users don't see a non-functional input field.
    return render_template("dashboard.html",
                           chat_enabled=_chat_enabled(),
                           chat_model=os.environ.get("DASHBOARD_CHAT_MODEL", ""))


@pages_bp.route("/skills")
def skills_page():
    return render_template("skills.html")


@pages_bp.route("/api/skills")
def api_skills():
    """Return structured skill data for the skills page."""
    skills_dir = Path(".claude/skills")
    skills = []
    if skills_dir.exists():
        for skill_dir in sorted(skills_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            text = skill_md.read_text()
            # Parse frontmatter
            name, description = skill_dir.name, ""
            if text.startswith("---"):
                parts = text.split("---", 2)
                if len(parts) >= 3:
                    for line in parts[1].strip().splitlines():
                        if line.startswith("name:"):
                            name = line.split(":", 1)[1].strip()
                        elif line.startswith("description:"):
                            description = line.split(":", 1)[1].strip()
                    text = parts[2].strip()

            # Extract sections
            sections = []
            current_heading = None
            current_lines = []
            for line in text.splitlines():
                if line.startswith("## "):
                    if current_heading:
                        sections.append({"heading": current_heading, "content": "\n".join(current_lines).strip()})
                    current_heading = line[3:].strip()
                    current_lines = []
                else:
                    current_lines.append(line)
            if current_heading:
                sections.append({"heading": current_heading, "content": "\n".join(current_lines).strip()})

            # List files in skill directory
            files = [f.name for f in sorted(skill_dir.rglob("*")) if f.is_file()]

            skills.append({
                "name": name,
                "description": description,
                "directory": str(skill_dir),
                "files": files,
                "sections": sections,
            })

    # Gather project modules from src/
    modules = []
    src_dir = Path("src")
    if src_dir.exists():
        for mod_dir in sorted(src_dir.iterdir()):
            if mod_dir.is_dir() and not mod_dir.name.startswith("_"):
                py_files = [f.name for f in mod_dir.glob("*.py") if f.name != "__init__.py"]
                modules.append({"name": mod_dir.name, "files": py_files})

    # Gather MCP / tool info
    tools = ["Bash", "Python", "File System"]

    return jsonify({
        "skills": skills,
        "modules": modules,
        "tools": tools,
    })


@pages_bp.route("/api/progress/<job_id>")
def progress(job_id):
    """SSE endpoint that streams job progress updates.

    Supports progressive slice streaming: if the background job appends
    slices via _add_job_slice(), each SSE poll includes only the *new*
    slices since the last message (tracked by slice_sent_idx).

    Automatically stops streaming when:
    - Job completes (done/error status)
    - Client disconnects (generator cleanup)
    - Job not found
    - 5-minute timeout (prevents orphaned SSE streams)
    """
    max_stream_seconds = 300  # 5-minute safety timeout

    def stream():
        start_time = time.time()
        slice_sent_idx = 0  # track how many slices we've already sent

        while True:
            # Safety timeout to prevent orphaned streams
            if time.time() - start_time > max_stream_seconds:
                yield f"data: {json.dumps({'status': 'error', 'message': 'Stream timeout'})}\n\n"
                break

            job = _get_job(job_id)
            if job is None:
                yield f"data: {json.dumps({'error': 'unknown job'})}\n\n"
                break

            payload = {
                "status": job["status"],
                "progress": job["progress"],
                "message": job["message"],
            }

            # Include any new slices since last SSE message
            job_slices = job.get("slices")
            if job_slices and len(job_slices) > slice_sent_idx:
                payload["new_slices"] = job_slices[slice_sent_idx:]
                payload["total_slices_so_far"] = len(job_slices)
                slice_sent_idx = len(job_slices)

            if job["status"] == "done":
                payload["result"] = job["result"]
                yield f"data: {json.dumps(payload)}\n\n"
                break
            if job["status"] == "error":
                yield f"data: {json.dumps(payload)}\n\n"
                break

            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(0.3)

    return Response(stream(), mimetype="text/event-stream")


@pages_bp.route("/api/jobs/<job_id>/cancel", methods=["POST"])
def cancel_job(job_id):
    """Cancel a running background job.

    The job's background thread should check is_job_cancelled() periodically
    and stop work early when True.
    """
    if _cancel_job(job_id):
        return jsonify({"cancelled": True})
    return jsonify({"cancelled": False, "error": "Job not found or already finished"}), 404
