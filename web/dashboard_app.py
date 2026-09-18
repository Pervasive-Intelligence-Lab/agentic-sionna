"""Flask-based live dashboard for Sionna RF simulation.

Provides API routes for scene creation, coverage computation, and BER analysis.
Serves an interactive frontend with Plotly.js visualizations and SSE progress.

Usage:
    python dashboard_app.py          # Start on port 8080
    python dashboard_app.py --port 9000  # Custom port
"""

import argparse
import json
import logging
import math
import os
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_file

from src.models.room import Door, FurnitureItem, Position, Room
from src.models.outdoor import Building, GroundPlane, OutdoorScene
from src.models.scene import Scene
from src.models.building_room import building_to_room
from src.optimizer.layout import FurnitureSpec, optimize_layout, BoundingBox
from src.exporters.xml import XMLExporter, get_material_for_category
from src.wireless.ray_tracing import compute_thz_coverage, THzConfig

logger = logging.getLogger(__name__)

# Load environment from repo-root .env file (gitignored). Users copy
# .env.example → .env and fill in their own API keys / paths.
# Existing shell env vars take precedence over .env — no override.
_REPO_ROOT = Path(__file__).resolve().parent.parent
try:
    from routes.env_loader import load_env_file
    load_env_file(_REPO_ROOT / ".env")
except Exception as _e:
    pass  # env_loader is optional — env vars can also be set by the shell

app = Flask(__name__, template_folder="templates")

# In-memory stores
_jobs: Dict[str, Dict[str, Any]] = {}
_scenes: Dict[str, Dict[str, Any]] = {}
_outdoor_scenes: Dict[str, OutdoorScene] = {}

OUTPUTS_DIR = (Path(__file__).resolve().parent / "outputs").resolve()
OUTPUTS_DIR.mkdir(exist_ok=True)

# Lazy-loaded 3D-FUTURE catalog
_catalog = None
_catalog_error = None


def _get_catalog():
    """Lazy-init FutureCatalog. Returns None if dataset unavailable.

    Dataset path is `data/3D-FUTURE-model` by default; can be overridden
    with the FUTURE_DATASET_PATH env var (matches routes/shared.py's
    catalog loader so a single env var works for both code paths).
    """
    global _catalog, _catalog_error
    if _catalog is not None:
        return _catalog
    if _catalog_error is not None:
        return None
    try:
        from src.catalog.furniture import FutureCatalog
        dataset_path = Path(
            os.environ.get("FUTURE_DATASET_PATH", "data/3D-FUTURE-model")
        ).expanduser()
        if not dataset_path.exists():
            _catalog_error = "3D-FUTURE dataset not found"
            logger.warning("3D-FUTURE catalog not available: %s (path=%s)",
                           _catalog_error, dataset_path)
            return None
        _catalog = FutureCatalog(dataset_path)
        logger.info("Loaded 3D-FUTURE catalog with %d models", len(_catalog))
        return _catalog
    except Exception as e:
        _catalog_error = str(e)
        logger.warning("Failed to load 3D-FUTURE catalog: %s", e)
        return None


# Generic furniture dimensions (width, depth, height) for dashboard-created items
GENERIC_FURNITURE_DIMS: Dict[str, BoundingBox] = {
    "bed": BoundingBox(width=1.5, depth=2.0, height=0.6),
    "desk": BoundingBox(width=1.2, depth=0.6, height=0.75),
    "chair": BoundingBox(width=0.5, depth=0.5, height=0.9),
    "sofa": BoundingBox(width=2.0, depth=0.9, height=0.85),
    "wardrobe": BoundingBox(width=1.2, depth=0.6, height=2.0),
    "nightstand": BoundingBox(width=0.5, depth=0.4, height=0.55),
    "bookcase": BoundingBox(width=0.8, depth=0.35, height=1.8),
    "cabinet": BoundingBox(width=0.8, depth=0.45, height=0.9),
    "table": BoundingBox(width=1.0, depth=1.0, height=0.75),
    "coffee_table": BoundingBox(width=1.0, depth=0.6, height=0.45),
    "tv_stand": BoundingBox(width=1.2, depth=0.4, height=0.5),
    "lamp": BoundingBox(width=0.3, depth=0.3, height=1.5),
}


# ─────────────────────────────────────────────────────────
# SSE progress helpers
# ─────────────────────────────────────────────────────────

def _create_job() -> str:
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {"status": "running", "progress": 0, "message": "", "result": None}
    return job_id


def _scene_name(prefix: str, detail: str, job_id: str) -> str:
    """Generate a descriptive scene directory name.

    Examples: Room_5x4_a1b2c3, Office_8x6_d4e5f6, Athens_Downtown_f7a8b9
    """
    # Sanitize detail: keep alphanumeric, spaces→underscores, truncate
    clean = "".join(c if c.isalnum() or c in " _-." else "" for c in detail)
    clean = clean.strip().replace(" ", "_")[:30]
    tag = job_id[:6]
    if clean:
        return f"{prefix}_{clean}_{tag}"
    return f"{prefix}_{tag}"


def _update_job(job_id: str, progress: int, message: str) -> None:
    if job_id in _jobs:
        _jobs[job_id]["progress"] = progress
        _jobs[job_id]["message"] = message


def _finish_job(job_id: str, result: Any) -> None:
    if job_id in _jobs:
        _jobs[job_id]["status"] = "done"
        _jobs[job_id]["progress"] = 100
        _jobs[job_id]["result"] = result


def _fail_job(job_id: str, error: str) -> None:
    if job_id in _jobs:
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["message"] = error


# ─────────────────────────────────────────────────────────
# Routes: pages
# ─────────────────────────────────────────────────────────

@app.route("/")
def index():
    # The chatbox uses the `claude` CLI (OAuth-backed), not the
    # ANTHROPIC_API_KEY env var, so it works regardless of whether a key
    # is set. Always render the sidebar.
    return render_template("dashboard.html", chat_enabled=True,
                           chat_model=os.environ.get("DASHBOARD_CHAT_MODEL", ""))


@app.route("/skills")
def skills_page():
    return render_template("skills.html")


@app.route("/api/skills")
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


# ─────────────────────────────────────────────────────────
# Routes: SSE progress
# ─────────────────────────────────────────────────────────

@app.route("/api/progress/<job_id>")
def progress(job_id):
    # Look in BOTH the inline _jobs dict (legacy fast-sync flows) and the
    # shared store routes.shared._jobs (agent_bp's async invocations).
    # Without this, /api/agent/step-async creates a job that the inline
    # progress endpoint can't see → SSE clients see "unknown job".
    from routes.shared import _get_job as _get_shared_job

    def stream():
        while True:
            job = _jobs.get(job_id) or _get_shared_job(job_id)
            if job is None:
                yield f"data: {json.dumps({'error': 'unknown job'})}\n\n"
                break
            payload = {
                "status": job["status"],
                "progress": job["progress"],
                "message": job["message"],
            }
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


# ─────────────────────────────────────────────────────────
# Routes: fast sync
# ─────────────────────────────────────────────────────────

@app.route("/api/scenes/list")
def list_scenes():
    scenes = []
    for p in sorted(OUTPUTS_DIR.iterdir()):
        if p.is_dir():
            entry = {"id": p.name, "files": [f.name for f in p.iterdir()]}
            # Read display name from metadata if available
            meta_path = p / "metadata.json"
            if meta_path.exists():
                try:
                    with open(meta_path) as mf:
                        meta = json.load(mf)
                    entry["name"] = meta.get("name", p.name)
                    entry["type"] = meta.get("type", "unknown")
                except Exception:
                    pass
            scenes.append(entry)
    return jsonify(scenes)


@app.route("/api/scenes/<scene_id>/delete", methods=["DELETE"])
def delete_scene(scene_id):
    """Delete a scene from the outputs directory."""
    import shutil
    scene_dir = OUTPUTS_DIR / scene_id
    if not scene_dir.is_dir():
        return jsonify({"error": "Scene not found"}), 404
    shutil.rmtree(scene_dir)
    return jsonify({"deleted": scene_id})


@app.route("/api/scenes/<scene_id>/export", methods=["POST"])
def export_scene(scene_id):
    """Export a scene in the requested format (glb, obj, xml)."""
    import io
    scene_dir = OUTPUTS_DIR / scene_id
    if not scene_dir.is_dir():
        return jsonify({"error": "Scene not found"}), 404

    fmt = (request.json or {}).get("format", "xml")

    # XML — serve existing scene.xml directly
    if fmt == "xml":
        xml_path = scene_dir / "scene.xml"
        if not xml_path.exists():
            return jsonify({"error": "No scene.xml found"}), 404
        return send_file(str(xml_path), mimetype="application/xml",
                         as_attachment=True, download_name=f"{scene_id}.xml")

    # GLB / OBJ — reconstruct from metadata
    meta_path = scene_dir / "metadata.json"
    if not meta_path.exists():
        return jsonify({"error": "No metadata.json found for reconstruction"}), 404

    try:
        with open(meta_path) as f:
            meta = json.load(f)

        from src.exporters.gltf import GLTFExporter
        scene_type = meta.get("type", "indoor")
        catalog = _get_catalog()

        if scene_type == "outdoor":
            # Reconstruct OutdoorScene with buildings
            buildings = []
            for bd in meta.get("buildings", []):
                fp = bd.get("footprint", bd.get("fp", []))
                if len(fp) >= 3:
                    buildings.append(Building(
                        id=bd.get("id", str(uuid.uuid4())),
                        footprint=[tuple(p) for p in fp],
                        height=float(bd.get("height", bd.get("h", 12))),
                        material=bd.get("material", "concrete"),
                        name=bd.get("name"),
                    ))
            ow = float(meta.get("width", 200))
            ol = float(meta.get("length", 200))
            ground = GroundPlane(width=ow, length=ol, material="wet_ground")
            outdoor = OutdoorScene(width=ow, length=ol, buildings=buildings, ground=ground)
            scene_obj = Scene(outdoor=outdoor)
        else:
            # Reconstruct indoor Room with furniture
            furniture_data = meta.get("furniture", [])
            room_w = float(meta.get("width", 5))
            room_l = float(meta.get("length", 4))
            room_h = float(meta.get("height", 2.7))

            furniture = []
            for fi in furniture_data:
                mid = fi.get("model_id", "")
                # Resolve model_path from catalog if available
                mpath = fi.get("model_path", "")
                if not mpath and catalog and mid:
                    try:
                        mpath = str(catalog.get_model_path(mid) / "raw_model.obj")
                    except Exception:
                        mpath = ""
                furniture.append(FurnitureItem(
                    id=fi.get("id", str(uuid.uuid4())),
                    category=fi.get("category", "unknown"),
                    model_id=mid or "generic",
                    model_path=mpath,
                    position=Position(
                        x=float(fi.get("x", 0)),
                        y=float(fi.get("y", 0)),
                        theta=float(fi.get("theta", 0)),
                    ),
                    dimensions=BoundingBox(
                        width=float(fi.get("width", 0.5)),
                        depth=float(fi.get("depth", 0.5)),
                        height=float(fi.get("height", 0.5)),
                    ),
                ))

            poly = meta.get("floor_polygon") or meta.get("polygon")
            room = Room(
                width=room_w, length=room_l, height=room_h,
                furniture=furniture, floor_polygon=poly,
            )
            scene_obj = Scene(room=room)

        gltf_exp = GLTFExporter()
        glb_bytes = gltf_exp.export(scene_obj)

        if fmt == "glb":
            return Response(glb_bytes, mimetype="model/gltf-binary",
                            headers={"Content-Disposition": f"attachment; filename={scene_id}.glb"})

        if fmt == "obj":
            import trimesh
            glb_scene = trimesh.load(io.BytesIO(glb_bytes), file_type="glb", force="scene")
            combined = trimesh.util.concatenate(
                [g for g in glb_scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
            )
            obj_data = combined.export(file_type="obj")
            if isinstance(obj_data, str):
                obj_data = obj_data.encode("utf-8")
            return Response(obj_data, mimetype="text/plain",
                            headers={"Content-Disposition": f"attachment; filename={scene_id}.obj"})

        return jsonify({"error": f"Unknown format: {fmt}"}), 400

    except Exception as e:
        logger.exception("Export failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/scenes/<scene_id>/building-furniture", methods=["PUT"])
def update_building_furniture(scene_id):
    """Save furniture placement for a specific building within an outdoor scene."""
    scene_dir = OUTPUTS_DIR / scene_id
    if not scene_dir.is_dir():
        return jsonify({"error": "Scene not found"}), 404

    meta_path = scene_dir / "metadata.json"
    if not meta_path.exists():
        return jsonify({"error": "No metadata found"}), 404

    data = request.json or {}
    building_id = data.get("building_id")
    furniture = data.get("furniture", [])

    if not building_id:
        return jsonify({"error": "building_id required"}), 400

    with open(meta_path) as f:
        meta = json.load(f)

    # Store furniture keyed by building_id
    if "building_furniture" not in meta:
        meta["building_furniture"] = {}
    meta["building_furniture"][building_id] = furniture

    with open(meta_path, "w") as f:
        json.dump(meta, f)

    return jsonify({"saved": building_id, "num_items": len(furniture)})


@app.route("/api/scenes/<scene_id>/load")
def load_scene_data(scene_id):
    """Load a previously generated scene from outputs directory."""
    scene_dir = OUTPUTS_DIR / scene_id
    if not scene_dir.is_dir():
        return jsonify({"error": "Scene not found"}), 404

    # Prefer metadata.json (preserves model_ids, full data)
    meta_path = scene_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        meta["files"] = [fn.name for fn in scene_dir.iterdir()]

        # Normalize outdoor metadata (OSM-generated scenes use different keys)
        if meta.get("scene_type") == "outdoor" and "type" not in meta:
            meta["type"] = "outdoor"
        if "dimensions" in meta and "width" not in meta:
            meta["width"] = meta["dimensions"].get("width_m", 200)
            meta["length"] = meta["dimensions"].get("length_m", 200)
        # Generate rectangular footprints from bounds when missing
        if meta.get("type") == "outdoor" and "buildings" in meta:
            for b in meta["buildings"]:
                if "footprint" not in b and "bounds" in b:
                    bn = b["bounds"]
                    x0 = bn.get("minx", 0)
                    y0 = bn.get("miny", 0)
                    x1 = bn.get("maxx", x0 + 10)
                    y1 = bn.get("maxy", y0 + 10)
                    b["footprint"] = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

        # Enrich furniture with 3D-FUTURE metadata if available
        catalog = _get_catalog()
        if catalog and "furniture" in meta:
            for item in meta["furniture"]:
                mid = item.get("model_id")
                if mid and mid in catalog.models:
                    cm = catalog.models[mid]
                    item.setdefault("super_category", cm.get("super-category"))
                    item.setdefault("style", cm.get("style"))
                    item.setdefault("theme", cm.get("theme"))
                    item.setdefault("model_material", cm.get("material"))
        return jsonify(meta)

    xml_path = scene_dir / "scene.xml"
    if not xml_path.exists():
        return jsonify({"error": "No scene.xml found"}), 404

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extract room dimensions from floor scale
    room_width, room_length, height = 5.0, 4.0, 2.7
    floor_shape = root.find(".//shape[@id='floor']")
    if floor_shape is not None:
        scale = floor_shape.find(".//scale")
        if scale is not None:
            room_width = float(scale.get('x', '2.5')) * 2
            room_length = float(scale.get('y', '2.0')) * 2

    # Extract height from first wall shape
    for shape in root.findall(".//shape[@type='rectangle']"):
        sid = shape.get('id', '')
        if sid.startswith('wall_'):
            wall_scale = shape.find(".//scale")
            if wall_scale is not None:
                height = float(wall_scale.get('y', '1.35')) * 2
            break

    # Extract furniture from obj and cube shapes
    furniture = []
    for shape in root.findall(".//shape"):
        shape_type = shape.get('type')
        sid = shape.get('id', '')

        # Skip non-furniture shapes
        if sid in ('floor', 'ceiling', 'ground') or sid.startswith('wall_'):
            continue
        if shape_type not in ('obj', 'cube'):
            continue
        if sid.startswith('building_'):
            continue

        category = sid.split('_')[0] if '_' in sid else 'unknown'

        # Position from translate
        translate = shape.find(".//translate[last()]")
        x = float(translate.get('x', '0')) if translate is not None else 0
        y = float(translate.get('y', '0')) if translate is not None else 0

        # Theta from Z-axis rotate (degrees)
        theta = 0
        for rotate in shape.findall(".//rotate"):
            if rotate.get('z') == '1':
                theta = float(rotate.get('angle', '0'))
                break

        # Material from ref
        ref = shape.find(".//ref")
        material = ref.get('id', 'wood_mat').replace('_mat', '') if ref is not None else 'wood'

        # Dimensions
        dims = GENERIC_FURNITURE_DIMS.get(category, BoundingBox(width=0.5, depth=0.5, height=0.5))
        if shape_type == 'cube':
            cube_scale = shape.find(".//scale")
            if cube_scale is not None:
                dims = BoundingBox(
                    width=float(cube_scale.get('x', '0.25')) * 2,
                    depth=float(cube_scale.get('y', '0.25')) * 2,
                    height=float(cube_scale.get('z', '0.25')) * 2,
                )

        furniture.append({
            "id": sid,
            "category": category,
            "x": x, "y": y,
            "theta": theta,
            "width": dims.width,
            "depth": dims.depth,
            "height": dims.height,
            "material": material,
        })

    # Determine scene type
    scene_type = "indoor"
    if root.find(".//shape[@id='ground']") is not None:
        scene_type = "outdoor"

    files = [f.name for f in scene_dir.iterdir()]

    return jsonify({
        "type": scene_type,
        "width": room_width,
        "length": room_length,
        "height": height,
        "furniture": furniture,
        "files": files,
    })


@app.route("/api/coverage/thz", methods=["POST"])
def thz_coverage():
    """Compute theoretical THz coverage (fast, no Sionna needed)."""
    data = request.json or {}
    room_width = float(data.get("room_width", 8.0))
    room_length = float(data.get("room_length", 6.0))
    frequency = float(data.get("frequency", 300e9))
    tx_x = float(data.get("tx_x", room_width / 2))
    tx_y = float(data.get("tx_y", room_length / 2))
    tx_z = float(data.get("tx_z", 2.5))
    resolution = float(data.get("resolution", 0.2))
    array_elements = int(data.get("array_elements", 256))

    # Build furniture items for blockage calculation
    furniture_items = []
    for f in data.get("furniture", []):
        furniture_items.append(FurnitureItem(
            id=f.get("id", "furn"),
            category=f.get("category", "unknown"),
            model_id="generic",
            model_path="",
            position=Position(
                x=float(f["x"]), y=float(f["y"]),
                theta=math.radians(float(f.get("theta", 0))),
            ),
            dimensions=BoundingBox(
                width=float(f.get("width", 0.5)),
                depth=float(f.get("depth", 0.5)),
                height=float(f.get("height", 0.5)),
            ),
        ))

    config = THzConfig(
        frequency=frequency,
        tx_position=(tx_x, tx_y, tx_z),
        resolution=resolution,
        array_elements=array_elements,
    )

    coverage = compute_thz_coverage(room_width, room_length, furniture_items, config)

    return jsonify({
        "coverage": coverage.tolist(),
        "width": room_width,
        "length": room_length,
        "min_dbm": float(np.min(coverage)),
        "max_dbm": float(np.max(coverage)),
        "mean_dbm": float(np.mean(coverage)),
    })


@app.route("/api/ber/compute", methods=["POST"])
def ber_compute():
    """Compute BER vs SNR curves for LDPC and Polar codes."""
    data = request.json or {}
    snr_min = float(data.get("snr_min", -2))
    snr_max = float(data.get("snr_max", 10))
    num_points = int(data.get("num_points", 25))

    snr_db = np.linspace(snr_min, snr_max, num_points)
    snr_linear = 10 ** (snr_db / 10)

    # Theoretical BPSK BER
    from scipy.special import erfc
    ber_uncoded = 0.5 * erfc(np.sqrt(snr_linear))

    # Approximate coded BER (coding gain ~5-7 dB)
    ldpc_gain = 10 ** (6.0 / 10)  # 6 dB coding gain
    polar_gain = 10 ** (5.0 / 10)  # 5 dB coding gain

    ber_ldpc = 0.5 * erfc(np.sqrt(snr_linear * ldpc_gain))
    ber_polar = 0.5 * erfc(np.sqrt(snr_linear * polar_gain))

    # Clamp to reasonable floor
    ber_ldpc = np.maximum(ber_ldpc, 1e-7)
    ber_polar = np.maximum(ber_polar, 1e-7)
    ber_uncoded = np.maximum(ber_uncoded, 1e-7)

    return jsonify({
        "snr_db": snr_db.tolist(),
        "ber_uncoded": ber_uncoded.tolist(),
        "ber_ldpc": ber_ldpc.tolist(),
        "ber_polar": ber_polar.tolist(),
    })


@app.route("/api/ofdm/grid", methods=["POST"])
def ofdm_grid():
    """Generate OFDM resource grid visualization data."""
    data = request.json or {}
    num_subcarriers = int(data.get("num_subcarriers", 72))
    num_symbols = int(data.get("num_symbols", 14))
    pilot_spacing = int(data.get("pilot_spacing", 6))

    rng = np.random.default_rng(42)
    grid = rng.uniform(0.3, 1.0, (num_subcarriers, num_symbols))

    # Mark pilots
    pilots = []
    for sc in range(0, num_subcarriers, pilot_spacing):
        for sym in [0, 7]:  # Pilot positions at symbol 0 and 7
            if sym < num_symbols:
                grid[sc, sym] = 1.0
                pilots.append([sc, sym])

    return jsonify({
        "grid": grid.tolist(),
        "pilots": pilots,
        "num_subcarriers": num_subcarriers,
        "num_symbols": num_symbols,
    })


@app.route("/api/coverage/thz/multi-z", methods=["POST"])
def thz_coverage_multi_z():
    """Compute THz coverage at multiple Z heights (SSE progress)."""
    data = request.json or {}
    job_id = _create_job()

    def work():
        try:
            room_width = float(data.get("room_width", 8.0))
            room_length = float(data.get("room_length", 6.0))
            frequency = float(data.get("frequency") or 300e9)
            tx_x = float(data.get("tx_x", room_width / 2))
            tx_y = float(data.get("tx_y", room_length / 2))
            tx_z = float(data.get("tx_z", 2.5))
            resolution = float(data.get("resolution", 0.2))
            # Support both legacy array_elements and new antenna_rows x antenna_cols
            ant_rows = int(data.get("antenna_rows", 0))
            ant_cols = int(data.get("antenna_cols", 0))
            if ant_rows > 0 and ant_cols > 0:
                array_elements = ant_rows * ant_cols
            else:
                array_elements = int(data.get("array_elements", 256))
            z_min = float(data.get("z_min", 0.5))
            z_max = float(data.get("z_max", 2.5))
            z_steps = int(data.get("z_steps", 10))

            furniture_items = []
            for f in data.get("furniture", []):
                furniture_items.append(FurnitureItem(
                    id=f.get("id", "furn"),
                    category=f.get("category", "unknown"),
                    model_id="generic",
                    model_path="",
                    position=Position(
                        x=float(f["x"]), y=float(f["y"]),
                        theta=math.radians(float(f.get("theta", 0))),
                    ),
                    dimensions=BoundingBox(
                        width=float(f.get("width", 0.5)),
                        depth=float(f.get("depth", 0.5)),
                        height=float(f.get("height", 0.5)),
                    ),
                ))

            # Convert building footprints to AABB obstacles for coverage.
            # Same list also carries INTERIOR walls for multi-room indoor
            # scenes (see block below), so ray-AABB blockage picks up wall
            # attenuation the same way it picks up outdoor buildings.
            building_obstacles = []
            for b in data.get("buildings", []):
                fp = b.get("footprint", [])
                bh = float(b.get("height", 12))
                mat = b.get("material", "concrete")
                if len(fp) >= 3:
                    xs = [p[0] for p in fp]
                    ys = [p[1] for p in fp]
                    # Material-dependent penetration loss
                    loss = {"concrete": 25, "brick": 18, "glass": 8, "metal": 40}.get(mat, 25)
                    building_obstacles.append({
                        "xmin": min(xs), "xmax": max(xs),
                        "ymin": min(ys), "ymax": max(ys),
                        "zmin": 0.0, "zmax": bh,
                        "loss_db": loss,
                    })

            # ── Multi-room INTERIOR WALLS ────────────────────────────
            # For imported scenes with a scene_state.json (generated by
            # scripts/generate_apartment.py), pack
            # each interior partition wall between two rooms as an AABB
            # obstacle so ray-blockage picks up wall attenuation.
            # Fixed properties:
            #   thickness  0.15 m
            #   loss_db    12 (plasterboard interior wall @ 5 GHz)
            #   door       0.9 m gap at wall center — split into 2 segs
            scene_id_ctx = data.get("scene_id") or ""
            if scene_id_ctx:
                state_path = OUTPUTS_DIR / scene_id_ctx / "scene_state.json"
                if state_path.exists():
                    try:
                        state_obj = json.loads(state_path.read_text())
                        rooms_meta = state_obj.get("rooms", []) or []
                        room_h = float(state_obj.get("scene", {})
                                       .get("bounds", {}).get("height", 3.0))
                        THICK = 0.15
                        LOSS  = 12.0
                        DOOR  = 0.9
                        seen  = set()
                        for i, ra in enumerate(rooms_meta):
                            for rb in rooms_meta[i+1:]:
                                # Vertical shared edge (ra.x1 == rb.x0)
                                if abs(ra["x1"] - rb["x0"]) < 1e-6:
                                    y_lo = max(ra["y0"], rb["y0"])
                                    y_hi = min(ra["y1"], rb["y1"])
                                    if y_hi - y_lo <= 0.2: continue
                                    key = ("V", round(ra["x1"], 3),
                                           round(y_lo, 3), round(y_hi, 3))
                                    if key in seen: continue
                                    seen.add(key)
                                    x = ra["x1"]
                                    dc = (y_lo + y_hi) / 2
                                    for lo, hi in [(y_lo, dc - DOOR / 2),
                                                   (dc + DOOR / 2, y_hi)]:
                                        if hi - lo > 0.05:
                                            building_obstacles.append({
                                                "xmin": x - THICK / 2,
                                                "xmax": x + THICK / 2,
                                                "ymin": lo, "ymax": hi,
                                                "zmin": 0.0, "zmax": room_h,
                                                "loss_db": LOSS,
                                            })
                                # Horizontal shared edge (ra.y1 == rb.y0)
                                if abs(ra["y1"] - rb["y0"]) < 1e-6:
                                    x_lo = max(ra["x0"], rb["x0"])
                                    x_hi = min(ra["x1"], rb["x1"])
                                    if x_hi - x_lo <= 0.2: continue
                                    key = ("H", round(ra["y1"], 3),
                                           round(x_lo, 3), round(x_hi, 3))
                                    if key in seen: continue
                                    seen.add(key)
                                    y = ra["y1"]
                                    dc = (x_lo + x_hi) / 2
                                    for lo, hi in [(x_lo, dc - DOOR / 2),
                                                   (dc + DOOR / 2, x_hi)]:
                                        if hi - lo > 0.05:
                                            building_obstacles.append({
                                                "xmin": lo, "xmax": hi,
                                                "ymin": y - THICK / 2,
                                                "ymax": y + THICK / 2,
                                                "zmin": 0.0, "zmax": room_h,
                                                "loss_db": LOSS,
                                            })
                        n_walls = len(building_obstacles) - len(
                            [x for x in data.get("buildings", []) if x])
                        if n_walls > 0:
                            logger.info(
                                "coverage: added %d interior wall segments "
                                "from scene_state.json (%s)",
                                n_walls, scene_id_ctx)
                    except Exception as e:
                        logger.warning(
                            "coverage: could not parse interior walls from %s: %s",
                            state_path, e)

            # Extract antenna config for analytical directivity
            ant_pattern = data.get("antenna_pattern", "tr38901")
            ant_azimuth = float(data.get("antenna_azimuth", 0))
            ant_elevation = float(data.get("antenna_elevation", 0))

            z_values = np.linspace(z_min, z_max, z_steps).tolist()
            slices = []

            # ── Real Sionna RT path ──────────────────────────────────
            # If the scene has a proper Mitsuba XML at outputs/<id>/scene.xml,
            # run sionna.rt.RadioMapSolver over it. It natively models walls,
            # reflection, transmission, and (optionally) diffraction, so we
            # can skip the analytical directivity layer below.
            sionna_ok = False
            if scene_id_ctx:
                scene_xml = OUTPUTS_DIR / scene_id_ctx / "scene.xml"
                if scene_xml.exists():
                    try:
                        from src.wireless.sionna_rt_backend import (
                            compute_sionna_rt_coverage,
                        )
                        # Room center + size from state or defaults
                        map_cx = room_width / 2
                        map_cy = room_length / 2
                        _update_job(job_id, 10,
                                    f"Sionna RT: loading {scene_xml.name}...")
                        rt_slices = compute_sionna_rt_coverage(
                            scene_xml_path=str(scene_xml),
                            tx_position=(tx_x, tx_y, tx_z),
                            tx_power_dbm=20.0,
                            freq_hz=frequency,
                            z_heights=z_values,
                            cell_size=(max(resolution, 0.1),
                                       max(resolution, 0.1)),
                            ant_pattern=("iso" if ant_pattern == "iso"
                                         else "tr38901"),
                            ant_pol="V",
                            ant_rows=max(1, ant_rows),
                            ant_cols=max(1, ant_cols),
                            ant_azimuth_deg=ant_azimuth,
                            ant_elevation_deg=ant_elevation,
                            samples_per_tx=100_000,
                            max_depth=3,
                            specular=True,
                            refraction=True,
                            diffraction=False,
                            map_center_xy=(map_cx, map_cy),
                            map_size_xy=(room_width, room_length),
                            progress_cb=lambda step, total, msg: _update_job(
                                job_id, int(10 + 80 * step / max(total, 1)), msg),
                        )
                        # rt_slices already in dBm; keep only successful ones
                        good = [s for s in rt_slices
                                if s.get("grid") and len(s["grid"]) > 0]
                        if good:
                            slices = good
                            sionna_ok = True
                            logger.info(
                                "coverage: Sionna RT computed %d slices for %s",
                                len(good), scene_id_ctx)
                    except Exception as e:
                        logger.warning(
                            "coverage: Sionna RT failed (%s); "
                            "falling back to analytical model", e)

            # ── Analytical fallback (simplified LOS ray-AABB) ────────
            if not sionna_ok:
                for i, z_h in enumerate(z_values):
                    _update_job(
                        job_id,
                        int(10 + 80 * i / z_steps),
                        f"Computing Z={z_h:.1f}m ({i+1}/{z_steps})...",
                    )
                    config = THzConfig(
                        frequency=frequency,
                        tx_position=(tx_x, tx_y, tx_z),
                        resolution=resolution,
                        array_elements=array_elements,
                        rx_height=z_h,
                    )
                    coverage = compute_thz_coverage(
                        room_width, room_length, furniture_items, config,
                        obstacles=building_obstacles,
                    )

                    if ant_pattern != "iso":
                        nx, ny = coverage.shape[1], coverage.shape[0]
                        gx = np.linspace(0, room_width, nx)
                        gy = np.linspace(0, room_length, ny)
                        GX, GY = np.meshgrid(gx, gy)
                        dx = GX - tx_x
                        dy = GY - tx_y
                        dz = z_h - tx_z

                        cell_az = np.degrees(np.arctan2(dy, dx))
                        horiz_dist = np.sqrt(dx**2 + dy**2)
                        cell_el = np.degrees(np.arctan2(dz, np.maximum(horiz_dist, 0.01)))

                        az_off = cell_az - ant_azimuth
                        az_off = (az_off + 180) % 360 - 180
                        el_off = cell_el - ant_elevation

                        if ant_pattern == "dipole":
                            az_rad = np.radians(az_off)
                            dir_factor = np.cos(az_rad) ** 2
                            dir_db = 10 * np.log10(np.maximum(dir_factor, 1e-3)) + 2.15
                        else:
                            theta = np.sqrt(az_off**2 + el_off**2)
                            dir_db = np.maximum(-12 * (theta / 65) ** 2, -30)

                        coverage = coverage + dir_db

                    slices.append({"z": z_h, "grid": coverage.tolist()})

            # Aggregate stats from middle slice
            mid = slices[len(slices) // 2]["grid"]
            mid_flat = np.array(mid).flatten()

            result = {
                "slices": slices,
                "z_values": z_values,
                "room": {"width": room_width, "length": room_length},
                "min_dbm": float(np.min(mid_flat)),
                "max_dbm": float(np.max(mid_flat)),
                "mean_dbm": float(np.mean(mid_flat)),
                "engine": ("Sionna RT (RadioMapSolver)" if sionna_ok
                           else "THz Model + Directivity"),
            }
            _finish_job(job_id, result)
        except Exception as e:
            _fail_job(job_id, str(e))

    threading.Thread(target=work, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/rays/generate", methods=["POST"])
def generate_rays():
    """Generate synthetic ray paths from TX bouncing off walls."""
    data = request.json or {}
    tx_x = float(data.get("tx_x", 2.5))
    tx_y = float(data.get("tx_y", 2.0))
    tx_z = float(data.get("tx_z", 2.5))
    room_width = float(data.get("room_width", 5.0))
    room_length = float(data.get("room_length", 4.0))
    room_height = float(data.get("room_height", 3.0))
    num_rays = int(data.get("num_rays", 24))
    max_bounces = int(data.get("max_bounces", 3))

    # Parse furniture for ray-furniture intersection
    furniture_boxes = []
    for f in data.get("furniture", []):
        fx, fy = float(f.get("x", 0)), float(f.get("y", 0))
        fw, fd, fh = float(f.get("width", 0)), float(f.get("depth", 0)), float(f.get("height", 0))
        if fw > 0 and fd > 0 and fh > 0:
            furniture_boxes.append({
                "xmin": fx - fw / 2, "xmax": fx + fw / 2,
                "ymin": fy - fd / 2, "ymax": fy + fd / 2,
                "zmin": 0.0, "zmax": fh,
            })

    # Parse building footprints as AABB obstacles (for outdoor scenes)
    for b in data.get("buildings", []):
        fp = b.get("footprint", [])
        bh = float(b.get("height", 12))
        if len(fp) >= 3:
            xs = [p[0] for p in fp]
            ys = [p[1] for p in fp]
            furniture_boxes.append({
                "xmin": min(xs), "xmax": max(xs),
                "ymin": min(ys), "ymax": max(ys),
                "zmin": 0.0, "zmax": bh,
            })

    # Detect outdoor: buildings present or height > 10m suggests outdoor
    is_outdoor = len(data.get("buildings", [])) > 0 or room_height > 10

    rays = _generate_synthetic_rays(
        tx=(tx_x, tx_y, tx_z),
        room_dims=(room_width, room_length, room_height),
        num_rays=num_rays,
        max_bounces=max_bounces,
        furniture=furniture_boxes,
        is_outdoor=is_outdoor,
    )
    return jsonify({"rays": rays})


def _ray_aabb_intersect(pos, direction, box):
    """Ray-AABB intersection. Returns (t, normal) or (None, None).

    Uses the slab method to find ray entry point into an axis-aligned box.
    """
    tmin = 0.001
    tmax = 1e6
    normal = None

    for axis, (bmin_key, bmax_key) in enumerate(
        [("xmin", "xmax"), ("ymin", "ymax"), ("zmin", "zmax")]
    ):
        d = direction[axis]
        bmin = box[bmin_key]
        bmax = box[bmax_key]
        if abs(d) < 1e-9:
            # Ray parallel to slab
            if pos[axis] < bmin or pos[axis] > bmax:
                return None, None
            continue

        t1 = (bmin - pos[axis]) / d
        t2 = (bmax - pos[axis]) / d
        n1 = [0, 0, 0]
        n1[axis] = -1.0
        n2 = [0, 0, 0]
        n2[axis] = 1.0

        if t1 > t2:
            t1, t2 = t2, t1
            n1, n2 = n2, n1

        if t1 > tmin:
            tmin = t1
            normal = n1
        if t2 < tmax:
            tmax = t2

        if tmin > tmax:
            return None, None

    if tmin > 0.001 and normal is not None:
        return tmin, normal
    return None, None


def _generate_synthetic_rays(
    tx: tuple,
    room_dims: tuple,
    num_rays: int,
    max_bounces: int,
    furniture: list = None,
    is_outdoor: bool = False,
) -> list:
    """Cast rays from TX, reflect off walls and furniture, return path point lists.

    For outdoor scenes (is_outdoor=True), rays that hit the ceiling escape to
    the sky instead of reflecting — there is no ceiling in outdoor environments.
    """
    w, l, h = room_dims
    furniture = furniture or []
    rays = []
    # Max travel distance per bounce scales with scene diagonal
    max_travel = math.sqrt(w * w + l * l + h * h) * 1.5

    for i in range(num_rays):
        azimuth = 2 * math.pi * i / num_rays
        # Slight downward elevation so rays hit floor or walls
        elevation = -0.1 - 0.3 * (i % 3) / 3

        dx = math.cos(azimuth) * math.cos(elevation)
        dy = math.sin(azimuth) * math.cos(elevation)
        dz = math.sin(elevation)

        points = [[tx[0], tx[1], tx[2]]]
        px, py, pz = tx
        bounces = 0

        for _ in range(max_bounces):
            # Find nearest wall/floor/ceiling intersection
            best_t = float("inf")
            best_normal = None
            hit_ceiling = False

            # Check each bounding plane (room walls/floor/ceiling)
            planes = [
                (0, dx, 0.0),     # x = 0 (west wall)
                (0, dx, w),       # x = w (east wall)
                (1, dy, 0.0),     # y = 0 (south wall)
                (1, dy, l),       # y = l (north wall)
                (2, dz, 0.0),     # z = 0 (floor)
                (2, dz, h),       # z = h (ceiling)
            ]

            pos = [px, py, pz]
            direction = [dx, dy, dz]

            for axis, d_comp, boundary in planes:
                if abs(d_comp) < 1e-9:
                    continue
                t = (boundary - pos[axis]) / d_comp
                if t > 0.001 and t < best_t:
                    # Check if hit point is within room bounds
                    hit = [pos[j] + direction[j] * t for j in range(3)]
                    if (
                        -0.01 <= hit[0] <= w + 0.01
                        and -0.01 <= hit[1] <= l + 0.01
                        and -0.01 <= hit[2] <= h + 0.01
                    ):
                        best_t = t
                        normal = [0, 0, 0]
                        normal[axis] = -1.0 if d_comp > 0 else 1.0
                        best_normal = normal
                        # Track if this hit is the ceiling
                        hit_ceiling = (axis == 2 and boundary == h)

            # Check furniture AABB intersections
            for box in furniture:
                t_hit, n_hit = _ray_aabb_intersect(pos, direction, box)
                if t_hit is not None and t_hit < best_t:
                    best_t = t_hit
                    best_normal = n_hit
                    hit_ceiling = False  # Hit an object, not the ceiling

            if best_normal is None or best_t > max_travel:
                break

            # Move to hit point
            px += dx * best_t
            py += dy * best_t
            pz += dz * best_t
            # Clamp to room
            px = max(0, min(w, px))
            py = max(0, min(l, py))
            pz = max(0, min(h, pz))

            points.append([round(px, 4), round(py, 4), round(pz, 4)])
            bounces += 1

            # Outdoor: ray escapes to sky on ceiling hit — stop tracing
            if is_outdoor and hit_ceiling:
                break

            # Reflect direction: d = d - 2*(d.n)*n
            dot = dx * best_normal[0] + dy * best_normal[1] + dz * best_normal[2]
            dx -= 2 * dot * best_normal[0]
            dy -= 2 * dot * best_normal[1]
            dz -= 2 * dot * best_normal[2]

        if len(points) > 1:
            rays.append({"points": points, "bounces": bounces})

    return rays


@app.route("/api/catalog/categories")
def catalog_categories():
    """Return available 3D-FUTURE furniture categories."""
    catalog = _get_catalog()
    if catalog is None:
        # Return generic categories when catalog unavailable
        return jsonify({
            "available": False,
            "categories": [
                {"name": cat, "count": 0}
                for cat in GENERIC_FURNITURE_DIMS.keys()
            ],
        })

    categories = []
    for name, model_ids in sorted(catalog._category_index.items()):
        categories.append({"name": name, "count": len(model_ids)})

    return jsonify({"available": True, "categories": categories})


@app.route("/api/catalog/search")
def catalog_search():
    """Search furniture models by query string (category/style/material)."""
    q = (request.args.get("q", "") or "").strip().lower()
    limit = min(int(request.args.get("limit", 50)), 200)
    catalog = _get_catalog()

    if not q:
        return catalog_categories()

    if catalog is None:
        # Filter generic categories by query
        matches = [
            {"name": cat, "count": 0}
            for cat in GENERIC_FURNITURE_DIMS
            if q in cat
        ]
        return jsonify({"available": False, "categories": matches})

    # Search across categories, styles, materials
    results = []
    seen_cats = set()
    for name, model_ids in sorted(catalog._category_index.items()):
        if q in name:
            results.append({"name": name, "count": len(model_ids)})
            seen_cats.add(name)

    # Also search individual model metadata
    for model_id, model in catalog.models.items():
        cat = (model.get("category") or "").lower()
        style = (model.get("style") or "").lower()
        material = (model.get("material") or "").lower()
        theme = (model.get("theme") or "").lower()
        if cat in seen_cats:
            continue
        if q in style or q in material or q in theme:
            if cat and cat not in seen_cats:
                count = len(catalog._category_index.get(cat, []))
                results.append({"name": cat, "count": count})
                seen_cats.add(cat)
        if len(results) >= limit:
            break

    return jsonify({"available": True, "categories": results[:limit]})


@app.route("/api/catalog/model/<model_id>/mesh")
def catalog_model_mesh(model_id: str):
    """Return OBJ mesh vertices and faces for a 3D-FUTURE model.

    Applies Y-up to Z-up coordinate transform.
    Decimates to max_faces if model is very detailed.
    """
    catalog = _get_catalog()
    if catalog is None:
        return jsonify({"has_mesh": False, "error": "Catalog unavailable"}), 404

    try:
        import trimesh
        obj_path = catalog.get_model_path(model_id) / "raw_model.obj"
        if not obj_path.exists():
            return jsonify({"has_mesh": False, "error": "Model file not found"}), 404

        mesh = trimesh.load(obj_path, force="mesh")

        # Decimate if too many faces for browser rendering
        max_faces = int(request.args.get("max_faces", 500000))
        if len(mesh.faces) > max_faces:
            try:
                mesh = mesh.simplify_quadric_decimation(max_faces)
            except Exception:
                # fast_simplification not installed; subsample faces instead
                step = max(1, len(mesh.faces) // max_faces)
                keep = mesh.faces[::step]
                used_verts = np.unique(keep.flatten())
                remap = np.full(len(mesh.vertices), -1, dtype=int)
                remap[used_verts] = np.arange(len(used_verts))
                mesh = trimesh.Trimesh(
                    vertices=mesh.vertices[used_verts],
                    faces=remap[keep],
                )

        verts = mesh.vertices.copy()

        # Y-up to Z-up: rotate -90 deg around X
        # new_y = -old_z, new_z = old_y
        verts_transformed = np.column_stack([
            verts[:, 0],      # X stays
            -verts[:, 2],     # new Y = -old Z
            verts[:, 1],      # new Z = old Y (height)
        ])

        # Center on origin (XY), keep Z base at 0
        mins = verts_transformed.min(axis=0)
        maxs = verts_transformed.max(axis=0)
        center_x = (mins[0] + maxs[0]) / 2
        center_y = (mins[1] + maxs[1]) / 2
        verts_transformed[:, 0] -= center_x
        verts_transformed[:, 1] -= center_y
        verts_transformed[:, 2] -= mins[2]  # base at z=0

        faces = mesh.faces.tolist()

        return jsonify({
            "has_mesh": True,
            "model_id": model_id,
            "vertices": verts_transformed.tolist(),
            "faces": faces,
            "num_vertices": len(verts_transformed),
            "num_faces": len(faces),
        })
    except Exception as e:
        return jsonify({"has_mesh": False, "error": str(e)}), 500


@app.route("/api/catalog/model/<model_id>/image")
def catalog_model_image(model_id: str):
    """Return the preview image (image.jpg) for a 3D-FUTURE model."""
    catalog = _get_catalog()
    if catalog is None:
        return jsonify({"error": "Catalog unavailable"}), 404

    try:
        model_dir = catalog.get_model_path(model_id)
        image_path = model_dir / "image.jpg"
        if not image_path.exists():
            return jsonify({"error": "Image not found"}), 404
        resp = send_file(str(image_path), mimetype="image/jpeg")
        resp.headers["Cache-Control"] = "public, max-age=86400"
        return resp
    except Exception:
        return jsonify({"error": "Model not found"}), 404


@app.route("/api/catalog/category/<name>/models")
def catalog_category_models(name: str):
    """Return individual models within a furniture category."""
    catalog = _get_catalog()
    if catalog is None:
        return jsonify({"error": "Catalog unavailable"}), 404

    limit = min(int(request.args.get("limit", 30)), 100)
    q = (request.args.get("q", "") or "").strip().lower()

    model_ids = catalog._category_index.get(name, [])
    results = []
    for mid in model_ids:
        model = catalog.models.get(mid, {})
        style = (model.get("style") or "")
        material = (model.get("material") or "")
        theme = (model.get("theme") or "")

        if q:
            searchable = f"{style} {material} {theme} {name}".lower()
            if q not in searchable:
                continue

        results.append({
            "model_id": mid,
            "category": name,
            "style": style,
            "material": material,
            "theme": theme,
            "super_category": model.get("super-category", ""),
        })
        if len(results) >= limit:
            break

    return jsonify({"category": name, "models": results, "total": len(model_ids)})


@app.route("/api/catalog/model/<model_id>/glb")
def catalog_model_glb(model_id: str):
    """Return a binary GLB file for a 3D-FUTURE model with textures.

    Preserves materials and textures by exporting the Scene directly
    (not concatenating into a single mesh). Only decimates individual
    geometries that exceed the face budget.
    """
    catalog = _get_catalog()
    if catalog is None:
        return jsonify({"error": "Catalog unavailable"}), 404

    try:
        import trimesh

        model_dir = catalog.get_model_path(model_id)
        obj_path = model_dir / "raw_model.obj"
        if not obj_path.exists():
            return jsonify({"error": "Model file not found"}), 404

        scene_or_mesh = trimesh.load(str(obj_path))

        max_total_faces = 500000

        if isinstance(scene_or_mesh, trimesh.Scene):
            # Decimate individual geometries to stay under budget
            # while preserving per-geometry materials/textures
            total = sum(len(g.faces) for g in scene_or_mesh.geometry.values())
            if total > max_total_faces:
                ratio = max_total_faces / total
                for name, geom in scene_or_mesh.geometry.items():
                    target = max(100, int(len(geom.faces) * ratio))
                    if len(geom.faces) > target:
                        try:
                            scene_or_mesh.geometry[name] = geom.simplify_quadric_decimation(target)
                        except Exception:
                            pass  # keep original if simplification fails
            glb_data = scene_or_mesh.export(file_type="glb")
        else:
            mesh = scene_or_mesh
            if len(mesh.faces) > max_total_faces:
                try:
                    mesh = mesh.simplify_quadric_decimation(max_total_faces)
                except Exception:
                    pass  # keep original if simplification fails
            glb_data = mesh.export(file_type="glb")

        resp = Response(glb_data, mimetype="model/gltf-binary")
        resp.headers["Cache-Control"] = "public, max-age=86400"
        return resp

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────
# Routes: slow (SSE progress)
# ─────────────────────────────────────────────────────────

def _room_furniture_to_list(room: Room) -> list:
    """Convert Room furniture items to JSON-serializable list."""
    catalog = _get_catalog()
    result = []
    for f in room.furniture:
        mat = get_material_for_category(f.category)
        item = {
            "id": f.id,
            "category": f.category,
            "model_id": f.model_id,
            "x": f.position.x,
            "y": f.position.y,
            "theta": math.degrees(f.position.theta),
            "width": f.dimensions.width,
            "depth": f.dimensions.depth,
            "height": f.dimensions.height,
            "material": mat,
        }
        # Enrich with 3D-FUTURE metadata if available
        if catalog and f.model_id and f.model_id in catalog.models:
            meta = catalog.models[f.model_id]
            item["super_category"] = meta.get("super-category")
            item["style"] = meta.get("style")
            item["theme"] = meta.get("theme")
            item["model_material"] = meta.get("material")
        result.append(item)
    return result


@app.route("/api/scene/indoor/create", methods=["POST"])
def create_indoor_scene():
    """Create indoor room with furniture and export XML."""
    data = request.json or {}
    job_id = _create_job()

    def work():
        try:
            _update_job(job_id, 10, "Parsing parameters...")
            room_width = float(data.get("room_width", 5.0))
            room_length = float(data.get("room_length", 4.0))
            room_height = float(data.get("room_height", 3.0))
            floor_polygon = data.get("floor_polygon")

            door_data = data.get("door", {"wall": "south", "position": 2.0, "width": 0.9})
            doors = [Door(**door_data)]

            _update_job(job_id, 30, "Optimizing layout...")
            room = optimize_layout(
                room_width=room_width,
                room_length=room_length,
                doors=doors,
                floor_polygon=floor_polygon,
            )

            _update_job(job_id, 70, "Exporting XML...")
            scene = Scene(room=room)
            exporter = XMLExporter()
            name = _scene_name("Room", f"{room_width:.0f}x{room_length:.0f}", job_id)
            scene_dir = OUTPUTS_DIR / name
            scene_dir.mkdir(exist_ok=True)
            xml_path = scene_dir / "scene.xml"
            exporter.export_file(scene, xml_path)

            _update_job(job_id, 90, "Preparing result...")
            furniture_list = _room_furniture_to_list(room)
            result = {
                "scene_id": name,
                "xml_path": str(xml_path),
                "room": {
                    "width": room.width,
                    "length": room.length,
                    "height": room.height,
                    "is_rectangular": room.is_rectangular,
                    "num_furniture": len(room.furniture),
                    "polygon": room.floor_polygon,
                    "furniture": furniture_list,
                },
                "wall_segments": [
                    {"start": list(s.start), "end": list(s.end),
                     "cardinal": s.cardinal_direction}
                    for s in room.wall_segments
                ],
            }
            with open(scene_dir / "metadata.json", "w") as mf:
                json.dump({
                    "type": "indoor",
                    "name": name,
                    "width": room.width, "length": room.length,
                    "height": room.height,
                    "polygon": room.floor_polygon,
                    "furniture": furniture_list,
                }, mf)
            _finish_job(job_id, result)
        except Exception as e:
            _fail_job(job_id, str(e))

    threading.Thread(target=work, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/scene/indoor/create-with-furniture", methods=["POST"])
def create_indoor_with_furniture():
    """Create indoor room with specified furniture, optimize, and export XML."""
    data = request.json or {}
    job_id = _create_job()

    def work():
        try:
            _update_job(job_id, 10, "Parsing parameters...")
            room_width = float(data.get("room_width", 5.0))
            room_length = float(data.get("room_length", 4.0))
            floor_polygon = data.get("floor_polygon")

            door_data = data.get("door", {"wall": "south", "position": 2.0, "width": 0.9})
            doors = [Door(**door_data)]

            # Build furniture specs from category/quantity pairs
            # Use 3D-FUTURE catalog for real dimensions when available
            catalog = _get_catalog()
            specs = []

            # Re-include existing furniture (from building interior) so they
            # get re-optimized alongside new items
            existing_furniture = data.get("existing_furniture", [])
            for ef in existing_furniture:
                dims = BoundingBox(
                    width=float(ef.get("width", 0.5)),
                    depth=float(ef.get("depth", 0.5)),
                    height=float(ef.get("height", 0.5)),
                )
                specs.append(FurnitureSpec(
                    category=ef.get("category", "unknown"),
                    model_id=ef.get("model_id", str(uuid.uuid4())),
                    model_path=ef.get("model_path", ""),
                    dimensions=dims,
                ))

            # Add new furniture from AI action
            furniture_input = data.get("furniture", [])
            for item in furniture_input:
                cat = item["category"]
                qty = int(item.get("quantity", 1))
                for _ in range(qty):
                    model_id = str(uuid.uuid4())
                    model_path = ""
                    dims = GENERIC_FURNITURE_DIMS.get(
                        cat.lower(), BoundingBox(width=0.5, depth=0.5, height=0.5)
                    )
                    if catalog is not None:
                        try:
                            model = catalog.get_random_model(cat)
                            model_id = model["model_id"]
                            model_path = str(catalog.get_model_path(model_id))
                            dims = catalog.get_dimensions(model_id)
                        except (ValueError, Exception):
                            pass  # Fall back to generic dims
                    specs.append(FurnitureSpec(
                        category=cat,
                        model_id=model_id,
                        model_path=model_path,
                        dimensions=dims,
                    ))

            # Cap furniture count based on room area to prevent overcrowding
            room_area = room_width * room_length
            max_items = max(3, int(room_area / 3))
            if len(specs) > max_items:
                logger.info(f"Trimming furniture from {len(specs)} to {max_items} (room area {room_area:.0f} m²)")
                specs = specs[:max_items]

            _update_job(job_id, 30, f"Optimizing layout for {len(specs)} items...")
            room = optimize_layout(
                room_width=room_width,
                room_length=room_length,
                doors=doors,
                furniture_specs=specs,
                floor_polygon=floor_polygon,
            )

            _update_job(job_id, 70, "Exporting XML...")
            scene = Scene(room=room)
            exporter = XMLExporter()
            detail = data.get("name", f"{room_width:.0f}x{room_length:.0f}")
            name = _scene_name("Room", detail, job_id)
            scene_dir = OUTPUTS_DIR / name
            scene_dir.mkdir(exist_ok=True)
            xml_path = scene_dir / "scene.xml"
            exporter.export_file(scene, xml_path)

            _update_job(job_id, 90, "Preparing result...")
            furniture_list = _room_furniture_to_list(room)
            result = {
                "scene_id": name,
                "xml_path": str(xml_path),
                "room": {
                    "width": room.width,
                    "length": room.length,
                    "height": room.height,
                    "is_rectangular": room.is_rectangular,
                    "num_furniture": len(room.furniture),
                    "polygon": room.floor_polygon,
                    "furniture": furniture_list,
                },
                "wall_segments": [
                    {"start": list(s.start), "end": list(s.end),
                     "cardinal": s.cardinal_direction}
                    for s in room.wall_segments
                ],
            }

            # Save metadata for scene reload (preserves model_ids)
            with open(scene_dir / "metadata.json", "w") as mf:
                json.dump({
                    "type": "indoor",
                    "name": name,
                    "width": room.width, "length": room.length,
                    "height": room.height,
                    "polygon": room.floor_polygon,
                    "furniture": furniture_list,
                }, mf)

            _finish_job(job_id, result)
        except Exception as e:
            _fail_job(job_id, str(e))

    threading.Thread(target=work, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/scene/outdoor/create", methods=["POST"])
def create_outdoor_scene():
    """Create outdoor scene from bbox or predefined buildings."""
    data = request.json or {}
    job_id = _create_job()

    def work():
        try:
            _update_job(job_id, 10, "Building scene...")

            width = float(data.get("width", 200.0))
            length = float(data.get("length", 200.0))
            ground_material = data.get("ground_material", "wet_ground")

            buildings_data = data.get("buildings", [])
            buildings = []
            for i, b in enumerate(buildings_data):
                buildings.append(Building(
                    id=b.get("id", f"bldg_{i}"),
                    footprint=[tuple(p) for p in b["footprint"]],
                    height=float(b.get("height", 12.0)),
                    material=b.get("material", "concrete"),
                ))

            _update_job(job_id, 40, "Creating scene model...")
            outdoor = OutdoorScene(
                width=width,
                length=length,
                ground=GroundPlane(width=width, length=length, material=ground_material),
                buildings=buildings,
            )
            _outdoor_scenes[job_id] = outdoor

            _update_job(job_id, 60, "Exporting XML...")
            scene = Scene(outdoor=outdoor)
            exporter = XMLExporter(ground_material=ground_material)
            detail = data.get("name", f"{width:.0f}x{length:.0f}")
            name = _scene_name("Outdoor", detail, job_id)
            scene_dir = OUTPUTS_DIR / name
            scene_dir.mkdir(exist_ok=True)
            xml_path = scene_dir / "scene.xml"
            exporter.export_file(scene, xml_path)

            _update_job(job_id, 90, "Done")
            buildings_list = [
                {"id": b.id, "footprint": b.footprint, "height": b.height,
                 "material": b.material, "bounds": list(b.bounds),
                 "name": getattr(b, 'name', None)}
                for b in buildings
            ]
            result = {
                "scene_id": name,
                "xml_path": str(xml_path),
                "width": width,
                "length": length,
                "num_buildings": len(buildings),
                "buildings": buildings_list,
            }

            # Save metadata for scene reload
            with open(scene_dir / "metadata.json", "w") as mf:
                json.dump({
                    "type": "outdoor",
                    "name": name,
                    "width": width, "length": length,
                    "buildings": buildings_list,
                }, mf)

            _finish_job(job_id, result)
        except Exception as e:
            _fail_job(job_id, str(e))

    threading.Thread(target=work, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/building/<building_id>/enter", methods=["POST"])
def enter_building(building_id):
    """Building entry has been disabled."""
    return jsonify({"error": "Building entry is no longer available"}), 410


@app.route("/api/coverage/compute", methods=["POST"])
def compute_coverage():
    """Compute Sionna RT coverage map (requires sionna)."""
    data = request.json or {}
    job_id = _create_job()

    def work():
        try:
            xml_path = data.get("xml_path")
            if not xml_path:
                _fail_job(job_id, "xml_path required")
                return

            _update_job(job_id, 10, "Loading scene...")

            # Use THz theoretical model as fallback if Sionna not available
            room_width = float(data.get("room_width", 8.0))
            room_length = float(data.get("room_length", 6.0))
            frequency = float(data.get("frequency", 60e9))
            tx_x = float(data.get("tx_x", room_width / 2))
            tx_y = float(data.get("tx_y", room_length / 2))
            tx_z = float(data.get("tx_z", 2.5))

            _update_job(job_id, 30, "Computing coverage...")

            try:
                from src.wireless.scene import load_scene, configure_transmitter, SceneConfig
                from src.wireless.ray_tracing import compute_coverage_map, CoverageConfig

                config = SceneConfig(frequency=frequency)
                scene = load_scene(Path(xml_path), config)
                configure_transmitter(scene, [tx_x, tx_y, tx_z])

                _update_job(job_id, 50, "Ray tracing...")
                cov_config = CoverageConfig(
                    max_depth=int(data.get("max_depth", 5)),
                    cell_size=(
                        float(data.get("cell_size", 0.5)),
                        float(data.get("cell_size", 0.5)),
                    ),
                    samples_per_tx=int(data.get("samples", 100000)),
                    z_height=float(data.get("z_height", 1.5)),
                )
                radio_map = compute_coverage_map(scene, cov_config)

                _update_job(job_id, 80, "Extracting data...")
                coverage_data = radio_map.path_gain.numpy().squeeze()
                coverage_db = 10 * np.log10(np.abs(coverage_data) + 1e-12)

                result = {
                    "coverage": coverage_db.tolist(),
                    "min_dbm": float(np.min(coverage_db)),
                    "max_dbm": float(np.max(coverage_db)),
                    "mean_dbm": float(np.mean(coverage_db)),
                    "engine": "sionna_rt",
                }
            except ImportError:
                _update_job(job_id, 50, "Using theoretical model (Sionna not available)...")
                config = THzConfig(
                    frequency=frequency,
                    tx_position=(tx_x, tx_y, tx_z),
                    resolution=float(data.get("cell_size", 0.5)),
                )
                coverage = compute_thz_coverage(room_width, room_length, [], config)
                result = {
                    "coverage": coverage.tolist(),
                    "min_dbm": float(np.min(coverage)),
                    "max_dbm": float(np.max(coverage)),
                    "mean_dbm": float(np.mean(coverage)),
                    "engine": "theoretical_thz",
                }

            _finish_job(job_id, result)
        except Exception as e:
            _fail_job(job_id, str(e))

    threading.Thread(target=work, daemon=True).start()
    return jsonify({"job_id": job_id})


# ─────────────────────────────────────────────────────────
# Routes: OSM fetch
# ─────────────────────────────────────────────────────────

@app.route("/api/scene/osm/fetch", methods=["POST"])
def fetch_osm_scene():
    """Fetch outdoor scene from OpenStreetMap by bbox or location name."""
    data = request.json or {}
    job_id = _create_job()

    def work():
        try:
            _update_job(job_id, 5, "Importing OSM modules...")
            from src.osm.converter import fetch_outdoor_scene_bbox
            from src.osm.config import OSMConfig

            osm_config = OSMConfig(
                include_buildings=True,
                include_roads=data.get("include_roads", True),
                include_trees=False,  # Trees rarely available, skip for speed
            )

            # Support either bbox (north/south/east/west) or lat/lon + radius
            if all(k in data for k in ("north", "south", "east", "west")):
                north = float(data["north"])
                south = float(data["south"])
                east = float(data["east"])
                west = float(data["west"])
                location_name = data.get("name", "")
            elif "lat" in data and "lon" in data:
                lat = float(data["lat"])
                lon = float(data["lon"])
                radius_m = float(data.get("radius", 300))
                # Approximate degree offsets from meters
                dlat = radius_m / 111320.0
                dlon = radius_m / (111320.0 * math.cos(math.radians(lat)))
                north = lat + dlat
                south = lat - dlat
                east = lon + dlon
                west = lon - dlon
                location_name = data.get("name", f"{lat:.4f}_{lon:.4f}")
            else:
                _fail_job(job_id, "Provide north/south/east/west or lat/lon")
                return

            _update_job(job_id, 15, f"Fetching OSM data for {location_name or 'bbox'}...")
            outdoor = fetch_outdoor_scene_bbox(
                north=north, south=south, east=east, west=west,
                config=osm_config,
            )
            _outdoor_scenes[job_id] = outdoor

            _update_job(job_id, 60, "Exporting XML...")
            scene = Scene(outdoor=outdoor)
            exporter = XMLExporter(ground_material="wet_ground")
            name = _scene_name("OSM", location_name or f"{outdoor.width:.0f}x{outdoor.length:.0f}", job_id)
            scene_dir = OUTPUTS_DIR / name
            scene_dir.mkdir(exist_ok=True)
            xml_path = scene_dir / "scene.xml"
            exporter.export_file(scene, xml_path)

            _update_job(job_id, 90, "Done")
            buildings_list = [
                {"id": b.id, "footprint": b.footprint, "height": b.height,
                 "material": b.material, "bounds": list(b.bounds),
                 "name": getattr(b, 'name', None)}
                for b in outdoor.buildings
            ]
            roads_list = []
            if hasattr(outdoor, 'roads') and outdoor.roads:
                roads_list = [
                    {"id": r.id, "centerline": r.centerline, "width": r.width,
                     "material": r.material}
                    for r in outdoor.roads
                ]

            result = {
                "scene_id": name,
                "xml_path": str(xml_path),
                "width": outdoor.width,
                "length": outdoor.length,
                "num_buildings": len(outdoor.buildings),
                "buildings": buildings_list,
                "roads": roads_list,
            }

            with open(scene_dir / "metadata.json", "w") as mf:
                json.dump({
                    "type": "outdoor",
                    "source": "osm",
                    "name": name,
                    "location": location_name,
                    "width": outdoor.width, "length": outdoor.length,
                    "buildings": buildings_list,
                    "roads": roads_list,
                }, mf)

            _finish_job(job_id, result)
        except Exception as e:
            logger.exception("OSM fetch failed")
            _fail_job(job_id, str(e))

    threading.Thread(target=work, daemon=True).start()
    return jsonify({"job_id": job_id})


# ─────────────────────────────────────────────────────────
# Routes: AI Chat (Claude Haiku)
# ─────────────────────────────────────────────────────────

CHAT_SYSTEM_PROMPT = """You are a helpful assistant for configuring RF simulation scenes in the Sionna Dashboard.
You help users add furniture, change room dimensions, set up wireless simulations, manage saved scenes, and download real-world locations from OpenStreetMap.

When the user asks to add furniture, respond with a JSON action block in your text.
Available furniture categories: bed, desk, chair, sofa, wardrobe, nightstand, bookcase, cabinet, table, coffee_table, tv_stand, lamp, armchair, barstool, bookcase, dining_chair, dining_table, dressing_chair, dressing_table, shelf, stool, tv_stand, pendant_lamp, floor_lamp, ceiling_lamp.

CRITICAL RULE: The TOTAL number of furniture items MUST NOT exceed floor(room_area / 3).
For a 5x4m room (20 m²), the MAX is 6 items. For a 6x5m room (30 m²), the MAX is 10.
Count each item individually — "quantity": 2 means 2 items.

Typical setups by room area:
- 5x4m (20 m²) → 4-6 items: sofa, coffee_table, tv_stand, desk, chair, lamp
- 6x5m (30 m²) → 6-8 items
- 8x6m (48 m²) → 8-12 items
- >60 m² → up to 15 items
If the user is inside a building, pick furniture appropriate for that building type.
Do NOT ask the user how many — just pick a sensible amount. If unsure, use FEWER items rather than more.

IMPORTANT: When you want to perform an action, include EXACTLY ONE JSON block wrapped in ```action tags like this:

To create a NEW room with furniture (specify room_width and room_length):
```action
{"type": "add_furniture", "room_width": 6, "room_length": 5, "items": [{"category": "chair", "quantity": 2}, {"category": "desk", "quantity": 1}]}
```

To add furniture to the CURRENT room (omit room_width/room_length — uses existing room):
```action
{"type": "add_furniture", "items": [{"category": "sofa", "quantity": 1}, {"category": "lamp", "quantity": 2}]}
```

If the user is inside a building, ALWAYS omit room_width/room_length so furniture is placed in the current building room.
If the user asks to "create a 5x4 room with furniture", include room_width and room_length.

Other action types:

```action
{"type": "compute_coverage"}
```

```action
{"type": "load_scene", "scene_id": "indoor_abc123"}
```

```action
{"type": "delete_scene", "scene_id": "indoor_abc123"}
```

```action
{"type": "create_outdoor", "width": 200, "length": 200}
```

To configure the antenna array (pattern, polarization, size, orientation):
```action
{"type": "configure_antenna", "pattern": "dipole", "polarization": "V", "rows": 4, "cols": 4, "azimuth": 45, "elevation": -10}
```
All fields are optional — only include what the user wants to change.
Valid patterns: iso, dipole, tr38901. Valid polarizations: V, H, VH, cross.
Rows/cols: 1-64 each. Azimuth: 0-360. Elevation: -90 to 90.

When the user asks to fetch/download/get a location from OpenStreetMap (OSM), use the fetch_osm action.
You can specify a bounding box (north/south/east/west) or a center point (lat/lon) with radius in meters.
Always include a human-readable "name" field for the location.
Common examples:
```action
{"type": "fetch_osm", "lat": 37.9755, "lon": 23.7348, "radius": 300, "name": "Athens Syntagma"}
```

```action
{"type": "fetch_osm", "north": 40.7580, "south": 40.7530, "east": -73.9830, "west": -73.9880, "name": "Times Square NYC"}
```

Always respond conversationally AND include the action block when needed. Keep responses concise.
When the user asks about available scenes, list the ones from the scene context.
When the user asks to switch/load/open a scene, use the load_scene action.
When the user asks to delete/remove a scene, use the delete_scene action.
When the user mentions OpenStreetMap, OSM, downloading a city/area/neighborhood, or fetching real-world buildings, use the fetch_osm action with appropriate coordinates."""


# /api/chat is now served by web/routes/chat.py (chat_bp blueprint).
# The blueprint calls the exchangetoken OpenAI-compat gateway directly
# with model=claude-sonnet-4-5, so no `claude` CLI is required.


# ─────────────────────────────────────────────────────────
# Agent blueprint registration
# ─────────────────────────────────────────────────────────
# The agent blueprint exposes /api/agent/step (synchronous) and
# /api/agent/file/<relpath> (serves files the agent produced). It is
# registered here instead of being part of routes/__init__.py because
# dashboard_app.py is a monolithic Flask app — registering the whole
# routes package would collide with inline @app.route handlers.
try:
    from routes.agent import agent_bp
    app.register_blueprint(agent_bp)
except Exception as _e:  # pragma: no cover
    logger.warning("agent blueprint not registered: %s", _e)

# Register the rest of the route blueprints (scenes / coverage / catalog /
# creation / rays / ase / segmentation). Each is best-effort: if a module
# fails to import or its route collides with an inline @app.route, we log
# and continue so the rest of the dashboard stays up.
_INLINE_ENDPOINTS = {rule.rule for rule in app.url_map.iter_rules()}
for _bp_path, _bp_attr in [
    ("routes.chat",         "chat_bp"),
    ("routes.scenes",       "scenes_bp"),
    ("routes.coverage",     "coverage_bp"),
    ("routes.catalog",      "catalog_bp"),
    ("routes.creation",     "creation_bp"),
    ("routes.rays",         "rays_bp"),
]:
    try:
        _mod = __import__(_bp_path, fromlist=[_bp_attr])
        _bp = getattr(_mod, _bp_attr)
        # Skip if any of the blueprint's routes already exist inline.
        _bp_rules = {r.rule for r in _bp.deferred_functions if hasattr(r, "rule")}
        app.register_blueprint(_bp)
        logger.info("registered blueprint: %s", _bp.name)
    except Exception as _e:
        logger.warning("%s not registered: %s", _bp_path, _e)


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sionna RF Dashboard")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    print(f"Starting dashboard on http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=args.debug, threaded=True)
