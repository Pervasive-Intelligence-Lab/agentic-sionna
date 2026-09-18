"""Shared state, helpers, and constants for dashboard routes.

Provides:
- Job management (create, update, finish, fail, cancel, cleanup)
- Lazy-loaded furniture catalog with thread-safe initialization
- Common data conversion helpers for furniture and buildings
- Scene directory and metadata helpers to reduce duplication in routes
"""

import json
import logging
import math
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.models.room import FurnitureItem, Position
from src.optimizer.layout import BoundingBox

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# Environment setup
# ─────────────────────────────────────────────────────────

def _load_api_key():
    """Load ANTHROPIC_API_KEY from .env or ~/.bashrc if not in environment."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    # Try .env file in project root
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip("'\"")
                return
    # Try ~/.bashrc
    bashrc = Path.home() / ".bashrc"
    if bashrc.exists():
        for line in bashrc.read_text().splitlines():
            if "ANTHROPIC_API_KEY=" in line and not line.strip().startswith("#"):
                val = line.split("ANTHROPIC_API_KEY=", 1)[1].strip().strip("'\"")
                os.environ["ANTHROPIC_API_KEY"] = val
                return


_load_api_key()


# ─────────────────────────────────────────────────────────
# Job management (SSE progress tracking)
# ─────────────────────────────────────────────────────────

# Jobs are cleaned up after this many seconds to prevent unbounded memory growth.
_JOB_TTL_SECONDS = 600  # 10 minutes

_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _create_job() -> str:
    """Create a new background job and return its ID.

    The job starts in 'running' status with 0% progress.
    Old completed/failed jobs are automatically cleaned up.
    """
    _cleanup_stale_jobs()
    job_id = str(uuid.uuid4())[:8]
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "running",
            "progress": 0,
            "message": "",
            "result": None,
            "cancelled": False,
            "created_at": time.time(),
        }
    return job_id


def _update_job(job_id: str, progress: int, message: str) -> None:
    """Update job progress (0-100) and status message."""
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]["progress"] = progress
            _jobs[job_id]["message"] = message


def _finish_job(job_id: str, result: Any) -> None:
    """Mark a job as successfully completed with its result data."""
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["progress"] = 100
            _jobs[job_id]["result"] = result


def _fail_job(job_id: str, error: str) -> None:
    """Mark a job as failed with an error message."""
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["message"] = error


def _cancel_job(job_id: str) -> bool:
    """Request cancellation of a running job.

    Returns True if the job was found and marked for cancellation.
    Background threads should check is_job_cancelled() periodically.
    """
    with _jobs_lock:
        if job_id in _jobs and _jobs[job_id]["status"] == "running":
            _jobs[job_id]["cancelled"] = True
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["message"] = "Cancelled by user"
            return True
    return False


def is_job_cancelled(job_id: str) -> bool:
    """Check if a job has been cancelled. Use in background threads."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        return job is not None and job.get("cancelled", False)


def _add_job_slice(job_id: str, slice_data: dict) -> None:
    """Append a computed slice to a running job's streaming buffer.

    Used by coverage computation to stream slices to the frontend
    as they are computed rather than waiting for all slices to finish.
    The SSE progress endpoint includes new slices in each poll.
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is not None:
            if "slices" not in job:
                job["slices"] = []
            job["slices"].append(slice_data)


def _get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Get a snapshot of the current job state (thread-safe).

    Returns a shallow copy. The 'slices' list (if present) is shared
    by reference — the SSE stream uses slice_sent_idx to track which
    slices have already been sent to the client.
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        return dict(job)


def _cleanup_stale_jobs() -> None:
    """Remove completed/failed jobs older than _JOB_TTL_SECONDS."""
    now = time.time()
    with _jobs_lock:
        stale = [
            jid for jid, job in _jobs.items()
            if job["status"] in ("done", "error")
            and now - job.get("created_at", 0) > _JOB_TTL_SECONDS
        ]
        for jid in stale:
            del _jobs[jid]


# ─────────────────────────────────────────────────────────
# Output directory
# ─────────────────────────────────────────────────────────

OUTPUTS_DIR = (Path(__file__).resolve().parent.parent / "outputs").resolve()
OUTPUTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────
# Furniture catalog (lazy-loaded, thread-safe)
# ─────────────────────────────────────────────────────────

_catalog = None
_catalog_error = None
_catalog_lock = threading.Lock()


def _get_catalog():
    """Lazy-init furniture catalog (ABO or 3D-FUTURE).

    Selection order controlled by FURNITURE_CATALOG env var:
    - "abo": Use ABO catalog only
    - "3dfuture": Use 3D-FUTURE catalog only
    - "auto" (default): Try ABO first, fall back to 3D-FUTURE

    Thread-safe: concurrent callers block until initialization completes.
    Returns None if no dataset is available.
    """
    global _catalog, _catalog_error

    # Fast path: already loaded (no lock needed for reads of immutable refs)
    if _catalog is not None:
        return _catalog
    if _catalog_error is not None:
        return None

    with _catalog_lock:
        # Double-check after acquiring lock
        if _catalog is not None:
            return _catalog
        if _catalog_error is not None:
            return None

        catalog_pref = os.environ.get("FURNITURE_CATALOG", "auto").lower()

        # Try ABO catalog
        if catalog_pref in ("abo", "auto"):
            try:
                from src.catalog.abo import ABOCatalog
                abo_path = Path("data/abo-metadata")
                if abo_path.exists() and (abo_path / "3dmodels.csv.gz").exists():
                    _catalog = ABOCatalog(abo_path)
                    logger.info("Loaded ABO catalog with %d models", len(_catalog))
                    return _catalog
            except Exception as e:
                if catalog_pref == "abo":
                    _catalog_error = f"ABO catalog failed: {e}"
                    logger.warning("ABO catalog not available: %s", e)
                    return None
                logger.info("ABO catalog not available, trying 3D-FUTURE: %s", e)

        # Try 3D-FUTURE catalog
        if catalog_pref in ("3dfuture", "auto"):
            try:
                from src.catalog.furniture import FutureCatalog
                # FUTURE_DATASET_PATH env var overrides the default relative
                # path (lets sunlab point at ~/3D-FUTURE-model without
                # creating a symlink in the repo).
                dataset_path = Path(
                    os.environ.get("FUTURE_DATASET_PATH", "data/3D-FUTURE-model")
                ).expanduser()
                if not dataset_path.exists():
                    if catalog_pref == "3dfuture":
                        _catalog_error = "3D-FUTURE dataset not found"
                        logger.warning("3D-FUTURE catalog not available: %s", _catalog_error)
                        return None
                else:
                    _catalog = FutureCatalog(dataset_path)
                    logger.info("Loaded 3D-FUTURE catalog with %d models", len(_catalog))
                    return _catalog
            except Exception as e:
                _catalog_error = str(e)
                logger.warning("Failed to load 3D-FUTURE catalog: %s", e)
                return None

        if _catalog is None:
            _catalog_error = "No furniture catalog available"
            logger.warning(_catalog_error)
        return None


# ─────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────

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

# RF penetration loss (dB) per material — used by analytical coverage model
# to approximate signal attenuation through building walls.
# Source: ITU-R P.2040, Table 3 (simplified for common materials).
MATERIAL_PENETRATION_LOSS_DB = {
    "concrete": 25,
    "brick": 18,
    "glass": 8,
    "metal": 40,
}


# ─────────────────────────────────────────────────────────
# Scene naming and directory helpers
# ─────────────────────────────────────────────────────────

def _scene_name(prefix: str, detail: str, job_id: str) -> str:
    """Generate a descriptive scene directory name.

    Examples: Room_5x4_a1b2c3, Office_8x6_d4e5f6, Athens_Downtown_f7a8b9
    """
    clean = "".join(c if c.isalnum() or c in " _-." else "" for c in detail)
    clean = clean.strip().replace(" ", "_")[:30]
    tag = job_id[:6]
    if clean:
        return f"{prefix}_{clean}_{tag}"
    return f"{prefix}_{tag}"


def _save_scene(scene, name: str, metadata: dict) -> Path:
    """Export scene XML and save metadata.json to outputs/<name>/.

    Returns the scene directory path.
    """
    from src.exporters.xml import XMLExporter

    scene_dir = OUTPUTS_DIR / name
    scene_dir.mkdir(exist_ok=True)

    # Export XML
    ground_mat = metadata.get("ground_material")
    exporter = XMLExporter(ground_material=ground_mat) if ground_mat else XMLExporter()
    xml_path = scene_dir / "scene.xml"
    exporter.export_file(scene, xml_path)

    # Save metadata
    with open(scene_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    return scene_dir


def _resolve_model_path(model_id: str, catalog=None):
    """Resolve a furniture model's file path from the catalog.

    Returns (model_path, model_file) tuple. Both may be empty strings
    if the model can't be resolved.
    """
    model_path = ""
    model_file = ""
    if model_id and catalog is not None:
        try:
            if hasattr(catalog, 'get_model_file'):
                model_file = str(catalog.get_model_file(model_id))
            else:
                model_path = str(catalog.get_model_path(model_id))
        except (KeyError, ValueError):
            logger.debug("Could not resolve model path for %s", model_id)
    return model_path, model_file


# ─────────────────────────────────────────────────────────
# Data conversion helpers
# ─────────────────────────────────────────────────────────

def _parse_furniture_items(furniture_data: list) -> List[FurnitureItem]:
    """Convert JSON furniture dicts from request data into FurnitureItem objects.

    Used by coverage and ray-tracing endpoints that need furniture geometry
    for blockage/obstacle calculations but don't need catalog model resolution.
    """
    items = []
    for f in furniture_data:
        items.append(FurnitureItem(
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
    return items


def _buildings_to_obstacles(buildings_data: list) -> list:
    """Convert building footprint dicts to AABB obstacle dicts.

    Each building's polygon footprint is reduced to an axis-aligned bounding
    box with a material-dependent RF penetration loss (dB).
    """
    obstacles = []
    for b in buildings_data:
        fp = b.get("footprint", [])
        bh = float(b.get("height", 12))
        mat = b.get("material", "concrete")
        if len(fp) >= 3:
            xs = [p[0] for p in fp]
            ys = [p[1] for p in fp]
            loss = MATERIAL_PENETRATION_LOSS_DB.get(mat, 25)
            obstacles.append({
                "xmin": min(xs), "xmax": max(xs),
                "ymin": min(ys), "ymax": max(ys),
                "zmin": 0.0, "zmax": bh,
                "loss_db": loss,
            })
    return obstacles


def _room_furniture_to_list(room) -> list:
    """Convert Room furniture items to JSON-serializable list."""
    from src.exporters.xml import get_material_for_category

    catalog = _get_catalog()
    result = []
    for f in room.furniture:
        mat = get_material_for_category(f.category)
        item = {
            "id": f.id,
            "category": f.category,
            "model_id": f.model_id,
            "model_file": getattr(f, "model_file", "") or "",
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


def _build_indoor_result(room, scene_id: str, xml_path: Path) -> dict:
    """Build the standard indoor scene result dict used by SSE responses.

    Centralizes the result format to avoid duplication across
    create_indoor_scene, create_indoor_with_furniture, and resize_indoor_scene.
    """
    furniture_list = _room_furniture_to_list(room)
    return {
        "scene_id": scene_id,
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


def _build_indoor_metadata(room, name: str) -> dict:
    """Build metadata.json content for an indoor scene."""
    return {
        "type": "indoor",
        "name": name,
        "width": room.width,
        "length": room.length,
        "height": room.height,
        "polygon": room.floor_polygon,
        "furniture": _room_furniture_to_list(room),
    }
