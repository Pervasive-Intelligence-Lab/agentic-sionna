"""Scene CRUD routes: list, load, delete, export, building-furniture."""

import json
import logging
import re
import sys
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from flask import Blueprint, Response, jsonify, request, send_file

from src.models.room import FurnitureItem, Position, Room
from src.models.outdoor import Building, GroundPlane, OutdoorScene
from src.models.scene import Scene
from src.optimizer.layout import BoundingBox

from routes.shared import OUTPUTS_DIR, GENERIC_FURNITURE_DIMS, _get_catalog, _resolve_model_path

logger = logging.getLogger(__name__)

scenes_bp = Blueprint("scenes", __name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SLUG_RE = re.compile(r"[^a-z0-9_-]+")


def _next_apartment_id() -> str:
    n = 1
    while (OUTPUTS_DIR / f"apartment-{n:02d}").exists():
        n += 1
    return f"apartment-{n:02d}"


_ITU_MATERIALS = {
    "brick", "ceiling_board", "chipboard", "concrete", "floorboard", "glass",
    "marble", "medium_dry_ground", "metal", "plasterboard", "plywood",
    "very_dry_ground", "wet_ground", "wood",
}
_MATERIAL_ALIASES = {"drywall": "plasterboard", "gypsum": "plasterboard",
                     "itu_metal": "metal", "steel": "metal", "stone": "marble"}
_SURFACE_SHAPE_IDS = {
    # surface name -> predicate on <shape id="...">
    "interior_wall": lambda sid: sid.startswith("iw_"),
    "wall":          lambda sid: sid.startswith("iw_") or sid.startswith("wall_"),
    "exterior_wall": lambda sid: sid.startswith("wall_"),
    "floor":         lambda sid: sid == "floor",
    "ceiling":       lambda sid: sid == "ceiling",
}


@scenes_bp.route("/api/scenes/<scene_id>/material", methods=["POST"])
def set_scene_material(scene_id):
    """Rewrite the ITU material of a surface class in a pre-baked
    scene.xml so the next Sionna RT run uses it.

    Body: {surface: interior_wall|wall|exterior_wall|floor|ceiling,
           material: <ITU name or alias>}
    """
    scene_dir = OUTPUTS_DIR / scene_id
    xml_path = scene_dir / "scene.xml"
    if not xml_path.exists():
        return jsonify({"error": "scene has no scene.xml"}), 404
    data = request.json or {}
    surface = str(data.get("surface") or "wall").lower()
    pred = _SURFACE_SHAPE_IDS.get(surface)
    if pred is None:
        return jsonify({"error": f"unknown surface {surface!r}"}), 400
    mat = str(data.get("material") or "").lower().strip()
    mat = _MATERIAL_ALIASES.get(mat, mat)
    if mat not in _ITU_MATERIALS:
        return jsonify({"error": f"unknown material {mat!r}; ITU names: "
                        f"{sorted(_ITU_MATERIALS)}"}), 400

    tree = ET.parse(xml_path)
    root = tree.getroot()
    bsdf_id = f"{mat}_mat"
    if root.find(f"./bsdf[@id='{bsdf_id}']") is None:
        b = ET.SubElement(root, "bsdf", type="itu-radio-material", id=bsdf_id)
        ET.SubElement(b, "string", name="type", value=mat)
        # Keep bsdfs ahead of shapes so <ref> resolves when Mitsuba parses.
        root.remove(b)
        first_shape = next((i for i, c in enumerate(list(root)) if c.tag == "shape"), len(root))
        root.insert(first_shape, b)
    changed = 0
    for shape in root.findall("./shape"):
        if pred(shape.get("id", "")):
            ref = shape.find("ref")
            if ref is not None and ref.get("id") != bsdf_id:
                ref.set("id", bsdf_id)
                changed += 1
    tree.write(xml_path, encoding="unicode", xml_declaration=True)
    logger.info("scene %s: %s -> %s (%d shapes)", scene_id, surface, mat, changed)
    return jsonify({"scene_id": scene_id, "surface": surface,
                    "material": mat, "shapes_changed": changed})


@scenes_bp.route("/api/scenes/apartment/generate", methods=["POST"])
def generate_apartment_scene():
    """Build a multi-room apartment (walls + interior partitions + 3D-FUTURE
    furniture) from a room list and register it as a loadable scene.

    Body: {rooms: [{id?, type, width, depth}], name?, height?, seed?}
      type ∈ living_room | kitchen | bedroom | bathroom | balcony
    """
    data = request.json or {}
    rooms = data.get("rooms") or []
    if not isinstance(rooms, list) or not rooms:
        return jsonify({"error": "rooms must be a non-empty list"}), 400
    if len(rooms) > 12:
        return jsonify({"error": "at most 12 rooms"}), 400

    name = str(data.get("name") or "").strip().lower().replace(" ", "-")
    scene_id = _SLUG_RE.sub("", name) if name else _next_apartment_id()
    if not scene_id:
        scene_id = _next_apartment_id()
    # A repeated "build me an apartment" must not fail on a name clash:
    # suffix -2, -3, ... unless the caller explicitly asks to overwrite.
    if (OUTPUTS_DIR / scene_id).exists() and not data.get("overwrite"):
        base, n = scene_id, 2
        while (OUTPUTS_DIR / f"{base}-{n}").exists():
            n += 1
        scene_id = f"{base}-{n}"
    out_dir = OUTPUTS_DIR / scene_id

    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    try:
        from scripts.generate_apartment import generate_apartment
        meta = generate_apartment(
            rooms, out_dir,
            height=float(data.get("height") or 3.0),
            seed=int(data.get("seed") or 42),
            log=logger.info,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("apartment generation failed")
        return jsonify({"error": f"generation failed: {e}"}), 500

    return jsonify({
        "scene_id": scene_id,
        "width": meta["width"], "length": meta["length"], "height": meta["height"],
        "rooms": meta["rooms"],
        "n_furniture": len(meta["furniture"]),
    })


@scenes_bp.route("/api/scenes/<scene_id>/file/<path:filename>")
def scene_file(scene_id: str, filename: str):
    """Serve a static asset (viewer.html, scene.glb, coverage_heatmap.png, ...)
    from an outputs/<scene_id>/ folder. Used to open imported scenes'
    standalone Three.js viewers or expose their GLB / coverage assets.

    Path traversal is blocked — filename must resolve inside the scene dir.
    """
    scene_dir = OUTPUTS_DIR / scene_id
    if not scene_dir.is_dir():
        return jsonify({"error": "scene not found"}), 404
    # Resolve target and confirm it stays inside scene_dir (blocks ../ escapes)
    target = (scene_dir / filename).resolve()
    try:
        target.relative_to(scene_dir.resolve())
    except ValueError:
        return jsonify({"error": "invalid path"}), 400
    if not target.exists():
        return jsonify({"error": f"{filename} not in scene"}), 404
    return send_file(target)


@scenes_bp.route("/api/scenes/list")
def list_scenes():
    scenes = []
    for p in sorted(OUTPUTS_DIR.iterdir()):
        if p.is_dir():
            entry = {"id": p.name, "files": [f.name for f in p.iterdir()]}
            meta_path = p / "metadata.json"
            if meta_path.exists():
                try:
                    with open(meta_path) as mf:
                        meta = json.load(mf)
                    entry["name"] = meta.get("name", p.name)
                    entry["type"] = meta.get("type", "unknown")
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning("Failed to read metadata for scene %s: %s", p.name, e)
            scenes.append(entry)
    return jsonify(scenes)


@scenes_bp.route("/api/scenes/<scene_id>/delete", methods=["DELETE"])
def delete_scene(scene_id):
    """Delete a scene from the outputs directory."""
    import shutil
    scene_dir = OUTPUTS_DIR / scene_id
    if not scene_dir.is_dir():
        return jsonify({"error": "Scene not found"}), 404
    shutil.rmtree(scene_dir)
    return jsonify({"deleted": scene_id})


@scenes_bp.route("/api/scenes/<scene_id>/export", methods=["POST"])
def export_scene(scene_id):
    """Export a scene in the requested format (glb, obj, xml)."""
    import io
    scene_dir = OUTPUTS_DIR / scene_id
    if not scene_dir.is_dir():
        return jsonify({"error": "Scene not found"}), 404

    fmt = (request.json or {}).get("format", "xml")

    # XML -- serve existing scene.xml directly
    if fmt == "xml":
        xml_path = scene_dir / "scene.xml"
        if not xml_path.exists():
            return jsonify({"error": "No scene.xml found"}), 404
        return send_file(str(xml_path), mimetype="application/xml",
                         as_attachment=True, download_name=f"{scene_id}.xml")

    # GLB / OBJ -- reconstruct from metadata
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
            scene_obj = _reconstruct_outdoor_scene(meta)
        else:
            scene_obj = _reconstruct_indoor_scene(meta, catalog)

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
        logger.exception("Export failed for scene %s", scene_id)
        return jsonify({"error": str(e)}), 500


def _reconstruct_outdoor_scene(meta: dict) -> Scene:
    """Reconstruct an OutdoorScene from metadata.json."""
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
    return Scene(outdoor=outdoor)


def _reconstruct_indoor_scene(meta: dict, catalog) -> Scene:
    """Reconstruct an indoor Room from metadata.json."""
    furniture = []
    for fi in meta.get("furniture", []):
        mid = fi.get("model_id", "")
        mpath = fi.get("model_path", "")
        mfile = fi.get("model_file", "")
        if not mpath and not mfile and mid:
            mpath, mfile = _resolve_model_path(mid, catalog)
        furniture.append(FurnitureItem(
            id=fi.get("id", str(uuid.uuid4())),
            category=fi.get("category", "unknown"),
            model_id=mid or "generic",
            model_path=mpath,
            model_file=mfile,
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
        width=float(meta.get("width", 5)),
        length=float(meta.get("length", 4)),
        height=float(meta.get("height", 2.7)),
        furniture=furniture,
        floor_polygon=poly,
    )
    return Scene(room=room)


@scenes_bp.route("/api/scenes/<scene_id>/furniture", methods=["PUT"])
def update_furniture(scene_id):
    """Save updated furniture positions/rotations for an indoor scene."""
    scene_dir = OUTPUTS_DIR / scene_id
    if not scene_dir.is_dir():
        return jsonify({"error": "Scene not found"}), 404

    meta_path = scene_dir / "metadata.json"
    if not meta_path.exists():
        return jsonify({"error": "No metadata found"}), 404

    data = request.json or {}
    furniture = data.get("furniture", [])

    with open(meta_path) as f:
        meta = json.load(f)

    meta["furniture"] = furniture

    with open(meta_path, "w") as f:
        json.dump(meta, f)

    return jsonify({"saved": len(furniture)})


@scenes_bp.route("/api/scenes/<scene_id>/building-furniture", methods=["PUT"])
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

    if "building_furniture" not in meta:
        meta["building_furniture"] = {}
    meta["building_furniture"][building_id] = furniture

    with open(meta_path, "w") as f:
        json.dump(meta, f)

    return jsonify({"saved": building_id, "num_items": len(furniture)})


@scenes_bp.route("/api/scenes/<scene_id>/load")
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

    # Fallback: parse scene.xml directly
    xml_path = scene_dir / "scene.xml"
    if not xml_path.exists():
        return jsonify({"error": "No scene.xml found"}), 404

    return jsonify(_parse_scene_xml(xml_path, scene_dir))


def _parse_scene_xml(xml_path: Path, scene_dir: Path) -> dict:
    """Parse a Mitsuba scene.xml to extract room dimensions and furniture.

    Used as fallback when metadata.json is not available.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extract room dimensions from floor scale.
    # Mitsuba rectangle <scale> is half-extent, so multiply by 2 for full size.
    room_width, room_length, height = 5.0, 4.0, 2.7
    floor_shape = root.find(".//shape[@id='floor']")
    if floor_shape is not None:
        scale = floor_shape.find(".//scale")
        if scale is not None:
            room_width = float(scale.get('x', '2.5')) * 2
            room_length = float(scale.get('y', '2.0')) * 2

    # Extract height from first wall shape (also half-extent)
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

    return {
        "type": scene_type,
        "width": room_width,
        "length": room_length,
        "height": height,
        "furniture": furniture,
        "files": files,
    }
