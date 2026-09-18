#!/usr/bin/env python3
"""Generate a multi-room apartment scene for Sionna ray tracing.

Supports:
  - N rooms placed in a grid (2 columns by default)
  - Interior walls between adjacent rooms (with door openings)
  - Exterior walls (with window on each room's outermost wall)
  - Room-specific furniture (bedroom → bed+wardrobe, bathroom → toilet+sink,
    living → sofa+coffee_table, kitchen → cabinet+desk)
  - Outputs Sionna 3.0 XML + Three.js GLB + viewer.html + scene_state.json

Usage:
    python generate_apartment.py --out-dir web/outputs/apartment-01
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, tostring

# ──────────────────────────────────────────────────────────
# 3D-FUTURE catalog (set FUTURE_DATASET_PATH, or place it in ~/ or data/)
# ──────────────────────────────────────────────────────────

def find_dataset(name: str, env_var: str, marker: str) -> Path | None:
    for cand in [
        os.environ.get(env_var),
        Path.home() / name,
        Path("data") / name,
    ]:
        if cand and Path(cand).is_dir() and (Path(cand) / marker).exists():
            return Path(cand)
    return None


THREED_FUTURE = find_dataset("3D-FUTURE-model", "FUTURE_DATASET_PATH", "model_info.json")


def load_catalog() -> dict[str, list[dict]]:
    if not THREED_FUTURE:
        return {}
    info = json.loads((THREED_FUTURE / "model_info.json").read_text())
    cat: dict[str, list[dict]] = {}
    for m in info:
        c = (m.get("category") or "").lower()
        if not c:
            continue
        cat.setdefault(c, []).append(m)
    return cat


def find_model(catalog: dict, cats: list[str], max_w: float, max_d: float) -> tuple[dict | None, dict | None]:
    """Pick a random model whose bbox fits (max_w × max_d)."""
    for c in cats:
        matches = [m for name, models in catalog.items() if c.lower() in name for m in models]
        random.shuffle(matches)
        for m in matches:
            dims = _model_dims(m["model_id"])
            if dims and dims["width"] <= max_w and dims["depth"] <= max_d:
                return m, dims
    return None, None


def _model_dims(model_id: str) -> dict | None:
    """Read AABB of a 3D-FUTURE model."""
    if not THREED_FUTURE:
        return None
    obj_path = THREED_FUTURE / model_id / "raw_model.obj"
    if not obj_path.exists():
        return None
    try:
        # Just parse vertices to get bbox — cheap
        xs, ys, zs = [], [], []
        with obj_path.open() as f:
            for line in f:
                if line.startswith("v "):
                    _, x, y, z = line.split()[:4]
                    xs.append(float(x)); ys.append(float(y)); zs.append(float(z))
        if not xs:
            return None
        # 3D-FUTURE is Y-up: width = X range, height = Y range, depth = Z range
        return {"width": max(xs) - min(xs),
                "height": max(ys) - min(ys),
                "depth":  max(zs) - min(zs)}
    except Exception:
        return None


# ──────────────────────────────────────────────────────────
# Room + furniture specs
# ──────────────────────────────────────────────────────────

ROOM_TYPES = {
    "bedroom": {
        "min_size": (3.0, 2.8),
        "max_size": (5.0, 4.5),
        "furniture": [
            ("bed",         ["king-size bed", "double bed", "single bed"], (1.6, 2.0, 0.6)),
            ("nightstand",  ["nightstand"],                                (0.5, 0.4, 0.55)),
            ("wardrobe",    ["wardrobe"],                                  (1.0, 0.6, 2.0)),
            ("desk",        ["desk"],                                      (0.9, 0.5, 0.75)),
        ],
    },
    "living_room": {
        "min_size": (4.0, 3.5),
        "max_size": (6.0, 5.0),
        "furniture": [
            ("sofa",         ["three-seat / multi-seat sofa", "l-shaped sofa", "loveseat sofa"], (2.0, 0.9, 0.85)),
            ("coffee_table", ["coffee table", "tea table"],                                       (1.0, 0.6, 0.45)),
            ("tv_stand",     ["tv stand"],                                                        (1.6, 0.4, 0.5)),
            ("armchair",     ["armchair", "lounge chair / cafe chair / office chair"],            (0.7, 0.7, 0.85)),
            ("bookcase",     ["bookcase / jewelry armoire", "shelf"],                             (1.0, 0.35, 1.8)),
        ],
    },
    "kitchen": {
        "min_size": (2.5, 2.5),
        "max_size": (4.0, 3.5),
        "furniture": [
            ("cabinet",      ["cabinet", "drawer chest / corner cabinet", "wine cabinet"], (0.8, 0.5, 1.2)),
            ("dining_table", ["dining table"],                                              (1.2, 0.8, 0.75)),
            ("dining_chair", ["dining chair"],                                              (0.5, 0.5, 0.9)),
            ("dining_chair2",["dining chair"],                                              (0.5, 0.5, 0.9)),
        ],
    },
    "bathroom": {
        "min_size": (1.8, 1.8),
        "max_size": (3.0, 2.5),
        "furniture": [
            # 3D-FUTURE has no toilet — use small cabinet as sink stand
            ("sink_cabinet", ["cabinet", "nightstand"], (0.6, 0.4, 0.85)),
            ("shelf",        ["shelf"],                 (0.5, 0.3, 1.5)),
        ],
    },
    "balcony": {
        "min_size": (1.5, 2.0),
        "max_size": (2.5, 4.0),
        "furniture": [
            ("chair",        ["armchair", "lounge chair / cafe chair / office chair"], (0.6, 0.6, 0.85)),
            ("plant_stand",  ["side table", "corner/side table"],                       (0.4, 0.4, 0.7)),
        ],
    },
}


# ──────────────────────────────────────────────────────────
# Layout: place rooms in a compact grid
# ──────────────────────────────────────────────────────────

def compute_layout(room_specs: list[dict]) -> list[dict]:
    """Given a list of room specs, place them in a 2-column grid.
    Returns each room augmented with x0, y0, x1, y1 (footprint in world).
    """
    n = len(room_specs)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols

    # Per-column widths, per-row heights
    col_widths = [0.0] * ncols
    row_heights = [0.0] * nrows
    for i, r in enumerate(room_specs):
        row, col = divmod(i, ncols)
        col_widths[col] = max(col_widths[col], r["width"])
        row_heights[row] = max(row_heights[row], r["depth"])

    # Compute anchor per (row, col)
    x_starts = [0.0]
    for w in col_widths[:-1]:
        x_starts.append(x_starts[-1] + w)
    y_starts = [0.0]
    for h in row_heights[:-1]:
        y_starts.append(y_starts[-1] + h)

    for i, r in enumerate(room_specs):
        row, col = divmod(i, ncols)
        r["x0"] = x_starts[col]
        r["y0"] = y_starts[row]
        r["x1"] = r["x0"] + r["width"]
        r["y1"] = r["y0"] + r["depth"]
        r["cx"] = (r["x0"] + r["x1"]) / 2
        r["cy"] = (r["y0"] + r["y1"]) / 2

    return room_specs


def _pick(catalog, cats, fallback_dims, max_w, max_d):
    model, dims = find_model(catalog, cats, max_w=max_w, max_d=max_d)
    if model:
        return model["model_id"], dims["width"], dims["depth"], dims["height"]
    return None, fallback_dims[0], fallback_dims[1], fallback_dims[2]


def _make(room, name, mid, cx, cy, width, depth, height, theta, material):
    """Emit a furniture record (schema matches metadata.json/dashboard.js)."""
    return {
        "id":       f"{room['id']}_{name}",
        "category": name,
        "room":     room["id"],
        "model_id": mid,
        "x": float(cx), "y": float(cy),
        "width": float(width), "depth": float(depth), "height": float(height),
        "theta": float(theta),
        "material": material,
    }


def _footprint(width, depth, theta):
    """World-frame AABB half-extents given a model AABB and Y-axis rotation."""
    ct, st = abs(math.cos(math.radians(theta))), abs(math.sin(math.radians(theta)))
    return ct * width + st * depth, st * width + ct * depth


def _place_bedroom(room, catalog):
    """Bed centered on far (north) wall as headboard, nightstand next to it,
    wardrobe on west wall, desk on south wall."""
    W, D = room["width"], room["depth"]
    x0, y0 = room["x0"], room["y0"]
    m = 0.15
    placed = []

    # Bed — headboard against north wall (y=D)
    bed_mid, bw, bd, bh = _pick(
        catalog, ["king-size bed", "double bed", "single bed"],
        (1.6, 2.0, 0.6), max_w=W * 0.6, max_d=D * 0.55)
    bed_cx = x0 + W / 2
    bed_cy = y0 + D - bd / 2 - m
    placed.append(_make(room, "bed", bed_mid, bed_cx, bed_cy, bw, bd, bh, 0, "wood"))

    # Nightstand — right of bed, same wall
    ns_mid, nw, nd, nh = _pick(
        catalog, ["nightstand"], (0.5, 0.45, 0.55), max_w=0.7, max_d=0.6)
    ns_cx = bed_cx + bw / 2 + nw / 2 + 0.1
    ns_cy = y0 + D - nd / 2 - m
    if ns_cx + nw / 2 <= x0 + W - m:
        placed.append(_make(room, "nightstand", ns_mid, ns_cx, ns_cy, nw, nd, nh, 0, "wood"))

    # Wardrobe — against west wall (rotated 90°, doors face east)
    if W >= 3.0:
        wm_id, ww, wdp, wh = _pick(
            catalog, ["wardrobe"], (1.0, 0.6, 2.0),
            max_w=min(1.6, D * 0.5), max_d=0.7)
        # rotated 90°: model X (ww) becomes world Y span, model Z (wdp) → world X
        wx = x0 + wdp / 2 + m
        wy = y0 + D / 2
        # ensure no clash with bed (bed spans x from bed_cx-bw/2 to bed_cx+bw/2)
        if wx + wdp / 2 + 0.1 < bed_cx - bw / 2:
            placed.append(_make(room, "wardrobe", wm_id, wx, wy, wdp, ww, wh, 90, "wood"))

    # Desk — against south wall (y=0)
    d_mid, dw, dd, dh = _pick(
        catalog, ["desk"], (0.9, 0.5, 0.75),
        max_w=W * 0.45, max_d=0.6)
    dcx = x0 + W - dw / 2 - m
    dcy = y0 + dd / 2 + m
    placed.append(_make(room, "desk", d_mid, dcx, dcy, dw, dd, dh, 180, "wood"))

    return placed


def _place_living_room(room, catalog):
    """Sofa on south wall facing north; coffee table in front; TV stand on
    north wall facing south; armchair SW corner; bookcase east wall."""
    W, D = room["width"], room["depth"]
    x0, y0 = room["x0"], room["y0"]
    m = 0.2
    placed = []

    # Sofa — south wall
    s_mid, sw, sd, sh = _pick(
        catalog,
        ["three-seat / multi-seat sofa", "l-shaped sofa", "loveseat sofa"],
        (2.0, 0.9, 0.85), max_w=W * 0.6, max_d=D * 0.35)
    scx = x0 + W / 2
    scy = y0 + sd / 2 + m
    placed.append(_make(room, "sofa", s_mid, scx, scy, sw, sd, sh, 0, "fabric"))

    # Coffee table — in front of sofa
    ct_mid, cw, cd, ch = _pick(
        catalog, ["coffee table", "tea table"],
        (1.0, 0.6, 0.45), max_w=W * 0.35, max_d=D * 0.25)
    ccx = x0 + W / 2
    ccy = scy + sd / 2 + 0.6 + cd / 2  # 0.6 m leg-room
    if ccy + cd / 2 < y0 + D - 1.5:  # keep space for TV opposite
        placed.append(_make(room, "coffee_table", ct_mid, ccx, ccy, cw, cd, ch, 0, "wood"))

    # TV stand — north wall facing south
    tv_mid, tw, td, th = _pick(
        catalog, ["tv stand"],
        (1.6, 0.4, 0.5), max_w=W * 0.55, max_d=0.5)
    tvcx = x0 + W / 2
    tvcy = y0 + D - td / 2 - m
    placed.append(_make(room, "tv_stand", tv_mid, tvcx, tvcy, tw, td, th, 180, "wood"))

    # Armchair — SW corner (angled toward coffee table)
    ac_mid, aw, ad, ah = _pick(
        catalog, ["armchair", "lounge chair / cafe chair / office chair"],
        (0.7, 0.7, 0.85), max_w=0.9, max_d=0.9)
    acx = x0 + aw / 2 + m + 0.1
    acy = y0 + D / 2 - 0.2
    # skip if it would clash with sofa footprint
    if acx + aw / 2 + 0.05 < scx - sw / 2:
        placed.append(_make(room, "armchair", ac_mid, acx, acy, aw, ad, ah, 45, "fabric"))

    # Bookcase — east wall (rotated 90°)
    bc_mid, bw2, bd2, bh2 = _pick(
        catalog, ["bookcase / jewelry armoire", "shelf"],
        (1.0, 0.35, 1.8), max_w=min(1.2, D * 0.3), max_d=0.5)
    bcx = x0 + W - bd2 / 2 - m
    bcy = y0 + D / 2
    # skip if TV zone or sofa zone would clash
    if bcx - bd2 / 2 > tvcx + tw / 2 + 0.15:
        placed.append(_make(room, "bookcase", bc_mid, bcx, bcy, bd2, bw2, bh2, 90, "wood"))

    return placed


def _place_kitchen(room, catalog):
    """Cabinet run along north wall; dining table + 2 chairs in center."""
    W, D = room["width"], room["depth"]
    x0, y0 = room["x0"], room["y0"]
    m = 0.15
    placed = []

    # Cabinet — north wall
    c_mid, cw, cd, ch = _pick(
        catalog, ["cabinet", "drawer chest / corner cabinet", "wine cabinet"],
        (0.8, 0.5, 1.2), max_w=W * 0.55, max_d=0.6)
    ccx = x0 + cw / 2 + m
    ccy = y0 + D - cd / 2 - m
    placed.append(_make(room, "cabinet", c_mid, ccx, ccy, cw, cd, ch, 180, "wood"))

    # Dining table — centered (a bit off cabinet)
    dt_mid, dw, dd, dh = _pick(
        catalog, ["dining table"],
        (1.2, 0.8, 0.75), max_w=W * 0.5, max_d=D * 0.4)
    dtcx = x0 + W / 2
    dtcy = y0 + D / 2 - 0.2  # slight offset south from center
    placed.append(_make(room, "dining_table", dt_mid, dtcx, dtcy, dw, dd, dh, 0, "wood"))

    # Two dining chairs — north and south of table
    ch_mid, cw2, cd2, ch2 = _pick(
        catalog, ["dining chair"], (0.5, 0.5, 0.9), max_w=0.6, max_d=0.6)
    # South chair (facing north into table)
    sy = dtcy - dd / 2 - cd2 / 2 - 0.1
    if sy > y0 + cd2 / 2 + m:
        placed.append(_make(room, "dining_chair_S", ch_mid,
                            dtcx, sy, cw2, cd2, ch2, 0, "wood"))
    # North chair (facing south)
    ny = dtcy + dd / 2 + cd2 / 2 + 0.1
    if ny < ccy - cd / 2 - m:
        placed.append(_make(room, "dining_chair_N", ch_mid,
                            dtcx, ny, cw2, cd2, ch2, 180, "wood"))
    return placed


def _place_bathroom(room, catalog):
    """Sink cabinet on one long wall, shelf on the other."""
    W, D = room["width"], room["depth"]
    x0, y0 = room["x0"], room["y0"]
    m = 0.1
    placed = []
    # Sink cabinet — north wall
    s_mid, sw, sd, sh = _pick(
        catalog, ["cabinet", "nightstand"],
        (0.6, 0.4, 0.85), max_w=min(0.8, W * 0.7), max_d=0.5)
    scx = x0 + W / 2
    scy = y0 + D - sd / 2 - m
    placed.append(_make(room, "sink_cabinet", s_mid, scx, scy, sw, sd, sh, 180, "wood"))
    # Shelf — south wall corner
    sh_mid, ssw, ssd, ssh = _pick(
        catalog, ["shelf"], (0.5, 0.3, 1.5), max_w=0.7, max_d=0.4)
    shcx = x0 + ssw / 2 + m
    shcy = y0 + ssd / 2 + m
    placed.append(_make(room, "shelf", sh_mid, shcx, shcy, ssw, ssd, ssh, 0, "wood"))
    return placed


def _place_balcony(room, catalog):
    W, D = room["width"], room["depth"]
    x0, y0 = room["x0"], room["y0"]
    m = 0.15
    placed = []
    ch_mid, cw, cd, ch = _pick(
        catalog, ["armchair", "lounge chair / cafe chair / office chair"],
        (0.6, 0.6, 0.85), max_w=0.8, max_d=0.8)
    placed.append(_make(room, "chair", ch_mid,
                        x0 + cw / 2 + m, y0 + D / 2, cw, cd, ch, 90, "wood"))
    p_mid, pw, pd, ph = _pick(
        catalog, ["side table", "corner/side table"],
        (0.4, 0.4, 0.7), max_w=0.5, max_d=0.5)
    placed.append(_make(room, "plant_stand", p_mid,
                        x0 + W - pw / 2 - m, y0 + D / 2, pw, pd, ph, 270, "wood"))
    return placed


_ROOM_PLACERS = {
    "bedroom":     _place_bedroom,
    "living_room": _place_living_room,
    "kitchen":     _place_kitchen,
    "bathroom":    _place_bathroom,
    "balcony":     _place_balcony,
}


DOOR_W = 0.9          # must match build_xml / build_glb door_w
DOOR_SWING = 1.0      # keep-clear depth into the room in front of a door


def compute_doors(rooms: list[dict]) -> None:
    """Attach room["doors"] = [{side, c}] using the same shared-edge rule
    as build_xml/build_glb (door centred on the shared span). side is the
    wall of *this* room the door sits on; c is the along-wall coordinate
    (y for E/W walls, x for N/S walls)."""
    for r in rooms:
        r["doors"] = []
    for i, a in enumerate(rooms):
        for b in rooms[i + 1:]:
            if abs(a["x1"] - b["x0"]) < 1e-6:
                y_lo, y_hi = max(a["y0"], b["y0"]), min(a["y1"], b["y1"])
                if y_hi - y_lo > 0.2:
                    c = (y_lo + y_hi) / 2
                    a["doors"].append({"side": "E", "c": c})
                    b["doors"].append({"side": "W", "c": c})
            if abs(a["y1"] - b["y0"]) < 1e-6:
                x_lo, x_hi = max(a["x0"], b["x0"]), min(a["x1"], b["x1"])
                if x_hi - x_lo > 0.2:
                    c = (x_lo + x_hi) / 2
                    a["doors"].append({"side": "N", "c": c})
                    b["doors"].append({"side": "S", "c": c})


def _door_zone(room, door):
    """Keep-clear rectangle (xmin, xmax, ymin, ymax) in front of a door."""
    half = DOOR_W / 2 + 0.15
    s, c = door["side"], door["c"]
    if s == "E":
        return room["x1"] - DOOR_SWING, room["x1"], c - half, c + half
    if s == "W":
        return room["x0"], room["x0"] + DOOR_SWING, c - half, c + half
    if s == "N":
        return c - half, c + half, room["y1"] - DOOR_SWING, room["y1"]
    return c - half, c + half, room["y0"], room["y0"] + DOOR_SWING


def _aabb(f):
    hw, hd = _footprint(f["width"], f["depth"], f["theta"])
    return f["x"] - hw / 2, f["x"] + hw / 2, f["y"] - hd / 2, f["y"] + hd / 2


def _overlaps(a, b, pad=0.0):
    return (a[0] < b[1] + pad and a[1] > b[0] - pad and
            a[2] < b[3] + pad and a[3] > b[2] - pad)


def clear_doorways(room: dict, placed: list[dict]) -> list[dict]:
    """Slide any piece sitting in front of a door along the wall until it
    is clear (and not on top of another piece); drop it if impossible."""
    zones = [_door_zone(room, d) for d in room.get("doors", [])]
    if not zones:
        return placed
    kept: list[dict] = []
    m = 0.1
    for f in placed:
        box = _aabb(f)
        hit = next((z for z in zones if _overlaps(box, z)), None)
        if hit is None:
            kept.append(f)
            continue
        hw, hd = _footprint(f["width"], f["depth"], f["theta"])
        # Slide along the wall axis: y for E/W doors, x for N/S doors.
        vertical = (hit[1] - hit[0]) <= DOOR_SWING + 1e-6 and (hit[3] - hit[2]) < room["depth"]
        cands = []
        if vertical:
            for y in (hit[3] + hd / 2 + m, hit[2] - hd / 2 - m):
                if room["y0"] + hd / 2 + m <= y <= room["y1"] - hd / 2 - m:
                    cands.append((f["x"], y))
        else:
            for x in (hit[1] + hw / 2 + m, hit[0] - hw / 2 - m):
                if room["x0"] + hw / 2 + m <= x <= room["x1"] - hw / 2 - m:
                    cands.append((x, f["y"]))
        moved = False
        for x, y in cands:
            g = dict(f, x=x, y=y)
            gb = _aabb(g)
            if any(_overlaps(gb, z) for z in zones):
                continue
            if any(_overlaps(gb, _aabb(k), pad=0.05) for k in kept):
                continue
            kept.append(g)
            moved = True
            break
        if not moved:
            print(f"  dropped {f['id']} (would block a doorway)")
    return kept


def place_furniture(room: dict, catalog: dict) -> list[dict]:
    """Room-type-specific realistic placement (bed against wall, sofa faces TV,
    dining chairs at table, etc.), then pushed clear of any doorway."""
    placer = _ROOM_PLACERS.get(room["type"])
    if not placer:
        return []
    return clear_doorways(room, placer(room, catalog))


# ──────────────────────────────────────────────────────────
# Sionna 3.0 XML builder — walls, ceilings, doors, windows
# ──────────────────────────────────────────────────────────

MATERIALS = [
    # (bsdf id used by <ref>, ITU material name Sionna 2.0 accepts)
    ("concrete_mat",     "concrete"),
    ("plasterboard_mat", "plasterboard"),
    ("glass_mat",        "glass"),
    ("wood_mat",         "wood"),
    ("metal_mat",        "metal"),
]


def _add_rect(parent, name, mat, sx, sy, tx, ty, tz, rx=0, ry=0, rz=0):
    shape = SubElement(parent, "shape", type="rectangle")
    shape.set("id", name)
    SubElement(shape, "ref", id=mat)
    t = SubElement(shape, "transform", name="to_world")
    if rx: SubElement(t, "rotate", x="1", angle=str(rx))
    if ry: SubElement(t, "rotate", y="1", angle=str(ry))
    if rz: SubElement(t, "rotate", z="1", angle=str(rz))
    SubElement(t, "scale", x=str(sx), y=str(sy), z="1")
    SubElement(t, "translate", x=str(tx), y=str(ty), z=str(tz))


def build_xml(rooms: list[dict], height: float, bounds_w: float, bounds_d: float) -> str:
    root = Element("scene", version="3.0.0")
    SubElement(root, "integrator", type="path")
    for mid, mname in MATERIALS:
        bsdf = SubElement(root, "bsdf", type="itu-radio-material", id=mid)
        s = SubElement(bsdf, "string")
        s.set("name", "type")
        s.set("value", mname)

    # Single unified floor + ceiling covering entire footprint
    _add_rect(root, "floor",   "concrete_mat",     bounds_w / 2, bounds_d / 2, bounds_w / 2, bounds_d / 2, 0)
    _add_rect(root, "ceiling", "plasterboard_mat", bounds_w / 2, bounds_d / 2, bounds_w / 2, bounds_d / 2, height, rx=180)

    # Exterior walls (world envelope)
    _add_rect(root, "wall_south", "concrete_mat", bounds_w / 2, height / 2, bounds_w / 2, 0,        height / 2, rx=90)
    _add_rect(root, "wall_north", "concrete_mat", bounds_w / 2, height / 2, bounds_w / 2, bounds_d, height / 2, rx=-90)
    _add_rect(root, "wall_east",  "concrete_mat", bounds_d / 2, height / 2, bounds_w,    bounds_d / 2, height / 2, rx=90, rz=90)
    _add_rect(root, "wall_west",  "concrete_mat", bounds_d / 2, height / 2, 0,           bounds_d / 2, height / 2, rx=90, rz=-90)

    # Interior partition walls between adjacent rooms (drop into rectangles).
    # Collect unique interior edges shared between two rooms.
    added = set()
    for i, a in enumerate(rooms):
        for b in rooms[i+1:]:
            # Vertical shared edge (a.x1 == b.x0)
            if abs(a["x1"] - b["x0"]) < 1e-6:
                y_lo, y_hi = max(a["y0"], b["y0"]), min(a["y1"], b["y1"])
                if y_hi - y_lo > 0.2:
                    edge = ("V", a["x1"], round(y_lo, 3), round(y_hi, 3))
                    if edge in added: continue
                    added.add(edge)
                    _add_partition_wall_vertical(root, a["x1"], y_lo, y_hi, height, name=f"iw_{i}_v")
            # Horizontal shared edge (a.y1 == b.y0)
            if abs(a["y1"] - b["y0"]) < 1e-6:
                x_lo, x_hi = max(a["x0"], b["x0"]), min(a["x1"], b["x1"])
                if x_hi - x_lo > 0.2:
                    edge = ("H", a["y1"], round(x_lo, 3), round(x_hi, 3))
                    if edge in added: continue
                    added.add(edge)
                    _add_partition_wall_horizontal(root, a["y1"], x_lo, x_hi, height, name=f"iw_{i}_h")

    # Room-specific radios (nothing — TX gets added by the user in the UI)
    return minidom.parseString(tostring(root)).toprettyxml(indent="  ")


def _add_partition_wall_vertical(root, x, y_lo, y_hi, height, name, door_w=0.9, door_h=2.1):
    """Partition wall at x=x, spanning y in [y_lo, y_hi], full ceiling height,
    with a door opening centered along the wall."""
    length = y_hi - y_lo
    door_c = (y_lo + y_hi) / 2
    # Left segment
    left_len = door_c - door_w / 2 - y_lo
    if left_len > 0.05:
        _add_rect(root, f"{name}_L", "plasterboard_mat",
                  left_len / 2, height / 2, x, y_lo + left_len / 2, height / 2, rx=90, rz=90)
    # Right segment
    right_len = y_hi - (door_c + door_w / 2)
    if right_len > 0.05:
        _add_rect(root, f"{name}_R", "plasterboard_mat",
                  right_len / 2, height / 2, x, y_hi - right_len / 2, height / 2, rx=90, rz=90)
    # Header above door
    header_h = height - door_h
    if header_h > 0.05:
        _add_rect(root, f"{name}_H", "plasterboard_mat",
                  door_w / 2, header_h / 2, x, door_c, door_h + header_h / 2, rx=90, rz=90)


def _add_partition_wall_horizontal(root, y, x_lo, x_hi, height, name, door_w=0.9, door_h=2.1):
    length = x_hi - x_lo
    door_c = (x_lo + x_hi) / 2
    left_len = door_c - door_w / 2 - x_lo
    if left_len > 0.05:
        _add_rect(root, f"{name}_L", "plasterboard_mat",
                  left_len / 2, height / 2, x_lo + left_len / 2, y, height / 2, rx=90)
    right_len = x_hi - (door_c + door_w / 2)
    if right_len > 0.05:
        _add_rect(root, f"{name}_R", "plasterboard_mat",
                  right_len / 2, height / 2, x_hi - right_len / 2, y, height / 2, rx=90)
    header_h = height - door_h
    if header_h > 0.05:
        _add_rect(root, f"{name}_H", "plasterboard_mat",
                  door_w / 2, header_h / 2, door_c, y, door_h + header_h / 2, rx=90)


# ──────────────────────────────────────────────────────────
# GLB export (Three.js viewport)
# ──────────────────────────────────────────────────────────

def build_glb(rooms: list[dict], furniture: list[dict], height: float,
              bounds_w: float, bounds_d: float, out_path: Path) -> None:
    import trimesh
    scene = trimesh.Scene()
    wall_thick = 0.15

    # Floor
    floor = trimesh.creation.box(extents=[bounds_w, 0.02, bounds_d])
    floor.visual.face_colors = [80, 80, 80, 200]
    floor.apply_translation([bounds_w / 2, 0, bounds_d / 2])
    scene.add_geometry(floor, node_name="floor")

    # Exterior walls (4)
    for name, w, d, cx, cz in [
        ("wall_south", bounds_w, wall_thick, bounds_w / 2, -wall_thick / 2),
        ("wall_north", bounds_w, wall_thick, bounds_w / 2, bounds_d + wall_thick / 2),
        ("wall_west",  wall_thick, bounds_d + 2 * wall_thick, -wall_thick / 2, bounds_d / 2),
        ("wall_east",  wall_thick, bounds_d + 2 * wall_thick, bounds_w + wall_thick / 2, bounds_d / 2),
    ]:
        w_mesh = trimesh.creation.box(extents=[w, height, d])
        w_mesh.visual.face_colors = [80, 80, 100, 60]
        w_mesh.apply_translation([cx, height / 2, cz])
        scene.add_geometry(w_mesh, node_name=name)

    # Interior partitions (with door hole)
    added = set()
    for i, a in enumerate(rooms):
        for b in rooms[i+1:]:
            if abs(a["x1"] - b["x0"]) < 1e-6:
                y_lo, y_hi = max(a["y0"], b["y0"]), min(a["y1"], b["y1"])
                if y_hi - y_lo > 0.2:
                    edge = ("V", a["x1"], round(y_lo, 3), round(y_hi, 3))
                    if edge in added: continue
                    added.add(edge)
                    _add_glb_partition_vertical(scene, a["x1"], y_lo, y_hi, height, wall_thick, f"iw_v_{i}")
            if abs(a["y1"] - b["y0"]) < 1e-6:
                x_lo, x_hi = max(a["x0"], b["x0"]), min(a["x1"], b["x1"])
                if x_hi - x_lo > 0.2:
                    edge = ("H", a["y1"], round(x_lo, 3), round(x_hi, 3))
                    if edge in added: continue
                    added.add(edge)
                    _add_glb_partition_horizontal(scene, a["y1"], x_lo, x_hi, height, wall_thick, f"iw_h_{i}")

    # Furniture — load real 3D-FUTURE meshes when available
    for fi in furniture:
        theta = float(fi.get("theta", 0))
        added_mesh = None
        if fi.get("model_id") and THREED_FUTURE:
            obj_path = THREED_FUTURE / fi["model_id"] / "raw_model.obj"
            if obj_path.exists():
                try:
                    m = trimesh.load(str(obj_path), force='mesh')
                    b_min, b_max = m.bounds
                    center = (b_min + b_max) / 2
                    m.apply_translation(-center)
                    m.apply_translation([0, (b_max[1] - b_min[1]) / 2, 0])
                    if theta != 0:
                        rot = trimesh.transformations.rotation_matrix(
                            math.radians(theta), [0, 1, 0])
                        m.apply_transform(rot)
                    m.apply_translation([fi["x"], 0, fi["y"]])
                    scene.add_geometry(m, node_name=fi["id"])
                    added_mesh = m
                except Exception as e:
                    print(f"  GLB: mesh load failed for {fi['id']}: {e}")
        if added_mesh is None:
            box = trimesh.creation.box(
                extents=[fi["width"], fi["height"], fi["depth"]])
            box.visual.face_colors = [110, 130, 160, 255]
            if theta != 0:
                rot = trimesh.transformations.rotation_matrix(
                    math.radians(theta), [0, 1, 0])
                box.apply_transform(rot)
            box.apply_translation([fi["x"], fi["height"] / 2, fi["y"]])
            scene.add_geometry(box, node_name=fi["id"])

    scene.export(str(out_path), file_type="glb")


def _add_glb_partition_vertical(scene, x, y_lo, y_hi, height, thick, name, door_w=0.9, door_h=2.1):
    import trimesh
    length = y_hi - y_lo
    door_c = (y_lo + y_hi) / 2
    left_len = door_c - door_w / 2 - y_lo
    if left_len > 0.05:
        w = trimesh.creation.box(extents=[thick, height, left_len])
        w.visual.face_colors = [90, 90, 110, 55]
        w.apply_translation([x, height / 2, y_lo + left_len / 2])
        scene.add_geometry(w, node_name=f"{name}_L")
    right_len = y_hi - (door_c + door_w / 2)
    if right_len > 0.05:
        w = trimesh.creation.box(extents=[thick, height, right_len])
        w.visual.face_colors = [90, 90, 110, 55]
        w.apply_translation([x, height / 2, y_hi - right_len / 2])
        scene.add_geometry(w, node_name=f"{name}_R")
    header_h = height - door_h
    if header_h > 0.05:
        w = trimesh.creation.box(extents=[thick, header_h, door_w])
        w.visual.face_colors = [90, 90, 110, 55]
        w.apply_translation([x, door_h + header_h / 2, door_c])
        scene.add_geometry(w, node_name=f"{name}_H")


def _add_glb_partition_horizontal(scene, y, x_lo, x_hi, height, thick, name, door_w=0.9, door_h=2.1):
    import trimesh
    length = x_hi - x_lo
    door_c = (x_lo + x_hi) / 2
    left_len = door_c - door_w / 2 - x_lo
    if left_len > 0.05:
        w = trimesh.creation.box(extents=[left_len, height, thick])
        w.visual.face_colors = [90, 90, 110, 55]
        w.apply_translation([x_lo + left_len / 2, height / 2, y])
        scene.add_geometry(w, node_name=f"{name}_L")
    right_len = x_hi - (door_c + door_w / 2)
    if right_len > 0.05:
        w = trimesh.creation.box(extents=[right_len, height, thick])
        w.visual.face_colors = [90, 90, 110, 55]
        w.apply_translation([x_hi - right_len / 2, height / 2, y])
        scene.add_geometry(w, node_name=f"{name}_R")
    header_h = height - door_h
    if header_h > 0.05:
        w = trimesh.creation.box(extents=[door_w, header_h, thick])
        w.visual.face_colors = [90, 90, 110, 55]
        w.apply_translation([door_c, door_h + header_h / 2, y])
        scene.add_geometry(w, node_name=f"{name}_H")


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

PRESETS = {
    # Compact 1-bed (default). 8.0 x 7.5 m.
    "1br": [
        {"id": "living",   "type": "living_room", "width": 5.0, "depth": 4.0},
        {"id": "kitchen",  "type": "kitchen",     "width": 3.0, "depth": 4.0},
        {"id": "bedroom",  "type": "bedroom",     "width": 4.0, "depth": 3.5},
        {"id": "bathroom", "type": "bathroom",    "width": 2.0, "depth": 3.5},
    ],
    # Open-plan studio — one big living + a small bath. 8.0 x 5.0 m.
    "studio": [
        {"id": "living",   "type": "living_room", "width": 5.5, "depth": 5.0},
        {"id": "bathroom", "type": "bathroom",    "width": 2.5, "depth": 5.0},
    ],
    # 2-bedroom family. 10.5 x 10.5 m. Grid: [living,kitchen] /
    # [bedroom1,bedroom2] / [bathroom,balcony].
    "2br": [
        {"id": "living",   "type": "living_room", "width": 5.5, "depth": 4.5},
        {"id": "kitchen",  "type": "kitchen",     "width": 3.5, "depth": 4.5},
        {"id": "bedroom1", "type": "bedroom",     "width": 4.0, "depth": 3.5},
        {"id": "bedroom2", "type": "bedroom",     "width": 5.0, "depth": 3.5},
        {"id": "bathroom", "type": "bathroom",    "width": 4.0, "depth": 2.5},
        {"id": "balcony",  "type": "balcony",     "width": 5.0, "depth": 2.5},
    ],
    # Large 3-bedroom, ~12 x 11 m. Grid: [living,kitchen] /
    # [bedroom1,bedroom2] / [bedroom3,bathroom].
    "3br": [
        {"id": "living",   "type": "living_room", "width": 6.0, "depth": 5.0},
        {"id": "kitchen",  "type": "kitchen",     "width": 4.0, "depth": 5.0},
        {"id": "bedroom1", "type": "bedroom",     "width": 4.5, "depth": 3.5},
        {"id": "bedroom2", "type": "bedroom",     "width": 4.5, "depth": 3.5},
        {"id": "bedroom3", "type": "bedroom",     "width": 4.5, "depth": 3.0},
        {"id": "bathroom", "type": "bathroom",    "width": 4.5, "depth": 3.0},
    ],
}
DEFAULT_APARTMENT = PRESETS["1br"]


VALID_ROOM_TYPES = set(ROOM_TYPES.keys())


def generate_apartment(room_specs: list[dict], out_dir, height: float = 3.0,
                       seed: int = 42, log=print) -> dict:
    """Build a multi-room apartment from a list of
    {id, type, width, depth} specs and write scene.xml / scene.glb /
    scene_state.json / metadata.json into out_dir. Returns the metadata.

    Callable from the CLI (main) and from the dashboard's
    /api/scenes/apartment/generate endpoint."""
    random.seed(seed)
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    specs = []
    for i, r in enumerate(room_specs):
        rtype = str(r.get("type", "")).lower().replace(" ", "_")
        if rtype == "living":
            rtype = "living_room"
        if rtype not in VALID_ROOM_TYPES:
            raise ValueError(f"unknown room type {r.get('type')!r}; "
                             f"expected one of {sorted(VALID_ROOM_TYPES)}")
        specs.append({
            "id":    str(r.get("id") or f"{rtype}{i+1}"),
            "type":  rtype,
            "width": float(r.get("width", 4.0)),
            "depth": float(r.get("depth", r.get("length", 3.5))),
        })
    if not specs:
        raise ValueError("no rooms given")

    catalog = load_catalog()
    log(f"3D-FUTURE catalog: {sum(len(v) for v in catalog.values())} models in {len(catalog)} categories")

    rooms = compute_layout(specs)
    compute_doors(rooms)
    bounds_w = max(r["x1"] for r in rooms)
    bounds_d = max(r["y1"] for r in rooms)
    log(f"Apartment footprint: {bounds_w}m × {bounds_d}m × {height}m")
    for r in rooms:
        log(f"  {r['id']:10} ({r['type']:12}) at ({r['x0']:.1f},{r['y0']:.1f})–({r['x1']:.1f},{r['y1']:.1f})")

    all_furniture = []
    for r in rooms:
        placed = place_furniture(r, catalog)
        all_furniture.extend(placed)
        log(f"  {r['id']}: placed {len(placed)} pieces")

    xml = build_xml(rooms, height, bounds_w, bounds_d)
    (out / "scene.xml").write_text(xml)
    log(f"Wrote {out}/scene.xml ({len(xml)} bytes)")

    try:
        build_glb(rooms, all_furniture, height, bounds_w, bounds_d, out / "scene.glb")
        log(f"Wrote {out}/scene.glb")
    except Exception as e:
        log(f"GLB failed: {e}")

    rooms_out = [{k: v for k, v in r.items() if k not in ("cx", "cy")} for r in rooms]
    state = {
        "version": "2.0",
        "meta": {
            "created": datetime.now(timezone.utc).isoformat(),
            "name": f"Apartment ({len(rooms)} rooms)",
            "description": f"Multi-room apartment: {', '.join(r['type'] for r in rooms)}",
        },
        "scene": {
            "type": "indoor",
            "bounds": {"width": bounds_w, "depth": bounds_d, "height": height},
            "frequency_hz": 5e9,
        },
        "rooms": rooms_out,
        "furniture": all_furniture,
    }
    (out / "scene_state.json").write_text(json.dumps(state, indent=2))

    # metadata.json — schema consumed by /api/scenes/<id>/load, which is
    # what makes the furniture visible and draggable in the dashboard.
    metadata = {
        "type":      "indoor",
        "name":      out.name,
        "width":     bounds_w,
        "length":    bounds_d,
        "height":    height,
        "polygon":   None,
        "rooms":     rooms_out,
        "furniture": all_furniture,
    }
    (out / "metadata.json").write_text(json.dumps(metadata))
    log(f"Wrote {out}/metadata.json")
    return metadata


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="web/outputs/apartment-01",
                    help="output directory (default: web/outputs/apartment-01)")
    ap.add_argument("--preset", choices=list(PRESETS.keys()), default="1br",
                    help="floorplan preset (1br|studio|2br|3br)")
    ap.add_argument("--height", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    print(f"THREED_FUTURE = {THREED_FUTURE}")
    print(f"Preset: {args.preset}")
    generate_apartment([dict(r) for r in PRESETS[args.preset]],
                       args.out_dir, height=args.height, seed=args.seed)

    print(f"\n✓ Done. Scene id = {out.name}")


if __name__ == "__main__":
    main()
