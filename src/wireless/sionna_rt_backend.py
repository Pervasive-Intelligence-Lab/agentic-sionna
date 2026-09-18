"""Real Sionna RT `RadioMapSolver` backend for the dashboard.

Replaces the simplified `compute_thz_coverage` (LOS ray-AABB) when the
scene has a proper Mitsuba/Sionna XML at outputs/<scene_id>/scene.xml.
Walls, reflections, transmission through materials, and (optionally)
diffraction are all modelled by real ray tracing.

Public entry point:
    compute_sionna_rt_coverage(
        scene_xml_path, tx_position, tx_power_dbm, freq_hz,
        z_heights, cell_size=(0.25, 0.25),
        ant_pattern="iso", ant_pol="V", ant_rows=1, ant_cols=1,
        ant_azimuth_deg=0.0, ant_elevation_deg=0.0,
        samples_per_tx=100_000, max_depth=3,
        specular=True, refraction=True, diffraction=False,
        progress_cb=None,
    ) -> list[dict]

Returns a list of {"z": float, "grid": 2D list of dBm}, one entry per
requested z height. `progress_cb(step, total, msg)` (optional) is fired
after each z-slice completes so the SSE endpoint can push updates.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Optional


def compute_sionna_rt_coverage(
    scene_xml_path: str | Path,
    tx_position: tuple[float, float, float],
    tx_power_dbm: float,
    freq_hz: float,
    z_heights: list[float],
    cell_size: tuple[float, float] = (0.25, 0.25),
    ant_pattern: str = "iso",
    ant_pol: str = "V",
    ant_rows: int = 1,
    ant_cols: int = 1,
    ant_azimuth_deg: float = 0.0,
    ant_elevation_deg: float = 0.0,
    samples_per_tx: int = 100_000,
    max_depth: int = 3,
    specular: bool = True,
    refraction: bool = True,
    diffraction: bool = False,
    map_center_xy: Optional[tuple[float, float]] = None,
    map_size_xy: Optional[tuple[float, float]] = None,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> list[dict]:
    """Ray-traced coverage using Sionna 2.0 `RadioMapSolver`."""
    import sionna.rt as rt
    import numpy as np

    # 1. Load scene from XML (Mitsuba 3.0 / Sionna radio material bsdf)
    scene = rt.load_scene(str(scene_xml_path))
    scene.frequency = float(freq_hz)

    # 2. Build antenna arrays
    pol = ant_pol.upper()
    if pol == "VH" or pol == "CROSS":
        polarization = "cross"
    elif pol == "H":
        polarization = "H"
    else:
        polarization = "V"
    pattern = ant_pattern if ant_pattern in ("iso", "tr38901") else "iso"
    scene.tx_array = rt.PlanarArray(
        num_rows=max(1, int(ant_rows)),
        num_cols=max(1, int(ant_cols)),
        pattern=pattern,
        polarization=polarization,
    )
    scene.rx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, pattern="iso", polarization="V"
    )

    # 3. Add transmitter
    tx = rt.Transmitter(
        name="tx",
        position=list(tx_position),
        power_dbm=float(tx_power_dbm),
    )
    scene.add(tx)
    # Antenna orientation (yaw, pitch, roll) — radians. Yaw = azimuth.
    tx.orientation = [
        math.radians(float(ant_azimuth_deg)),
        math.radians(float(ant_elevation_deg)),
        0.0,
    ]

    # 4. Determine world-frame center + size for the radio-map plane.
    #    Caller supplies known room extents (from scene_state.json) when
    #    available; otherwise fall back to Mitsuba's bbox, then to a box
    #    around the tx.
    if map_center_xy is not None and map_size_xy is not None:
        map_cx = float(map_center_xy[0])
        map_cy = float(map_center_xy[1])
        map_size = (float(map_size_xy[0]), float(map_size_xy[1]))
    else:
        try:
            bounds = scene.mi_scene.bbox()
            b_min = [float(bounds.min.x), float(bounds.min.y)]
            b_max = [float(bounds.max.x), float(bounds.max.y)]
            if not (math.isfinite(b_min[0]) and math.isfinite(b_max[0])):
                raise RuntimeError("bbox not finite")
            map_size = (b_max[0] - b_min[0], b_max[1] - b_min[1])
            map_cx = (b_min[0] + b_max[0]) / 2
            map_cy = (b_min[1] + b_max[1]) / 2
        except Exception:
            map_cx = float(tx_position[0])
            map_cy = float(tx_position[1])
            map_size = (20.0, 20.0)

    solver = rt.RadioMapSolver()
    slices: list[dict] = []

    n = len(z_heights) if z_heights else 1
    for i, z in enumerate(z_heights):
        try:
            radio_map = solver(
                scene,
                center=[map_cx, map_cy, float(z)],
                orientation=[0.0, 0.0, 0.0],
                size=list(map_size),
                cell_size=list(cell_size),
                samples_per_tx=int(samples_per_tx),
                max_depth=int(max_depth),
                los=True,
                specular_reflection=bool(specular),
                diffuse_reflection=False,
                refraction=bool(refraction),
                diffraction=bool(diffraction),
                seed=42,
            )
            # rss shape: (num_tx, n_y, n_x) linear Watts
            rss = radio_map.rss.numpy()
            # Sum over TX (currently 1)
            if rss.ndim == 3:
                rss = rss.sum(axis=0)
            # Convert W → dBm; guard log10(0)
            with_floor = np.maximum(rss, 1e-15)
            dbm = 10.0 * np.log10(with_floor) + 30.0
            slices.append({
                "z": float(z),
                "grid": dbm.tolist(),
            })
        except Exception as e:
            # If a slice fails (e.g. cell_size too fine), keep going.
            slices.append({"z": float(z), "grid": [], "error": str(e)[:200]})

        if progress_cb:
            progress_cb(i + 1, n, f"Sionna RT z={z:.1f}m done")

    # Clean up (remove tx so subsequent calls don't accumulate)
    try:
        scene.remove(tx.name)
    except Exception:
        pass

    return slices
