"""Pre-compute N1 reference oracle coverage maps for the 4 candidate scenes.

This is the "ground truth" we'll compare agent outputs against (Layer C
in the verifier). Also serves as a visual sanity check of:
  - AP positions (sensible? not embedded in a wall?)
  - Solver params (cell_size / max_depth / samples_per_tx) finishing in
    reasonable time
  - The resulting RSS range (does it look like a coverage map or a
    noisy mess?)

Run with the conda env python:
  $RF_SIONNA_PY \
    benchmark/_smoke_rt/run_n1_reference.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sionna
import sionna.rt as rt

# N1 scene configurations
N1_SCENES = [
    {
        "name": "box_two_screens",
        "kind": "indoor",
        "ap": (0.0, 0.0, 4.5),
        "cell_size": 0.25,
        "max_depth": 3,
        "samples_per_tx": 100_000,
        "vmin_dbm": -90, "vmax_dbm": -20,
    },
    {
        "name": "box_one_screen",
        "kind": "indoor",
        "ap": (0.0, 0.0, 4.5),
        "cell_size": 0.25,
        "max_depth": 3,
        "samples_per_tx": 100_000,
        "vmin_dbm": -90, "vmax_dbm": -20,
    },
    {
        "name": "simple_street_canyon",
        "kind": "outdoor",
        "ap": (0.0, 0.0, 15.0),
        "cell_size": 1.0,
        "max_depth": 2,
        "samples_per_tx": 50_000,
        "vmin_dbm": -120, "vmax_dbm": -30,
    },
    {
        "name": "etoile",
        "kind": "outdoor",
        "ap": (0.0, 0.0, 30.0),
        "cell_size": 2.0,
        "max_depth": 2,
        "samples_per_tx": 50_000,
        "vmin_dbm": -130, "vmax_dbm": -40,
    },
]

# Common physical params
FREQ_HZ = 5e9
TX_POWER_DBM = 20.0

OUT_DIR = Path(__file__).resolve().parents[1] / "oracles" / "n1"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_reference(cfg: dict) -> dict:
    name = cfg["name"]
    print(f"\n=== {name}  ({cfg['kind']}) ===")
    xml = getattr(rt.scene, name)
    scene = rt.load_scene(xml)

    # Set frequency + arrays (must be BEFORE adding TX)
    scene.frequency = FREQ_HZ
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1,
                                    pattern="iso", polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1,
                                    pattern="iso", polarization="V")

    # Place AP
    ap = cfg["ap"]
    scene.add(rt.Transmitter(name="ap0",
                             position=list(ap),
                             power_dbm=TX_POWER_DBM))
    print(f"  AP at {ap}  freq={FREQ_HZ/1e9:.1f} GHz  power={TX_POWER_DBM} dBm")

    # Compute coverage map
    t0 = time.time()
    solver = rt.RadioMapSolver()
    cm = solver(scene,
                max_depth=cfg["max_depth"],
                cell_size=(cfg["cell_size"], cfg["cell_size"]),
                samples_per_tx=cfg["samples_per_tx"])
    rss = cm.rss[0].numpy()  # tx 0, in Watts
    t1 = time.time()
    print(f"  solver elapsed: {t1-t0:.1f} s  cell_size={cfg['cell_size']}m  "
          f"max_depth={cfg['max_depth']}  samples={cfg['samples_per_tx']}")

    # Convert W → dBm; mask zeros to a floor
    rss_safe = np.where(rss > 0, rss, 1e-15)
    rss_dbm = 10 * np.log10(rss_safe) + 30
    rss_dbm = np.where(rss > 0, rss_dbm, np.nan)
    print(f"  grid shape: {rss.shape}")
    valid = ~np.isnan(rss_dbm)
    if valid.any():
        print(f"  RSS dBm: min={np.nanmin(rss_dbm):.1f}  "
              f"max={np.nanmax(rss_dbm):.1f}  "
              f"mean={np.nanmean(rss_dbm):.1f}  "
              f"valid_cells={valid.sum()}/{valid.size}")

    # Save artifacts
    np.save(OUT_DIR / f"{name}.npy", rss_dbm)
    bb = scene.mi_scene.bbox()
    extent = [float(bb.min.x), float(bb.max.x),
              float(bb.min.y), float(bb.max.y)]
    meta = {
        "scene_name": name,
        "kind": cfg["kind"],
        "ap_position": list(ap),
        "frequency_hz": FREQ_HZ,
        "tx_power_dbm": TX_POWER_DBM,
        "cell_size_m": cfg["cell_size"],
        "max_depth": cfg["max_depth"],
        "samples_per_tx": cfg["samples_per_tx"],
        "grid_shape": list(rss.shape),
        "extent_m": extent,
        "solver_elapsed_s": round(t1 - t0, 1),
        "rss_dbm_min": float(np.nanmin(rss_dbm)),
        "rss_dbm_max": float(np.nanmax(rss_dbm)),
        "rss_dbm_mean": float(np.nanmean(rss_dbm)),
        "valid_cells": int(valid.sum()),
        "total_cells": int(valid.size),
        "sionna_version": sionna.__version__,
    }
    (OUT_DIR / f"{name}.json").write_text(json.dumps(meta, indent=2))

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(rss_dbm, origin="lower", extent=extent,
                   cmap="viridis",
                   vmin=cfg["vmin_dbm"], vmax=cfg["vmax_dbm"])
    ax.plot(ap[0], ap[1], "r*", markersize=20,
            markeredgecolor="white", markeredgewidth=1.5,
            label=f"AP at z={ap[2]:.1f} m")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    sz_x = extent[1] - extent[0]
    sz_y = extent[3] - extent[2]
    ax.set_title(f"{name}  ({sz_x:.0f}×{sz_y:.0f} m, "
                 f"{cfg['kind']}, {FREQ_HZ/1e9:.1f} GHz, "
                 f"max_depth={cfg['max_depth']})\n"
                 f"RSS range: {meta['rss_dbm_min']:.1f} to "
                 f"{meta['rss_dbm_max']:.1f} dBm   "
                 f"({t1-t0:.1f} s)")
    plt.colorbar(im, ax=ax, label="RSS (dBm)")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{name}.png", dpi=110)
    plt.close()
    return meta


def main() -> int:
    print(f"sionna {sionna.__version__}")
    print(f"oracle output dir: {OUT_DIR}")
    all_meta = []
    for cfg in N1_SCENES:
        try:
            m = compute_reference(cfg)
            all_meta.append(m)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"!! {cfg['name']} FAILED: {e}")

    print("\n=== N1 reference summary ===")
    print(f"{'scene':24s} {'shape':>13s} {'time':>7s} {'RSS range (dBm)':>22s}")
    for m in all_meta:
        sh = f"{m['grid_shape'][0]}x{m['grid_shape'][1]}"
        rss = f"{m['rss_dbm_min']:.1f} .. {m['rss_dbm_max']:.1f}"
        print(f"{m['scene_name']:24s} {sh:>13s} {m['solver_elapsed_s']:>6.1f}s "
              f"{rss:>22s}")
    print(f"\nPNGs + NPYs + JSON written to: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
