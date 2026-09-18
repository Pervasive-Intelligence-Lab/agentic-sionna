"""Pre-compute N2 reference oracles: same 4 scenes + AP as N1, but at 2.4 GHz.

The N2 task asks the agent to compute coverage at the N1 frequency (5 GHz),
then re-run at 2.4 GHz, and emit a delta. We need the 2.4 GHz reference
for the verifier's after-state comparison.

The 5 GHz "before" reference is reused from benchmark/oracles/n1/.

Run:
  $RF_SIONNA_PY \
      benchmark/_smoke_rt/run_n2_reference.py
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

# Same 4 scenes + AP positions as N1, but at 2.4 GHz
N2_SCENES = [
    {
        "name": "box_two_screens", "kind": "indoor",
        "ap": (0.0, 0.0, 4.5), "cell_size": 0.25,
        "max_depth": 3, "samples_per_tx": 100_000,
        "vmin_dbm": -90, "vmax_dbm": -20,
    },
    {
        "name": "box_one_screen", "kind": "indoor",
        "ap": (0.0, 0.0, 4.5), "cell_size": 0.25,
        "max_depth": 3, "samples_per_tx": 100_000,
        "vmin_dbm": -90, "vmax_dbm": -20,
    },
    {
        "name": "simple_street_canyon", "kind": "outdoor",
        "ap": (0.0, 0.0, 15.0), "cell_size": 1.0,
        "max_depth": 2, "samples_per_tx": 50_000,
        "vmin_dbm": -120, "vmax_dbm": -30,
    },
    {
        "name": "etoile", "kind": "outdoor",
        "ap": (0.0, 0.0, 30.0), "cell_size": 2.0,
        "max_depth": 2, "samples_per_tx": 50_000,
        "vmin_dbm": -130, "vmax_dbm": -40,
    },
]

FREQ_HZ = 2.4e9   # ← the N2 "after" frequency
TX_POWER_DBM = 20.0

OUT_DIR = Path(__file__).resolve().parents[1] / "oracles" / "n2_freq"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_reference(cfg: dict) -> dict:
    name = cfg["name"]
    print(f"\n=== {name}  ({cfg['kind']}) @ 2.4 GHz ===")
    xml = getattr(rt.scene, name)
    scene = rt.load_scene(xml)
    scene.frequency = FREQ_HZ
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1,
                                    pattern="iso", polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1,
                                    pattern="iso", polarization="V")
    ap = cfg["ap"]
    scene.add(rt.Transmitter(name="ap0", position=list(ap),
                             power_dbm=TX_POWER_DBM))

    t0 = time.time()
    solver = rt.RadioMapSolver()
    cm = solver(scene, max_depth=cfg["max_depth"],
                cell_size=(cfg["cell_size"], cfg["cell_size"]),
                samples_per_tx=cfg["samples_per_tx"])
    rss = cm.rss[0].numpy()
    t1 = time.time()

    rss_dbm = np.where(rss > 0, 10 * np.log10(np.maximum(rss, 1e-15)) + 30,
                       np.nan)
    valid = ~np.isnan(rss_dbm)
    print(f"  shape={rss.shape}  t={t1-t0:.1f}s  RSS_dBm "
          f"min={np.nanmin(rss_dbm):.1f}  max={np.nanmax(rss_dbm):.1f}  "
          f"mean={np.nanmean(rss_dbm):.1f}  "
          f"valid={valid.sum()}/{valid.size}")

    np.save(OUT_DIR / f"{name}_2ghz.npy", rss_dbm)

    bb = scene.mi_scene.bbox()
    extent = [float(bb.min.x), float(bb.max.x),
              float(bb.min.y), float(bb.max.y)]
    meta = {
        "scene_name": name, "kind": cfg["kind"],
        "ap_position": list(ap), "frequency_hz": FREQ_HZ,
        "tx_power_dbm": TX_POWER_DBM,
        "cell_size_m": cfg["cell_size"], "max_depth": cfg["max_depth"],
        "samples_per_tx": cfg["samples_per_tx"],
        "grid_shape": list(rss.shape), "extent_m": extent,
        "solver_elapsed_s": round(t1 - t0, 1),
        "rss_dbm_min": float(np.nanmin(rss_dbm)),
        "rss_dbm_max": float(np.nanmax(rss_dbm)),
        "rss_dbm_mean": float(np.nanmean(rss_dbm)),
        "sionna_version": sionna.__version__,
    }
    (OUT_DIR / f"{name}_2ghz.json").write_text(json.dumps(meta, indent=2))

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(rss_dbm, origin="lower", extent=extent,
                   cmap="viridis",
                   vmin=cfg["vmin_dbm"], vmax=cfg["vmax_dbm"])
    ax.plot(ap[0], ap[1], "r*", markersize=20,
            markeredgecolor="white", markeredgewidth=1.5,
            label=f"AP @ z={ap[2]:.1f} m")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"{name}  @ 2.4 GHz   "
                 f"RSS range {meta['rss_dbm_min']:.1f}..{meta['rss_dbm_max']:.1f} dBm")
    plt.colorbar(im, ax=ax, label="RSS (dBm)")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{name}_2ghz.png", dpi=110)
    plt.close()

    # Also save the delta vs the 5 GHz reference (sanity-check)
    ref5_path = (Path(__file__).resolve().parents[1] / "oracles" / "n1"
                 / f"{name}.npy")
    if ref5_path.exists():
        ref5 = np.load(ref5_path)
        if ref5.shape == rss_dbm.shape:
            both_valid = np.isfinite(ref5) & np.isfinite(rss_dbm)
            delta = rss_dbm - ref5
            mean_delta = float(np.nanmean(delta[both_valid])) if both_valid.any() else float("nan")
            print(f"  delta (2.4 - 5 GHz) mean over {int(both_valid.sum())} cells: "
                  f"{mean_delta:+.2f} dB  (FSPL theory: +6.4 dB)")
            np.save(OUT_DIR / f"{name}_delta_2ghz_minus_5ghz.npy", delta)
            return {**meta, "delta_dbm_mean_vs_5ghz": mean_delta}

    return meta


def main() -> int:
    print(f"sionna {sionna.__version__}  freq={FREQ_HZ/1e9:.1f} GHz  "
          f"oracle out: {OUT_DIR}")
    summary = []
    for cfg in N2_SCENES:
        try:
            m = compute_reference(cfg)
            summary.append(m)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"!! {cfg['name']} FAILED: {e}")

    print("\n=== N2 reference summary (2.4 GHz) ===")
    print(f"{'scene':24s} {'shape':>13s} {'time':>7s} "
          f"{'RSS_dBm range':>22s} {'delta vs 5GHz':>16s}")
    for m in summary:
        sh = f"{m['grid_shape'][0]}x{m['grid_shape'][1]}"
        rss = f"{m['rss_dbm_min']:.1f}..{m['rss_dbm_max']:.1f}"
        dlt = m.get("delta_dbm_mean_vs_5ghz")
        dlt_s = f"{dlt:+.2f} dB" if dlt is not None else "(no ref)"
        print(f"{m['scene_name']:24s} {sh:>13s} "
              f"{m['solver_elapsed_s']:>6.1f}s {rss:>22s} {dlt_s:>16s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
