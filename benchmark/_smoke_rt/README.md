# Sionna RT smoke test

End-to-end verification that the v10 fix chain actually drives Sionna RT
(not FSPL fallback). Run from repo root with the conda env python:

```bash
$RF_SIONNA_PY benchmark/_smoke_rt/run_smoke.py
```

## What this proves

1. `import sionna` succeeds (v2.0.1, in `~/miniconda3/envs/sionna/`)
2. The `scene_gen` exporter writes a Mitsuba 3.0 XML that Sionna 2.0
   accepts (fixes a real exporter bug — `mat-{name}` ids were missing the
   `itu_` substring that Sionna 2.0's `process_xml` requires)
3. `sionna.rt.load_scene(xml)` consumes the XML and exposes the room
   geometry as RT objects
4. `RadioMapSolver` returns a real coverage grid

## Input

`benchmark/_review_dataset/S01/scene_state.json` — the first scene from
our 20-scene benchmark (5×4 m home office, drywall walls, 3 furniture
items, AP at centroid 2.5/2.0/2.5 m, 5 GHz, 20 dBm).

## Outputs (in `out/`)

| File | What it is |
|---|---|
| `scene.xml` | Mitsuba 3.0 scene Sionna RT loads from (the "agent produced an XML" signal in the verifier) |
| `scene.png` | Room layout (top-down) — for sanity-checking placement |
| `scene.glb` | 3D mesh (open in any glTF viewer) |
| `coverage_map.png` | **The headline image**: RT-computed RSS heatmap in dBm, AP marked as red star |
| `coverage_map.npy` | Raw RSS-in-dBm grid (NumPy) |

## Observed numbers

- Coverage grid: 7×8 cells @ 0.5 m → 4 m × 5 m (matches scene bounds)
- RSS range at this AP: roughly -60 to -50 dBm in the LOS region,
  lower near the bookshelf corner
- max_depth=2, samples_per_tx=10 000 (fast pilot settings — production
  T1 will use higher)

## Why this is a useful diagnostic

The advisor's question was: *"can the agent actually drive Sionna RT, or
is it silently falling back to a closed-form FSPL formula?"* The
`coverage_map.png` here is **not** what you'd get from FSPL — it has the
wall + furniture shadowing pattern that ray tracing produces. The
`scene.xml` is the artifact the verifier now requires (no XML = no RT =
verifier rejects the trial regardless of what the simulation reported).
