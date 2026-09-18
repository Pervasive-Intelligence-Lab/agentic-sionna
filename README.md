# AutoNetSim

**Turn "build a 6×5 office and compute coverage at 5 GHz" into a runnable
NVIDIA Sionna simulation — with a live 3D dashboard.**

AutoNetSim is a Claude-Code-native agent that generates, executes, and
verifies wireless network simulations from natural-language requests.
It ships as:

- **`.claude/skills/rf-simulator/`** — the procedural skill (routing,
  templates, references, memory) that turns intent into Sionna 2.0 code
- **`web/`** — a Flask + Three.js dashboard for interactive scene
  building, coverage visualisation, and chat-driven control
- **`benchmark/`** — 100+ tasks with a 3-layer verifier (artifact,
  executable, oracle) that scored the paper's results

Paper: **[AutoNetSim: Intent-Driven Wireless Network Experimentation
with Self-Evolving Agents](main.pdf)**  
Runnan Si\*, John Song\*, Haijian Sun, Zhenlin An — University of Georgia  
**IEEE ICNP 2026** · [Project page](https://pervasive-intelligence-lab.github.io/agentic-sionna/)

## Abstract

Wireless network simulation and optimization are difficult because they
require solving a complex cross-layer configuration problem while
interacting with complex 3D radio environments. For decades, engineers
have relied on channel models, channel simulators, and extensive
standards to guide this process, but using them correctly still requires
substantial domain expertise and engineering effort. Recent
language-model agents can automate parts of wireless reasoning and code
generation, yet they are not trained for the cross-layer setting in
which an agent must construct a 3D radio environment, bind it to radio
and network assumptions, and run simulations inside it. We present
**AutoNetSim**, a self-evolving language-agent system for intent-driven
wireless network simulation. Given a natural-language request, it
constructs an executable *radio environment* that combines a 3D scene
with a cross-layer wireless-system configuration and optimization. The
system uses a multi-agent architecture with specialized scene-building,
simulation, reflection, planning, and skill-learning agents. It learns
from previous tutorials, worked examples, prior simulation results, and
its own failed trajectories, then updates a procedural skill and
knowledge base through verifier-driven optimization. We evaluate the
system on benchmark suites covering 3D radio environment generation,
wireless simulation, and token efficiency across multiple indoor and
outdoor scenarios. On 3D radio environment generation, AutoNetSim
reaches 72.5% held-out pass rate, whereas both baselines fail to solve
any test task. Across ten ray-tracing, physical-layer, and
network/system-level simulation families, it averages 95.5% pass rate,
while both baselines stay below 70.0%.

---

## Quick start (5 min)

```bash
# 1. Clone
git clone https://github.com/Pervasive-Intelligence-Lab/agentic-sionna.git
cd agentic-sionna

# 2. Install (creates conda env `sionna`, installs deps)
bash install.sh

# 3. Configure your LLM key
cp .env.example .env
$EDITOR .env               # fill in DASHBOARD_CHAT_API_KEY

# 4. Launch the dashboard
conda activate sionna
PYTHONPATH=. python web/dashboard_app.py --port 8080

# 5. Open http://localhost:8080
```

Ask the chat panel: *"Build a 6×5 room with a desk and two chairs, put
the AP at (3, 2.5, 2.8), then compute coverage at 5 GHz."*

You should see the room render, furniture drop in, AP marker move,
and a coverage heatmap appear — all driven by one prompt.

Want a whole home instead of one room? See the next section.

---

## New: multi-room 3D scenes from one sentence

Describe a home in plain language and the dashboard builds it — rooms,
interior partitions with doorways, room-appropriate 3D furniture — then
ray-traces it with Sionna RT. No floor plan, no XML, no coordinates.

**1. Build it.** Type into the chat panel:

```
Build an apartment with a living room, a kitchen, two bedrooms and a
bathroom, and load it.
```

In 20–40 s the scene appears in the viewport and in the scene dropdown
(as `apartment-NN`, or the name you give it: *"…call it family-flat"*).

**2. Say as much or as little as you like.**

| You say | What happens |
|---|---|
| *"a living room and a bedroom"* | Typical sizes are used (living room 5×4 m, kitchen 3×4, bedroom 4×3.5, bathroom 2×3.5, balcony 3×2; 3 m ceiling) |
| *"a 6 by 4.5 m living room and two 4 by 3.5 m bedrooms"* | Your dimensions are used |
| *"…with a 2.7 m ceiling"* | Sets the height |
| *"put some furniture in each room"* | Always done: sofa / coffee table / TV stand in the living room, bed / nightstand / wardrobe / desk in bedrooms, table and chairs in the kitchen, … (real 3D-FUTURE meshes if the dataset is installed, boxes otherwise) |

Room types: `living_room`, `kitchen`, `bedroom`, `bathroom`, `balcony`.
Rooms are placed in a two-column grid in the order you list them, so
name the living room first. Walls shared by two rooms become
plasterboard partitions with a 0.9 m doorway, and furniture is kept
clear of the doorways.

**3. Experiment on it, one sentence at a time.**

```
Put a 5 GHz access point on the ceiling in the middle of the living room and compute the coverage.
Move the access point to the living-room corner farthest from the bedroom and recompute.
How much did the mean signal in the bedroom drop compared with the previous run?
Change the interior wall to concrete and recompute.
Compare the bedroom signal before and after the concrete wall.
Change the frequency to 2.4 GHz and recompute.
Using the measured coverage, what SNR and data rate can a WiFi 6 user expect in the bedroom, assuming an 80 MHz channel and a 7 dB noise figure?
```

Every answer about "how much did it change" quotes the ray-traced
measurements — whole-scene and **per-room** mean RSS from the last few
runs — not an estimate. Ask for a change and for its numbers in two
separate messages: the reply to *"…and recompute"* is written before
that run has finished. Furniture stays draggable in the viewport, so
you can also rearrange the generated home by hand.

**Without the chat** (scripting, batch generation):

```bash
# presets: 1br | studio | 2br | 3br
PYTHONPATH=. python scripts/generate_apartment.py --preset 2br --out-dir web/outputs/my-flat

# or any room list, through the running dashboard
curl -X POST http://localhost:8080/api/scenes/apartment/generate \
  -H 'Content-Type: application/json' \
  -d '{"name": "my-flat", "rooms": [
        {"id": "living",   "type": "living_room", "width": 5.5, "depth": 4.5},
        {"id": "bedroom1", "type": "bedroom",     "width": 4.0, "depth": 3.5},
        {"id": "bedroom2", "type": "bedroom",     "width": 4.5, "depth": 3.5}]}'
```

Each scene is a folder under `web/outputs/<name>/` holding `scene.xml`
(Mitsuba / Sionna RT, ITU radio materials), `scene.glb` (viewport) and
`metadata.json` (rooms + furniture); it can be loaded straight into
your own Sionna scripts with `sionna.rt.load_scene(".../scene.xml")`.

---

## Bring your own LLM

No API key is bundled. The dashboard reads its provider config from
environment variables (or from `.env`, which is gitignored):

| Provider | `DASHBOARD_CHAT_BASE_URL` | `DASHBOARD_CHAT_MODEL` | Get a key |
|---|---|---|---|
| **Anthropic direct** | `https://api.anthropic.com/v1` | `claude-sonnet-4-6` | https://console.anthropic.com |
| OpenAI-compat relay (Kimi, PackyCode, exchangetoken, ZetaAPI, ...) | `https://<gateway>/v1` | any model the relay offers | relay dashboard |
| Local model (Ollama, LM Studio, vLLM) | `http://localhost:11434/v1` | e.g. `llama3.1` | no key needed |

If either the URL or the key is missing at startup, the chat panel
returns *"Chat is disabled — please set …"* rather than silently
falling back to any shared account.

See [docs/DASHBOARD.md](docs/DASHBOARD.md) for provider-specific setup
and the full list of chat actions.

---

## What the chat can do (18 actions)

| Category | Examples |
|---|---|
| **Scene** | `Build a 6x5 room`, `Build an apartment with a living room and two bedrooms`, `Change ceiling to 3.5 m`, `Change the interior wall to concrete` |
| **Furniture** | `Add a sofa and two chairs`, `Move the desk to (3, 2)`, `Rotate the chair 90°`, `Remove the bookcase` |
| **AP / antenna** | `Move the AP to (5, 4, 2.8)`, `Set TX power to 15 dBm`, `Change frequency to 28 GHz`, `Use a 4x4 tr38901 antenna`, `Point the AP north with 15° downtilt` |
| **Simulation** | `Compute the coverage map` |
| **Scenes** | `Load Room_5x4_abc`, `Delete <scene>`, `Switch to outdoor`, `Fetch downtown Athens from OSM` |

Chat and direct 3D interaction (drag / rotate / delete) share the same
scene state — you can seed a room by chat and then hand-tune it in the
viewport.

**Coverage engine.** Scenes that ship a Mitsuba/Sionna `scene.xml`
(generated apartments, imported scenes) are solved with
`sionna.rt.RadioMapSolver` — walls, reflections and transmission through
ITU materials are ray-traced, and the stats panel reports
`Sionna RT`. Scenes without one fall back to a fast analytical model.
After every run the measured whole-scene and per-room mean RSS are fed
back to the chat agents, so "how did it change?" questions are answered
from data.


---

## Optional: real furniture meshes (3D-FUTURE)

Without the dataset, furniture renders as box primitives — the RF
simulation is still correct (ray tracing uses AABBs). For photorealistic
meshes, install the ~19 GB **3D-FUTURE** dataset:

1. Register at [Tianchi](https://tianchi.aliyun.com/dataset/98063)
2. Download `3D-FUTURE-model.zip`, extract somewhere
3. In `.env`: `FUTURE_DATASET_PATH=/absolute/path/to/3D-FUTURE-model`
4. Restart the dashboard

The dataset directory must contain `model_info.json` and per-model
`<uuid>/raw_model.obj` files (this is the layout the Tianchi ZIP uses).

---

## Repo layout

```
agentic-sionna/
├── .claude/skills/rf-simulator/    # The skill (SKILL.md + references + templates + agents)
├── web/                            # Flask dashboard (backend + frontend)
│   ├── dashboard_app.py            #   Flask app entry
│   ├── routes/                     #   API endpoints (chat, scenes, coverage, catalog, ...)
│   ├── static/                     #   Three.js viewport + CSS
│   └── templates/dashboard.html
├── src/                            # Runtime libraries (models, optimizer, exporters, wireless)
│   └── wireless/sionna_rt_backend.py  # RadioMapSolver backend used by the dashboard
├── scripts/
│   └── generate_apartment.py       # Multi-room apartment generator (XML + GLB + metadata)
├── benchmark/                      # Benchmark suite (verifier + tasks + oracles + metrics)
│   ├── verifier.py                 #   3-layer verifier
│   ├── tasks/                      #   Task specs (100+ scene-gen, RT, PHY, opt, system tasks)
│   ├── oracles/                    #   Reference answers per task
│   ├── compute_metrics.py          #   Extract continuous quality metrics from trial output
│   ├── paper_appendix_table.md     #   Failure taxonomy + wall-clock tables (paper appendix)
│   └── metrics_per_trial.csv       #   All 2094 trials' quality numbers
├── docs/
│   ├── QUICKSTART.md
│   ├── DASHBOARD.md                #   Chat actions + provider setup
│   └── BENCHMARK.md                #   How to reproduce paper numbers
├── main.pdf, architecture.pdf      # The paper + system diagram
├── install.sh, pyproject.toml
└── .env.example                    # Copy to .env with your keys
```

---

## Reproducing the paper

The `benchmark/` folder is self-contained.

```bash
# Run one trial (output goes to benchmark/results/<label>/)
PYTHONPATH=. python benchmark/run_benchmark.py \
    --label n1_demo \
    --tasks-file benchmark/tasks/_sources/n1_v2.json \
    --task-ids N1_cov_box_one_screen \
    --conditions with_skill --k 1

# Compute continuous quality metrics on existing trial output
python benchmark/compute_metrics.py
python benchmark/aggregate_metrics.py
python benchmark/paper_appendix_table.py

# Read the results
less benchmark/paper_appendix_table.md
```

See [docs/BENCHMARK.md](docs/BENCHMARK.md) for the full pipeline.

---

## Citation

If you use AutoNetSim in your work, please cite:

```bibtex
@inproceedings{si2026autonetsim,
  title     = {AutoNetSim: Intent-Driven Wireless Network Experimentation with Self-Evolving Agents},
  author    = {Si, Runnan and Song, John and Sun, Haijian and An, Zhenlin},
  booktitle = {Proceedings of the IEEE International Conference on Network Protocols (ICNP)},
  year      = {2026}
}
```

---

## Contributing

We welcome bug reports, tasks, and skill improvements. Please open an
issue before large PRs. Do **not** commit API keys or the 3D-FUTURE
dataset — both are excluded via `.gitignore`.

## License

MIT — see [LICENSE](LICENSE).

3D-FUTURE dataset is separately licensed by Alibaba; obtain it from
[Tianchi](https://tianchi.aliyun.com/dataset/98063) under their terms.
Sionna is licensed by NVIDIA under Apache-2.0.
