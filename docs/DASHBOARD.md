# Dashboard usage

The AutoNetSim dashboard is a Flask + Three.js web app that lets you
build wireless simulation scenes, drive coverage / ray-tracing runs,
and chat with an LLM agent that has full access to the rf-simulator
skill.

---

## Launch

```bash
conda activate sionna
PYTHONPATH=. python web/dashboard_app.py --port 8080
```

Then open **http://localhost:8080**. On a remote server, use SSH port
forwarding or your VPN — see the "Remote access" section below.

---

## Configure the chat LLM

The chat panel talks to any OpenAI-compatible endpoint. Pick one and
fill in `.env`:

### Option A: Anthropic direct

```bash
DASHBOARD_CHAT_BASE_URL=https://api.anthropic.com/v1
DASHBOARD_CHAT_API_KEY=sk-ant-YOUR-KEY
DASHBOARD_CHAT_MODEL=claude-sonnet-4-6
```

Get a key at https://console.anthropic.com.

### Option B: OpenAI-compat relay (Kimi, PackyCode, exchangetoken, ZetaAPI, ...)

```bash
DASHBOARD_CHAT_BASE_URL=https://<gateway>/v1
DASHBOARD_CHAT_API_KEY=<your-relay-key>
DASHBOARD_CHAT_MODEL=claude-sonnet-4-6      # or any model the relay offers
```

Ping the relay's `/v1/models` endpoint to see supported model IDs.

### Option C: Local model (Ollama / LM Studio / vLLM)

```bash
DASHBOARD_CHAT_BASE_URL=http://localhost:11434/v1     # Ollama
DASHBOARD_CHAT_API_KEY=any-non-empty-string
DASHBOARD_CHAT_MODEL=llama3.1
```

Large models behind a slow gateway can take more than a minute per reply
(the worker prompt carries the whole skill, ~35k tokens). The per-call
HTTP timeout defaults to 300 s; override it with
`DASHBOARD_CHAT_TIMEOUT=<seconds>`. The badge next to "AI ASSISTANT"
shows whatever `DASHBOARD_CHAT_MODEL` is set to.

`.env` is gitignored — never commit real keys. If either the URL or key
is missing at startup, the chat panel returns *"Chat is disabled …"*
rather than falling back to any shared account.

---

## Chat actions

The chat uses a two-agent architecture:

1. **Worker** — the full rf-simulator skill (SKILL.md + Sionna API
   reference) is injected as the system prompt. Produces technical
   answers *and* emits structured `action` blocks that drive the UI.
2. **General** — summarizes the worker's output as a short, friendly
   reply for the user (hides code, PARAMS blocks, and internal tags).

The frontend auto-sorts actions by dependency priority
(`set_material → set_room_size/height → add_furniture → set_tx_* /
configure_antenna / set_ap_orientation → move/rotate/remove_furniture
→ compute_coverage`) so ordering slips don't break the pipeline.

### Supported action types

| Action | Example prompt |
|---|---|
| `set_room_size {width, length}` | "Make the room 8 x 6 meters" |
| `set_room_height {height}` | "Change ceiling to 3.5 m" |
| `create_apartment {rooms: [{id, type, width, depth}], name?, height?}` | "Build an apartment with a living room, a kitchen and two bedrooms" |
| `set_material {surface, material}` | "Change the interior wall to concrete" — `surface` is `interior_wall`, `exterior_wall`, `wall`, `floor` or `ceiling`; `material` is an ITU name (`plasterboard`, `brick`, `concrete`, `wood`, `glass`, `marble`, `metal`). On a loaded scene with a `scene.xml` the change is applied immediately and coverage is recomputed |
| `add_furniture {items: [{quantity, category}]}` | "Add a sofa and 2 chairs" |
| `move_furniture {category, x, y}` | "Move the desk to (3, 2)" |
| `rotate_furniture {category, absolute_deg or delta_deg}` | "Rotate the chair 90°" |
| `remove_furniture {category}` | "Remove the bookcase" |
| `set_tx_position {x, y, z}` | "Move the AP to (5, 4, 2.8)" |
| `set_tx_power {power_dbm}` | "Set TX power to 15 dBm" |
| `set_frequency {ghz}` | "Change frequency to 28 GHz" |
| `configure_antenna {pattern, rows, cols, polarization, azimuth, elevation}` | "Use a 4x4 tr38901 antenna" |
| `set_ap_orientation {azimuth, elevation}` | "Point the AP north with 15° downtilt" |
| `compute_coverage {}` | "Compute the coverage map" |
| `load_scene {scene_id}` | "Load Room_5x4_abc" |
| `delete_scene {scene_id}` | "Delete Room_5x4_abc" |
| `create_outdoor {}` | "Switch to an outdoor scene" |
| `fetch_osm {name, lat, lon, radius}` | "Fetch Athens downtown from OSM" |

Coordinates are clamped to room bounds — asking for the AP at
(100, 100, 100) in a 6×5×3 room pins it to the corner and adds a
"clamped from … to …" note to the chat.

### Multi-room apartments

`create_apartment` calls `POST /api/scenes/apartment/generate`, which runs
`scripts/generate_apartment.py`: rooms (`living_room`, `kitchen`,
`bedroom`, `bathroom`, `balcony`) are laid out in a two-column grid in
list order, shared walls become plasterboard partitions with a 0.9 m
doorway, and each room gets type-appropriate furniture that is kept clear
of the doorways. The scene is written as `scene.xml` (Sionna RT),
`scene.glb` (viewport) and `metadata.json` (draggable furniture), then
loaded. A repeated name gets a `-2`, `-3`, … suffix instead of failing.

### Coverage engine and measured results

If the loaded scene has a `scene.xml`, coverage is ray-traced with
`sionna.rt.RadioMapSolver` (3 bounces, reflection + refraction) and the
stats panel shows `Sionna RT`; otherwise a fast analytical model is used.
After each run the dashboard records frequency, AP position, whole-scene
mean / min / max RSS, coverage % above −80 dBm and the mean RSS of every
room, and sends the last few runs with each chat message. Ask for a
change and the numbers in two separate messages ("…and recompute", then
"how much did the bedroom drop?") — the reply to the first message is
written before its run has finished.

Positions of category "lamp" / "chandelier" / "pendant" / "ceiling_fan"
are auto-mounted at ceiling height instead of the floor.

---

## Direct 3D interaction

The 3D viewport supports:

| Gesture | Result |
|---|---|
| Click furniture | Select (green ring appears) |
| Drag selected furniture | Move on floor plane, walls stop it |
| Drag the ring | Rotate about vertical axis |
| `Delete` / `Backspace` | Remove selected |
| `Ctrl+Z` / `Ctrl+Shift+Z` | Undo / Redo |
| Sidebar `tx-x` / `tx-y` / `tx-height` | Manual AP position |
| ◀ tab on the sidebar edge / ▼ tab above the bottom panel | Collapse or restore that panel (remembered across reloads) |

Chat and direct interaction share scene state, so you can chat to seed
a room, then hand-tune, then chat "compute coverage".

---

## Remote access

If you're running the dashboard on a remote server (not your laptop):

**Simplest** — SSH port forward from your laptop:
```bash
ssh -L 8080:localhost:8080 user@server
# then open http://localhost:8080 in your laptop browser
```

**VSCode Remote-SSH users**: the PORTS panel auto-forwards. Click the
🌐 icon next to port 8080 to open in browser.

**On a campus / private network**: point your browser at the server's
LAN IP: `http://<server-ip>:8080`.

---

## Troubleshooting

**Chat says "Chat is disabled — please set …"** → `.env` is missing or
incomplete. Copy `.env.example`, fill in the three `DASHBOARD_CHAT_*`
vars, restart the dashboard.

**Chat replies `[worker error] TimeoutError: timed out`** → the LLM
endpoint did not answer within `DASHBOARD_CHAT_TIMEOUT` (default 300 s).
Raise it, or pick a faster model.

**Edited `templates/dashboard.html` but nothing changed** → Flask caches
templates when not in debug mode; restart the dashboard. Changes under
`static/` only need a hard refresh.

**Furniture catalog is empty** → `FUTURE_DATASET_PATH` not set or points
to an invalid directory. Verify `model_info.json` and per-model
`raw_model.obj` files exist at that path. Without the dataset, boxes
are used — RF simulation is still correct.

**Heatmap doesn't refresh after chat changes** → the auto-recompute
listener needs the browser to have the current JS. Do a hard refresh
(Cmd/Ctrl + Shift + R) to clear the cache.

**Coverage returns wrong RSS scale (e.g., −1 to +4 dBm)** → the button
uses `/api/coverage/thz/multi-z` (correct). The plain `/api/coverage/thz`
has a known bug and is not called by the UI. If you're hitting it
directly, switch to `/multi-z`.
