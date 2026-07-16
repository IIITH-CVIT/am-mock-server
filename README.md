# Mock AM Server

A lightweight FastAPI mock of the `am-master-server` face registration/identify
API, for local development and testing without the full stack (Postgres,
Celery, S3/MinIO, Qdrant). Data is stored in a local SQLite file instead.

## Features

- **Registration** — capture a name, visit date, time slot, ticket category,
  and one face photo. The face is embedded on the spot with the configured
  face model (see below); only the name, visit details, and the embedding are
  persisted (the image bytes are discarded).
- **Identify** — look up a registration by ID, or by face vector (nearest
  stored embedding within a configurable distance threshold).
- **Web UI** — a small registration form at `/` with webcam capture (falls
  back to file upload), for exercising the API by hand.

## One face model, selected by config

Detection always runs via OpenCV 5's native `cv2.FaceDetectorYN` (YuNet).
Embedding is one of two never-mixed pairings, selected by `models.embedder_model`
in `config.yaml`:

| Model | Vector size | How it embeds | Match cutoff |
|---|---|---|---|
| **sface** (default) | 128 numbers | `cv2.FaceRecognizerSF`, native OpenCV 5, no `onnxruntime` | distance < `0.8` |
| **auraface** | 512 numbers | `aurar100.onnx` (ArcFace-style) via `onnxruntime` | distance < `0.8` |

You don't pick a model per-request — the server always embeds with whichever
pairing `config.yaml` names, and on identify it expects a vector of that same
size. Whichever pairing is configured, the client (`Am-FaceRecognition-Client`)
must be set to send vectors from the same model — the server never re-derives
embeddings from pixels on identify, it only vector-searches whatever the client
submits.

Fingerprint identification (`type=fingerprint`) is wired into the schema and
`/api/v1/identify/` endpoint for API-shape compatibility with the real server,
but there's no fingerprint capture route in this mock, so it will always
return "no match found".

## Running

**Prerequisites:** Python 3.13 on the host (used only to build the binary). No container runtime, no Podman/Docker.

```bash
./run.sh
```

`run.sh` creates a `.venv`, installs dependencies, and builds a standalone binary with PyInstaller (`dist/mock-server`) the first time you run it, then execs it directly. On later runs it skips the rebuild unless something under `app/` or `requirements.txt` changed, so re-running `./run.sh` is the same "one command" whether it's the first launch or the hundredth. The server listens on `http://localhost:8000`:

- `/` — registration web UI
- `/docs` — interactive API docs (Swagger UI)
- `/health` — health check

`./models` (ONNX weights), `./config.yaml`, and `./data` (the SQLite DB) are
plain files read directly from disk next to the binary — no bind mounts
involved.

Since the binary bundles the application code, after changing files under `app/` you need to rebuild (`./run.sh` again) to pick up the changes.

### Concurrency

`register` and `identify` both run in FastAPI's threadpool rather than on the main event loop, so concurrent requests (e.g. simulating multiple kiosks registering at once) don't block each other or `/health`. If you're benchmarking or load-testing against this mock, note that SQLite itself becomes the bottleneck under heavy concurrent writes before the app layer does. This mock isn't a substitute for load-testing against the real Postgres-backed server.

## How to use

A walkthrough once the server is up (`./run.sh`, then `curl http://localhost:8000/health` → `{"status":"ok"}`).

### 1. Register a face (web UI)

Open **http://localhost:8000** in a browser and fill the form:

| Field | Example | Rule |
|---|---|---|
| Full name | `Alice Kumar` | ≥ 5 characters |
| Photo | webcam capture, or the file-upload fallback | a clear front-facing face |

`date_of_visit`, `timeslot`, and `ticket_category` aren't collected from the
form — the server stamps them itself (today's date, and fixed defaults).

Submit. The server detects the face, computes the configured embedding (128-dim sface by default), and stores the registration — you get back a `registration_id` and a message naming the model and dimension stored. If it reports **"no face detected"**, use a clearer, front-facing photo.

### 2. Confirm it was stored

```bash
curl http://localhost:8000/api/v1/registrations/        # add "| jq" if you have it
```
Lists everyone registered, with their stored vector metadata (kind / model / dim).

### 3. Identify

**By registration ID** (exact lookup — no photo needed):
```bash
curl -X POST http://localhost:8000/api/v1/identify/ -F "type=id" -F "id=<registration_id>"
```

**By face** — this route matches on a face *vector*, not an image (the server vector-searches; it does not re-embed here). The vector must match the dimension of the configured `models.embedder_model` (128 for sface, 512 for auraface). Two easy ways to produce a vector from a photo:
- **Mock client** (the realistic edge-device flow): `client.py --server photo.jpg`, configured to send vectors from the same model this server is configured with — see the client README, and `## Running end-to-end with the mock client` below.
- **Locally**: `python -m app.cli_identify /path/to/photo.jpg` (from the venv, with the server running) — embeds the photo (with the configured model) and calls identify for you.

A match returns the person's details with `distance` (lower = better; under `identify.face_recognition_threshold` — `0.8` by default — counts as a match) and `confidence`.

### 4. Inspect the database directly

```bash
./query_db.sh
sqlite> SELECT id, full_name, date_of_visit FROM registrations;
```

### Interactive API docs

**http://localhost:8000/docs** — Swagger UI. Try every endpoint from the browser and see the exact request/response schemas.

## Known Issues & Fixes

The following were found and fixed:

- **Registration DB writes failing (`sqlite3.OperationalError: attempt to write a readonly database`)** caused by `./data` being owned by `root` on the host from an earlier container run under a different UID mapping. Fixed by `sudo chown -R $(id -u):$(id -g) ./data`. If you hit this again after a fresh clone, check `ls -ld ./data` first, it must be owned by your own
user, not root.

- **`register()` blocked the event loop** was `async def` calling a synchronous, CPU-bound ONNX embed directly; now runs threadpooled like identify()` already did, so concurrent requests (including `/health`) don't stall during a registration.

- **Malformed `face_vector`/`fingerprint_vector` caused an unhandled 500**: `_parse_vector` now catches JSON/parse errors and returns a clean `400`, and rejects empty vectors or vectors over 4096 elements.

- **`face_vector` with the wrong dimension silently returned "no match found"**. Now returns an explicit `400` naming the expected dimension (512, read from the loaded model) vs. what was sent. This is a response-shape change from earlier behavior: a wrong-dim vector used to look
identical to a genuine no-match; it's a `400` now.

- **Unbounded file upload**: `register`'s `image` field now requires `content_type` to start with `image/` (`415` otherwise) and caps upload size at 10MB (`413` otherwise).

- **No global exception handler**: any unhandled exception previously returned Starlette's raw plain-text 500 (which broke the web UI's `JSON.parse`). A global handler now logs the full traceback server-side and returns clean JSON `{"detail": "internal server error"}`.

- **`full_name` had no length limits; `GET /registrations/` had no `limit` cap**, now `5-200` characters and `limit` capped at 500, both enforced with FastAPI's own `422` validation. (`ticket_category` is no longer client input at all — see `## API` below.)

- **Config defaults drifted from `config.yaml`**: in-code dataclass defaults now match the shipped `config.yaml` exactly, and a missing config file now logs a `WARNING` at startup instead of silently using different values.

- **YuNet model output names weren't validated**: a mismatched ONNX export now fails fast at container startup with a clear message, instead of an obscure `KeyError` mid-request.

- **Missing model file crashed boot with a raw onnxruntime error**: `FaceEngine()` is built at import time, so if `./models` isn't bind-mounted the container died with an unhelpful `NoSuchFile` stack trace — the classic first-run mistake. Fixed: model loading now fails fast with an actionable message that names the path and points at the `./models` bind mount (and distinguishes a missing file from a present-but-corrupt one).

- **Dockerfile → Containerfile, Podman migration**: see `## Running` below.

- **compose dropped in favor of plain `podman build`/`podman run`**: this project is a single container with no inter-container networking or dependency ordering, so `podman-compose`/`podman compose` (and the Podman API socket it talks to) added a toolchain dependency without buying anything. `run.sh` now calls `podman build` and `podman run` directly.

- **Bind mounts failed under rootless Podman on SELinux-enforcing hosts (Fedora/RHEL)**: without an SELinux relabel suffix the container gets permission-denied reading `./config.yaml`/`./models` and writing `./data`. Fixed at the time by carrying `:Z` (private relabel) on each bind mount in `run.sh`. No longer applicable — see the next entry.

- **Podman/containers dropped in favor of a standalone binary**: this mock doesn't need process isolation or image distribution, so the container layer (`Containerfile`, `compose.yml`, Podman itself) was pure toolchain overhead. `run.sh` now builds the app into a single native executable with PyInstaller (`dist/mock-server`) and runs that directly — no container runtime, no bind mounts (just plain files in `./config.yaml`, `./models`, `./data` read straight off disk), no SELinux relabeling to worry about.

## API

### `POST /api/v1/registrations/register`

`multipart/form-data`:

| field             | type | notes                                |
|-------------------|------|---------------------------------------|
| `full_name`       | str  | required, 5 - 200 characters          |
| `image`           | file | required, one face photo  , `image/*`, max 10MB|

`date_of_visit`, `timeslot`, and `ticket_category` are not client input —
they're always set server-side: `date_of_visit` to the current date at
registration time, `timeslot` fixed to `"10:00"`, and `ticket_category` fixed
to `"general"` (see `DEFAULT_TIMESLOT` / `DEFAULT_TICKET_CATEGORY` in
`app/routers/registrations.py`).

Returns `{ registration_id, status, message }`. 422 if no face is detected in
the image; 415 if the uploaded file's content-type isn't `image/*`; 413 if it exceeds 10MB.

Detection + embedding runs synchronously on CPU (~a few hundred ms per image depending on hardware), so a single `register` call blocks for that long. This mirrors the real server's own per-request latency for this step, it isn't mock-specific overhead.

### `GET /api/v1/registrations/` and `GET /api/v1/registrations/{id}`

List/fetch registrations, including visit details and stored vector metadata
(kind/model/dim — not the raw vector).
`limit` defaults to 100, capped at 500 (`422` if exceeded)

### `POST /api/v1/identify/`

`multipart/form-data`, `type` selects the lookup mode:

- `type=id`, `id=<registration_id>` — direct lookup.
- `type=face`, `face_vector=<JSON array or comma-separated floats>` — nearest
  stored face embedding. The vector must match the configured `models.embedder_model`
  dimension (**128-dim** for sface, **512-dim** for auraface), matched within
  `identify.face_recognition_threshold`; any other length returns a `400` error
  naming the expected size.
- `type=fingerprint` — accepted but always returns "no match found" (see
  above). Vectors longer than 4096 elements are rejected with `400`
  regardless of type.

Optional `vector_type` (`registration` / `fru` / `sau`) filters candidates by
source. Returns an `IdentifyResponse` with `match_type`, `distance`,
`confidence`, and the matched registration's fields when found.

400 if `face_vector` / `fingerprint_vector` isn't a valid JSON or comma separated floating-point values, or is empty. 

## Configuration

`config.yaml` (read from disk on startup, next to the binary — see [app/core/config.py](app/core/config.py)):

```yaml
server:
  host: "0.0.0.0"
  port: 8000

database:
  path: "./data/db.sqlite"

models:
  face_detector_path: "./models/face_detection_yunet_2026may.onnx"
  face_recognizer_path: "./models/face_recognition_sface_2021dec.onnx"
  face_recognizer_auraface_path: "./models/aurar100.onnx"
  embedder_model: "sface"   # sface (128-dim, native cv2) | auraface (512-dim, onnxruntime)
  face_detector_input_size: 640
  face_detector_score_threshold: 0.5

identify:
  face_recognition_threshold: 0.8
  fingerprint_recognition_threshold: 0.7
  default_n: 10
```

The YuNet detector and both embedder models are loaded from `./models`.
`aurar100.onnx` is large (~250MB) and gitignored — it's expected to exist on
disk (copied or downloaded there) even though it's not tracked by git.

`config.yaml` is the single source of truth for all settings. If it's missing or unreadable at startup (e.g. `CONFIG_PATH` misconfigured, or the file just isn't next to the binary), the server logs a `WARNING` and falls back to the built-in defaults in `app/core/config.py`, and those defaults are kept in sync with the shipped `config.yaml` and covered by a test (`tests/test_config.py`), but if you ever
see that warning in the logs, something is wrong with where `config.yaml` lives, not your config values.

## Face pipeline

[app/core/face_engine.py](app/core/face_engine.py) always detects via OpenCV 5's
native `cv2.FaceDetectorYN` (YuNet), returning a bbox + 5 landmarks. Embedding is
one of two never-mixed pairings, selected by `models.embedder_model`:

- **sface** (default) — `cv2.FaceRecognizerSF` aligns the crop via `alignCrop()`
  and produces a normalized 128-dim embedding via `feature()`. No `onnxruntime`,
  runs entirely through OpenCV's built-in `objdetect` API.
- **auraface** — `aurar100.onnx` (ArcFace-style) via `onnxruntime`: the detector's
  landmarks are warped into the 112x112 ArcFace reference pose
  (`estimateAffinePartial2D` + `warpAffine`), BGR→RGB, `(x-127.5)/128`, producing a
  normalized 512-dim embedding.

Whichever pairing is configured, the client (`Am-FaceRecognition-Client`) must be
set to the same `embedder.model` — the server never re-derives embeddings from
pixels, it only vector-searches whatever the client submits.

## Utilities

- `./query_db.sh` — open a `sqlite3` shell on `data/db.sqlite`.
- `python -m app.cli_identify photo.jpg` (from the venv) — embeds a
  photo and calls `/api/v1/identify/` with it, like a real edge device would.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt   # adds pytest, httpx, ruff, pyinstaller on top of requirements.txt
```

Run the tests. `app/core/face_engine.py` builds a real `FaceEngine` at import
time, so the paths in `config.yaml` (or whatever `CONFIG_PATH` points at) must
actually resolve — the shipped `config.yaml` already uses paths relative to
the repo root (`./models/...`), so it resolves as-is when run from the repo:

```bash
python -m pytest tests/ -v
```

Use `python -m pytest`, not a bare `pytest` — the `-m` form adds the repo root
to `sys.path` so `import app...` resolves; a bare `pytest` invocation doesn't,
and fails every test file with `ModuleNotFoundError: No module named 'app'`.

Lint (also enforced by `tests/test_lint.py`, so a `pytest` run catches lint
regressions too):

```bash
ruff check .
```

## Project layout

```
app/
  core/
    config.py       # loads config.yaml into typed Settings
    database.py      # SQLite schema + queries
    face_engine.py    # YuNet detection + sface (native cv2) or auraface (onnxruntime) embedding
  routers/
    registrations.py # register / list / get
    identify.py       # id / face / fingerprint lookup
  schemas/            # pydantic request/response models
  static/             # registration web UI (HTML/CSS/JS)
  main.py             # FastAPI app, routes, static mount
  __main__.py         # uvicorn entrypoint (also the PyInstaller build target)
models/                # ONNX weights, read from disk at runtime
data/                  # SQLite DB (gitignored)
config.yaml
run.sh                 # builds the binary (once) and runs it
query_db.sh
```

## Running end-to-end with the mock client

The full loop on one machine — mock server + mock client — in four steps. Assumes the two repos sit side by side (`am-mock-server/` and `am-mock-client/`).

**1. Start the server** (this repo):
```bash
cd am-mock-server
./run.sh                                    # builds the binary (first run) and starts it
curl http://localhost:8000/health           # -> {"status":"ok"}
```

**2. Register a face** via the web UI at **http://localhost:8000** (see `## How to use` above). Note the name you used.

**3. Set up the client** (`am-mock-client`):
```bash
cd ../am-mock-client
./setup.sh                                   # native venv
```

**4. Identify** with a photo of the same person:
```bash
.venv/bin/python client.py --server <photo.jpg>
# -> >>> Recognised: <name>     (distance comfortably under the match cutoff)
```

The client must be configured to send vectors from the **same** model this
server is configured with (`models.embedder_model` in `config.yaml` — `sface`
by default, 128-dim). Registration and identification only agree if both sides
speak the same embedding space; see `am-mock-client/README.md` for how to
select its embedder. Talks to this same server on `localhost:8000`.
