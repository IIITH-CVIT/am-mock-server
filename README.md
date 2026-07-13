# Mock AM Server

A lightweight FastAPI mock of the `am-master-server` face registration/identify
API, for local development and testing without the full stack (Postgres,
Celery, S3/MinIO, Qdrant). Data is stored in a local SQLite file instead.

## Features

- **Registration** — capture a name, visit date, time slot, ticket category,
  and one face photo. The face is embedded on the spot with **two face models**
  (see below); only the name, visit details, and the two embeddings are
  persisted (the image bytes are discarded).
- **Identify** — look up a registration by ID, or by face vector (nearest
  stored embedding within a configurable distance threshold). Works with either
  face model.
- **Web UI** — a small registration form at `/` with webcam capture (falls
  back to file upload), for exercising the API by hand.

## Two face models, enrolled together

Every registration stores **two** face embeddings computed from the same photo:

| Model | Vector size | How the client sends it | Match cutoff |
|---|---|---|---|
| **dlib** | 128 numbers | client's default (`config.yaml`) | distance < `0.6` |
| **YuNet + MobileFaceNet** | 512 numbers | client's alternate (`config.yunet.yaml`) | distance < `0.8` |

This means the matching mock **client** works out of the box whichever model it's
set to: its default is dlib, and it can switch to YuNet+MobileFaceNet by changing
one setting. On identify, the server looks at how many numbers the incoming vector
has (128 vs 512) and searches the matching gallery automatically. You don't pick a
model in the API — the vector's size does it for you.

> The two models live in **different number spaces**, so you can't compare a dlib
> vector against a MobileFaceNet one. That's fine — the server keeps them separate
> and never mixes them. It just means: whatever model the client registered with
> is the model it must identify with. Since registration enrols both, either works.

Fingerprint identification (`type=fingerprint`) is wired into the schema and
`/api/v1/identify/` endpoint for API-shape compatibility with the real server,
but there's no fingerprint capture route in this mock, so it will always
return "no match found".

## Running

**Prerequisites:** a Linux host with `sudo` and internet access on the first run (to install Podman and build the image). Nothing else to install by hand — Python 3.13 and every library live inside the container, and the ONNX models are bundled in `./models`. (macOS works too via `brew`, but needs `podman machine start` first.)

> **First build takes ~10-15 minutes.** The dlib model compiles from source inside
> the image (the Containerfile installs the C++ build tools it needs). This happens
> **once** — later rebuilds after code changes are fast because the compile is cached.
> Just let the first `./run.sh` run to completion.

```bash
./run.sh
```

`run.sh` bootstraps the toolchain: if `podman` isn't installed, it installs it via the host package manager (`dnf`/`yum` on Fedora/RHEL, `apt` on Debian/Ubuntu, also `zypper`/`pacman`/`brew`) — this needs `sudo` and network access. It's idempotent: if `podman` is already present it just builds the image and (re)starts the container (`podman build` + `podman run`, replacing any previous container of the same name). There's no compose provider or Podman API socket involved as a single container doesn't need one, so `run.sh` talks to `podman` directly. The
`Containerfile` uses standard Docker-compatible build syntax, so it also builds fine under plain Docker (`docker build` / `docker run`) if you prefer. The server listens on `http://localhost:8000`:

- `/` — registration web UI
- `/docs` — interactive API docs (Swagger UI)
- `/health` — health check

`./models` (ONNX weights) and `./config.yaml` are bind-mounted read-only;
`./data` (the SQLite DB) is bind-mounted read-write so it survives container
rebuilds/restarts.

The container image bakes in the application code, so after changing files under `app/` you need to rebuild (`./run.sh` again) to pick up the changes.

### Concurrency

`register` and `identify` both run in FastAPI's threadpool rather than on the main event loop, so concurrent requests (e.g. simulating multiple kiosks registering at once) don't block each other or `/health`. If you're benchmarking or load-testing against this mock, note that SQLite itself becomes the bottleneck under heavy concurrent writes before the app layer does. This mock isn't a substitute for load-testing against the real Postgres-backed server.

## How to use

A walkthrough once the server is up (`./run.sh`, then `curl http://localhost:8000/health` → `{"status":"ok"}`).

### 1. Register a face (web UI)

Open **http://localhost:8000** in a browser and fill the form:

| Field | Example | Rule |
|---|---|---|
| Full name | `Alice Kumar` | ≥ 5 characters |
| Date of visit | `2026-07-08` | `YYYY-MM-DD` |
| Time slot | `10:00` | `HH:MM` |
| Ticket category | `general` | ≥ 5 characters |
| Photo | webcam capture, or the file-upload fallback | a clear front-facing face |

Submit. The server detects the face, computes **both** embeddings (128-dim dlib + 512-dim MobileFaceNet), and stores the registration — you get back a `registration_id` and a message saying how many embeddings were stored. If it reports **"no face detected"**, use a clearer, front-facing photo.

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

**By face** — this route matches on a face *vector*, not an image (the server vector-searches; it does not re-embed here). The vector is either 128 numbers (dlib) or 512 numbers (MobileFaceNet); the server figures out which gallery to search from the size. Two easy ways to produce a vector from a photo:
- **Mock client** (the realistic edge-device flow): `client.py --server photo.jpg` (default dlib), or add `--config config.yunet.yaml` for the 512-dim model — see the client README, and `## Running end-to-end with the mock client` below.
- **Inside the container**: `podman exec mock-server python -m app.cli_identify /path/to/photo.jpg` — embeds the photo (with MobileFaceNet) and calls identify for you.

A match returns the person's details with `distance` (lower = better; under the model's cutoff — `0.6` for dlib, `0.8` for MobileFaceNet — counts as a match) and `confidence`.

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

- **`full_name`/`ticket_category` had no length limits; `GET /registrations/` had no `limit` cap**, now `5-200` characters each and `limit` capped at 500, both enforced with FastAPI's own `422` validation.

- **Config defaults drifted from `config.yaml`**: in-code dataclass defaults now match the shipped `config.yaml` exactly, and a missing config file now logs a `WARNING` at startup instead of silently using different values.

- **YuNet model output names weren't validated**: a mismatched ONNX export now fails fast at container startup with a clear message, instead of an obscure `KeyError` mid-request.

- **Missing model file crashed boot with a raw onnxruntime error**: `FaceEngine()` is built at import time, so if `./models` isn't bind-mounted the container died with an unhelpful `NoSuchFile` stack trace — the classic first-run mistake. Fixed: model loading now fails fast with an actionable message that names the path and points at the `./models` bind mount (and distinguishes a missing file from a present-but-corrupt one).

- **Dockerfile → Containerfile, Podman migration**: see `## Running` below.

- **compose dropped in favor of plain `podman build`/`podman run`**: this project is a single container with no inter-container networking or dependency ordering, so `podman-compose`/`podman compose` (and the Podman API socket it talks to) added a toolchain dependency without buying anything. `run.sh` now calls `podman build` and `podman run` directly.

- **Bind mounts failed under rootless Podman on SELinux-enforcing hosts (Fedora/RHEL)**: without an SELinux relabel suffix the container gets permission-denied reading `./config.yaml`/`./models` and writing `./data`. Fixed: the bind mounts in `run.sh` carry `:Z` (private relabel) on each (`:ro,Z` for the read-only ones). It's a no-op on non-SELinux hosts (plain Docker on Ubuntu), so it's safe cross-platform; switch to `:z` if you ever share a mount between containers.

## API

### `POST /api/v1/registrations/register`

`multipart/form-data`:

| field             | type | notes                                |
|-------------------|------|---------------------------------------|
| `full_name`       | str  | required, 5 - 200 characters          |
| `date_of_visit`   | str  | required, ISO date (`YYYY-MM-DD`)     |
| `timeslot`        | str  | required, ISO time (`HH:MM[:SS]`)     |
| `ticket_category` | str  | required, 5 - 200 characters          |
| `image`           | file | required, one face photo  , `image/*`, max 10MB|

Returns `{ registration_id, status, message }`. 422 if no face is detected in
the image; 400 if the date/time can't be parsed; 415 if the uploaded file's content-type isn't `image/*`; 413 if it exceeds 10MB.

Detection + embedding runs synchronously on CPU (~a few hundred ms per image depending on hardware), so a single `register` call blocks for that long. This mirrors the real server's own per-request latency for this step, it isn't mock-specific overhead.

### `GET /api/v1/registrations/` and `GET /api/v1/registrations/{id}`

List/fetch registrations, including visit details and stored vector metadata
(kind/model/dim — not the raw vector).
`limit` defaults to 100, capped at 500 (`422` if exceeded)

### `POST /api/v1/identify/`

`multipart/form-data`, `type` selects the lookup mode:

- `type=id`, `id=<registration_id>` — direct lookup.
- `type=face`, `face_vector=<JSON array or comma-separated floats>` — nearest
  stored face embedding. The vector must be either **128-dim** (dlib, matched
  within `identify.dlib_face_recognition_threshold`) or **512-dim** (MobileFaceNet,
  matched within `identify.face_recognition_threshold`); any other length returns a
  `400` error naming the two accepted sizes.
- `type=fingerprint` — accepted but always returns "no match found" (see
  above). Vectors longer than 4096 elements are rejected with `400`
  regardless of type.

Optional `vector_type` (`registration` / `fru` / `sau`) filters candidates by
source. Returns an `IdentifyResponse` with `match_type`, `distance`,
`confidence`, and the matched registration's fields when found.

400 if `face_vector` / `fingerprint_vector` isn't a valid JSON or comma separated floating-point values, or is empty. 

## Configuration

`config.yaml` (bind-mounted, read on startup — see [app/core/config.py](app/core/config.py)):

```yaml
server:
  host: "0.0.0.0"
  port: 8000

database:
  path: "/app/data/db.sqlite"

models:
  face_detector_path: "/app/models/face_detection_yunet_2023mar.onnx"
  face_recognizer_path: "/app/models/mobilefacenet.onnx"
  face_detector_input_size: 640
  face_detector_score_threshold: 0.5

identify:
  face_recognition_threshold: 0.8        # MobileFaceNet (512-dim) match cutoff
  dlib_face_recognition_threshold: 0.6   # dlib (128-dim) match cutoff
  fingerprint_recognition_threshold: 0.7
  default_n: 10
```

The YuNet + MobileFaceNet models are bind-mounted from `./models`. The **dlib**
model has no path here on purpose: its weights ship inside the
`face_recognition_models` Python package (installed into the image), not in
`./models`, so there's nothing extra to mount.

`config.yaml` is the single source of truth for all settings. If it's missing or unreadable at startup (e.g. `CONFIG_PATH` misconfigured, bind mount missing), the server logs a `WARNING` and falls back to the built-in defaults in `app/core/config.py`, and those defaults are kept in sync with the shipped `config.yaml` and covered by a test (`tests/test_config.py`), but if you ever
see that warning in the logs, something is wrong with your bind mount, not your config values.

## Face pipeline

[app/core/face_engine.py](app/core/face_engine.py) mirrors the real server's
**two** identification backends, and registration runs both on the same photo:

- **MobileFaceNet (`FaceEngine`)** — YuNet (ONNX) detects a face and 5 landmarks,
  the face is aligned into a 112x112 ArcFace pose, and MobileFaceNet (ONNX)
  produces a normalized 512-dim embedding. Both models run via `onnxruntime` (CPU).
- **dlib (`DlibEngine`)** — dlib's HOG detector finds the face, a 5-point shape
  predictor aligns it, and dlib's ResNet produces a raw (not normalized) 128-dim
  descriptor. These weights ship inside the `face_recognition_models` package, so
  dlib is compiled into the image (see the first-build note under `## Running`).
  This is byte-for-byte the same pipeline the mock client uses for its default
  dlib model, so the vectors match exactly.

If the dlib packages somehow aren't installed, the server still boots and
registers — it just stores only the MobileFaceNet embedding and logs a warning.

The detector is expected to be a YuNet export with per-stride outputs named `cls_{8,16,32}`, `obj_{8,16,32}`, `bbox_{8,16,32}`, `kps_{8,16,32}`. This is validated at startup. If you swap in a different YuNet export (e.g. a differently-converted ONNX file) and the container fails to start with a `RuntimeError` mentioning "missing expected output tensor", that's this check
—point `models.face_detector_path` in `config.yaml` at a model exported with the standard YuNet output naming.

## Utilities

- `./query_db.sh` — open a `sqlite3` shell on `data/db.sqlite`.
- `python -m app.cli_identify photo.jpg` (run inside the container) — embeds a
  photo and calls `/api/v1/identify/` with it, like a real edge device would.

## Project layout

```
app/
  core/
    config.py       # loads config.yaml into typed Settings
    database.py      # SQLite schema + queries
    face_engine.py    # FaceEngine (YuNet+MobileFaceNet) + DlibEngine (dlib)
  routers/
    registrations.py # register / list / get
    identify.py       # id / face / fingerprint lookup
  schemas/            # pydantic request/response models
  static/             # registration web UI (HTML/CSS/JS)
  main.py             # FastAPI app, routes, static mount
models/                # ONNX weights (bind-mounted)
data/                  # SQLite DB (bind-mounted, gitignored)
config.yaml
Containerfile
run.sh
query_db.sh
```

## Running end-to-end with the mock client

The full loop on one machine — mock server + mock client — in four steps. Assumes the two repos sit side by side (`am-mock-server/` and `am-mock-client/`).

**1. Start the server** (this repo):
```bash
cd am-mock-server
./run.sh                                    # installs Podman if needed, builds, starts
curl http://localhost:8000/health           # -> {"status":"ok"}
```

**2. Register a face** via the web UI at **http://localhost:8000** (see `## How to use` above). Note the name you used.

**3. Set up the client** (`am-mock-client`):
```bash
cd ../am-mock-client
./setup.sh                                   # native venv; default includes dlib (~10-15 min compile)
```

**4. Identify** with a photo of the same person:
```bash
.venv/bin/python client.py --server <photo.jpg>
# -> >>> Recognised: <name>     (distance comfortably under the match cutoff)
```

That uses the client's **default dlib model**, which now works against this mock
because registration enrolled a dlib vector too. To use the 512-dim
YuNet+MobileFaceNet model instead, add `--config config.yunet.yaml`:
```bash
.venv/bin/python client.py --config config.yunet.yaml --server <photo.jpg>
```
Both talk to this same server on `localhost:8000`. Full client docs:
`am-mock-client/README.md`.