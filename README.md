# Mock AM Server

A lightweight FastAPI mock of the `am-master-server` face registration/identify
API, for local development and testing without the full stack (Postgres,
Celery, S3/MinIO, Qdrant). Data is stored in a local SQLite file instead.

## Features

- **Registration** — capture a name, visit date, time slot, ticket category,
  and one face photo. The face is embedded on the spot; only the name, visit
  details, and embedding are persisted (the image bytes are discarded).
- **Identify** — look up a registration by ID, or by face vector (nearest
  stored embedding within a configurable distance threshold).
- **Web UI** — a small registration form at `/` with webcam capture (falls
  back to file upload), for exercising the API by hand.

Fingerprint identification (`type=fingerprint`) is wired into the schema and
`/api/v1/identify/` endpoint for API-shape compatibility with the real server,
but there's no fingerprint capture route in this mock, so it will always
return "no match found".

## Running

```bash
./run.sh
```

This runs `podman compose up --build` (Podman; the image also builds fine under plain Docker if you don't have Podman installed. Swap `podman compose` for `docker compose` in `run.sh` and it works identically, since `Containerfile` uses standard Docker-compatible build syntax). The
server listens on `http://localhost:8000`:

- `/` — registration web UI
- `/docs` — interactive API docs (Swagger UI)
- `/health` — health check

`./models` (ONNX weights) and `./config.yaml` are bind-mounted read-only;
`./data` (the SQLite DB) is bind-mounted read-write so it survives container
rebuilds/restarts.

The Docker image bakes in the application code, so after changing files under
`app/` you need to rebuild (`./run.sh` again) to pick up the changes.

### Concurrency

`register` and `identify` both run in FastAPI's threadpool rather than on the main event loop, so concurrent requests (e.g. simulating multiple kiosks registering at once) don't block each other or `/health`. If you're benchmarking or load-testing against this mock, note that SQLite itself becomes the bottleneck under heavy concurrent writes before the app layer does. This mock isn't a substitute for load-testing against the real Postgres-backed server.

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

- **Bind mounts failed under rootless Podman on SELinux-enforcing hosts (Fedora/RHEL)**: without an SELinux relabel suffix the container gets permission-denied reading `./config.yaml`/`./models` and writing `./data`. Fixed: `compose.yml` now carries `:Z` (private relabel) on each bind mount (`:ro,Z` for the read-only ones). It's a no-op on non-SELinux hosts (plain Docker on Ubuntu), so it's safe cross-platform; switch to `:z` if you ever share a mount between containers.

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
  stored face embedding (512-dim, MobileFaceNet) within
  `identify.face_recognition_threshold`. Must exactly be 512-dimensions, else a `400` error is thrown
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
  face_recognition_threshold: 0.8
  fingerprint_recognition_threshold: 0.7
  default_n: 10
```

`config.yaml` is the single source of truth for all settings. If it's missing or unreadable at startup (e.g. `CONFIG_PATH` misconfigured, bind mount missing), the server logs a `WARNING` and falls back to the built-in defaults in `app/core/config.py`, and those defaults are kept in sync with the shipped `config.yaml` and covered by a test (`tests/test_config.py`), but if you ever
see that warning in the logs, something is wrong with your bind mount, not your config values.

## Face pipeline

[app/core/face_engine.py](app/core/face_engine.py) mirrors the real server's
pipeline: YuNet (ONNX) detects a face and 5 landmarks, the face is aligned into
a 112x112 ArcFace pose, and MobileFaceNet (ONNX) produces a normalized 512-dim
embedding. Both models run via `onnxruntime` (CPU).

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
    face_engine.py    # YuNet detection + MobileFaceNet embedding
  routers/
    registrations.py # register / list / get
    identify.py       # id / face / fingerprint lookup
  schemas/            # pydantic request/response models
  static/             # registration web UI (HTML/CSS/JS)
  main.py             # FastAPI app, routes, static mount
models/                # ONNX weights (bind-mounted)
data/                  # SQLite DB (bind-mounted, gitignored)
config.yaml
compose.yml
Dockerfile
run.sh
query_db.sh
```

## Running end-to-end with the mock client

1. Start this server: `podman compose up --build` (see `## Running` above). Confirm: `curl http://localhost:8000/health`

2. Register at least one face via the web UI at `http://localhost:8000`.

3. In `iiith-cvit-am-mock-client`, create `config.mock-server.yaml` (copy of `config.yaml` with `server.url: http://localhost:8000`, `detection.detector: yunet`, `embedder.model: mobilefacenet`).

4. `.venv/bin/python client.py --config config.mock-server.yaml --server <photo>` — should print `Recognised: <name>` for a photo of someone you registered in step 2, with `distance` comfortably under `0.8`.

See `iiith-cvit-am-mock-client/README.md` for full client setup.