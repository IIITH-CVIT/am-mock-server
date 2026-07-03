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

This runs `docker compose up --build`. The server listens on
`http://localhost:8000`:

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

## API

### `POST /api/v1/registrations/register`

`multipart/form-data`:

| field             | type | notes                                |
|-------------------|------|---------------------------------------|
| `full_name`       | str  | required, 1 - 200 characters          |
| `date_of_visit`   | str  | required, ISO date (`YYYY-MM-DD`)     |
| `timeslot`        | str  | required, ISO time (`HH:MM[:SS]`)     |
| `ticket_category` | str  | required, 1 - 100 characters          |
| `image`           | file | required, one face photo              |

Returns `{ registration_id, status, message }`. 422 if no face is detected in
the image; 400 if the date/time can't be parsed.

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
  `identify.face_recognition_threshold`.
- `type=fingerprint` — accepted but always returns "no match found" (see
  above).

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
