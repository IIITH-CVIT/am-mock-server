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

## API

### `POST /api/v1/registrations/register`

`multipart/form-data`:

| field             | type | notes                                |
|-------------------|------|---------------------------------------|
| `full_name`       | str  | required                              |
| `date_of_visit`   | str  | required, ISO date (`YYYY-MM-DD`)     |
| `timeslot`        | str  | required, ISO time (`HH:MM[:SS]`)     |
| `ticket_category` | str  | required, free text                   |
| `image`           | file | required, one face photo              |

Returns `{ registration_id, status, message }`. 422 if no face is detected in
the image; 400 if the date/time can't be parsed.

### `GET /api/v1/registrations/` and `GET /api/v1/registrations/{id}`

List/fetch registrations, including visit details and stored vector metadata
(kind/model/dim — not the raw vector).

### `POST /api/v1/identify/`

`multipart/form-data`, `type` selects the lookup mode:

- `type=id`, `id=<registration_id>` — direct lookup.
- `type=face`, `face_vector=<JSON array or comma-separated floats>` — nearest
  stored face embedding (dimension follows `models.embedder_model`: 128-dim for
  sface, 512-dim for auraface) within `identify.face_recognition_threshold`.
- `type=fingerprint` — accepted but always returns "no match found" (see
  above).

Optional `vector_type` (`registration` / `fru` / `sau`) filters candidates by
source. Returns an `IdentifyResponse` with `match_type`, `distance`,
`confidence`, and the matched registration's fields when found.

## Configuration

`config.yaml` (bind-mounted, read on startup — see [app/core/config.py](app/core/config.py)):

```yaml
server:
  host: "0.0.0.0"
  port: 8000

database:
  path: "/app/data/db.sqlite"

models:
  face_detector_path: "/app/models/face_detection_yunet_2026may.onnx"
  face_recognizer_path: "/app/models/face_recognition_sface_2021dec.onnx"
  face_recognizer_auraface_path: "/app/models/aurar100.onnx"
  embedder_model: "sface"   # sface (128-dim, native cv2) | auraface (512-dim, onnxruntime)
  face_detector_input_size: 640
  face_detector_score_threshold: 0.5

identify:
  face_recognition_threshold: 0.8
  fingerprint_recognition_threshold: 0.7
  default_n: 10
```

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
- `python -m app.cli_identify photo.jpg` (run inside the container) — embeds a
  photo and calls `/api/v1/identify/` with it, like a real edge device would.

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
models/                # ONNX weights (bind-mounted)
data/                  # SQLite DB (bind-mounted, gitignored)
config.yaml
compose.yml
Dockerfile
run.sh
query_db.sh
```
