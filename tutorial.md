# Mock AM Server Tutorial

For the full business logic, API reference, and troubleshooting, see [README.md](README.md).

---

## What this is

A local FastAPI mock of the face registration/identify API. It detects a face in a photo, stores its embedding, and lets you look people up by ID or by face. Data lives in a local SQLite file. No Postgres, Qdrant, or S3 needed.

---

## 1. Prerequisites

- A Linux host with `sudo` and internet access **on the first run** (to install Podman and build the image).
- Nothing else to install by hand. Python and all libraries live inside the
container, and the ONNX models are already bundled in `./models`.

---

## 2. Start the server

```bash
./run.sh
```

This installs Podman if it's missing, builds the image, and starts the container on **http://localhost:8000**.

> **The first build takes ~10-15 minutes** as dlib compiles from source inside the image. 
> This happens only once; later rebuilds are fast. Just let it finish.

---

## 3. Verify it's up

```bash
curl http://localhost:8000/health
# -> {"status":"ok"}
```

Once healthy, three things are available:

| URL | What |
|---|---|
| http://localhost:8000 | Registration web UI |
| http://localhost:8000/docs | Interactive API docs (Swagger UI) |
| http://localhost:8000/health | Health check |

---

## 4. Register a face

Open **http://localhost:8000** and fill the form:

| Field | Example | Rule |
|---|---|---|
| Full name | `Alice Kumar` | ≥ 5 characters |
| Date of visit | `2026-07-08` | `YYYY-MM-DD` |
| Time slot | `10:00` | `HH:MM` |
| Ticket category | `general` | pick from dropdown |
| Photo | webcam capture, or file upload | a clear, front-facing face |

Submit. You get back a `registration_id`. If it says **"no face detected"**, use a clearer, front-facing photo.

---

## 5. Confirm it was stored

```bash
curl http://localhost:8000/api/v1/registrations/     # add "| jq" if you have it
```

---

## 6. Identify

**By registration ID** (exact lookup, no photo):

```bash
curl -X POST http://localhost:8000/api/v1/identify/ -F "type=id" -F "id=<registration_id>"
```

**By face** — the identify route matches on a face *vector*, not an image. The easiest way to produce one from a photo, right inside the container:

```bash
podman exec mock-server python -m app.cli_identify /path/to/photo.jpg
```

A match returns the person's details with a `distance` (lower = better) and
`confidence`.

> For the realistic edge-device flow (producing the vector on a separate client), see the `am-mock-client` repo and its `tutorial.md`.

---

## 7. Inspect the database (optional)

```bash
./query_db.sh
sqlite> SELECT id, full_name, date_of_visit FROM registrations;
```

---

## After code changes

The image bakes in the app code, so after editing anything under `app/`, rerun `./run.sh` to rebuild and pick up the changes. Your data in `./data` survives every rebuild.

---

That's it, and you're running. For everything else (full API, config options, the face pipeline, known issues), see [README.md](README.md).
