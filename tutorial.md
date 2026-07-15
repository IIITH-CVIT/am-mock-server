# Mock AM Server Tutorial

For the full business logic, API reference, and troubleshooting, see [README.md](README.md).

---

## What this is

A local FastAPI mock of face registration. It detects a face in a photo, stores its embedding, and lets you look people up by ID or by face. Data lives in a local SQLite file. 

---

## 1. Start the server
Go to the terminal and run the following command: 
```bash
./run.sh
```

This installs Podman if it's missing, builds the image, and starts the container on **http://localhost:8000**. 
You can access the UI by typing this URL in the search bar. It is recommended to use Firefox browser.

---

## 2. Verify it's up
To check whether the application is healthy, type out the following URLs in the browser's search bar.

| URL | What |
|---|---|
| http://localhost:8000 | Registration web UI |
| http://localhost:8000/docs | Interactive API docs (Swagger UI) |
| http://localhost:8000/health | Health check |

---

## 3. Register a face

Open **http://localhost:8000** in the broswer and fill the form:

| Field | Example | Rule |
|---|---|---|
| Full name | `Alice Kumar` | ≥ 5 characters |
| Date of visit | `2026-07-08` | `YYYY-MM-DD` |
| Time slot | `10:00` | `HH:MM` |
| Ticket category | `general` | pick from dropdown |
| Photo | webcam capture, or file upload | a clear, front-facing face |

Submit. You get back a `registration_id`. If it says **"no face detected"**, use a clearer, front-facing photo.

---

## 4. Confirm it was stored
Check `data/db.sqlite` to see the registrations. (Install SQLite Viewer in VSCode extensions) 

---

## 5. Identify

**By registration ID** (exact lookup, no photo):
Go to the terminal and run the following command: 
```bash
curl -X POST http://localhost:8000/api/v1/identify/ -F "type=id" -F "id=<registration_id>"
```

---

That's it, and you're running. For everything else (full API, config options, the face pipeline, known issues), see [README.md](README.md).
