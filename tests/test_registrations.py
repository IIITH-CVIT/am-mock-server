import asyncio
import time

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.mark.anyio
async def test_register_does_not_block_event_loop(monkeypatch):
    from app.core import face_engine as fe_module
    from app.main import app

    def slow_embed(image_bytes):
        time.sleep(1.0)
        return [0.1] * 128

    monkeypatch.setattr(fe_module.face_engine, "embed", slow_embed)

    transport = ASGITransport(app = app)

    async with AsyncClient(transport = transport, base_url = "http://test") as client:
        async def do_register():
            return await client.post(
                "/api/v1/registrations/register",
                data={"full_name": "Tester"},
                files={"image": ("f.jpg", b"fake-bytes", "image/jpeg")},
            )

        async def do_health():
            start = time.monotonic()
            resp = await client.get("/health")
            return resp, time.monotonic() - start

        register_task = asyncio.create_task(do_register())
        await asyncio.sleep(0.1)

        health_resp, health_elapsed = await do_health()
        await register_task

    assert health_resp.status_code == 200

    assert health_elapsed < 0.5, (
        f"/health took {health_elapsed:.2f}s - register() is blocking the event loop"
    )

def test_register_stores_embedding_and_identify_matches(client, monkeypatch):
    """End-to-end: one registration enrolls a single (sface, 128-dim) vector,
    which can then be used to identify the person."""
    import json

    from app.core import face_engine as fe_module

    # Non-constant vector: other tests enroll a constant [0.1]*128, and any two
    # constant vectors normalize to the same unit vector — a distinctive pattern
    # avoids a spurious tie in the shared test DB.
    face_vec = [float((i % 13) + 1) for i in range(128)]

    monkeypatch.setattr(fe_module.face_engine, "embed", lambda b: list(face_vec))

    resp = client.post(
        "/api/v1/registrations/register",
        data={"full_name": "Alice Kumar"},
        files={"image": ("f.jpg", b"fake-bytes", "image/jpeg")},
    )
    assert resp.status_code == 200
    reg_id = resp.json()["registration_id"]

    detail = client.get(f"/api/v1/registrations/{reg_id}").json()
    stored = {(v["model"], v["dim"]) for v in detail["vectors"]}
    assert ("sface", 128) in stored

    r = client.post("/api/v1/identify/", data={
        "type": "face", "face_vector": json.dumps(face_vec)})
    assert r.status_code == 200
    assert r.json()["name"] == "Alice Kumar"


def test_register_stamps_default_visit_details(client, monkeypatch):
    """date_of_visit/timeslot/ticket_category are no longer client input — the
    server always stamps today's date and the DEFAULT_TIMESLOT/DEFAULT_TICKET_CATEGORY
    constants from app/routers/registrations.py."""
    from datetime import date

    from app.core import face_engine as fe_module
    from app.routers.registrations import DEFAULT_TICKET_CATEGORY, DEFAULT_TIMESLOT

    monkeypatch.setattr(fe_module.face_engine, "embed", lambda b: [0.1] * 128)

    resp = client.post(
        "/api/v1/registrations/register",
        data={"full_name": "Bob Stamped"},
        files={"image": ("f.jpg", b"fake-bytes", "image/jpeg")},
    )
    assert resp.status_code == 200
    reg_id = resp.json()["registration_id"]

    detail = client.get(f"/api/v1/registrations/{reg_id}").json()
    assert detail["date_of_visit"] == date.today().isoformat()
    assert detail["timeslot"] == DEFAULT_TIMESLOT
    assert detail["ticket_category"] == DEFAULT_TICKET_CATEGORY


def test_register_rejects_oversized_full_name(client):
    resp = client.post(
        "/api/v1/registrations/register",
        data={"full_name": "A" * 201},
        files={"image": ("f.jpg", b"fake-bytes", "image/jpeg")},
    )
    assert resp.status_code == 422

def test_register_rejects_empty_full_name(client):
    resp = client.post(
        "/api/v1/registrations/register",
        data={"full_name": ""},
        files={"image": ("f.jpg", b"fake-bytes", "image/jpeg")},
    )
    assert resp.status_code == 422

def test_list_registrations_caps_limit(client):
    resp = client.get("/api/v1/registrations/?limit=999999")
    assert resp.status_code == 422

def test_list_registrations_default_limit_ok(client):
    resp = client.get("/api/v1/registrations/")
    assert resp.status_code == 200

def test_register_rejects_non_image_content_type(client):
    resp = client.post(
        "/api/v1/registrations/register",
        data={"full_name": "Tester"},
        files={"image": ("f.txt", b"not an image", "text/plain")},
    )
    assert resp.status_code == 415

def test_register_rejects_oversized_image(client):
    huge = b"\x00" * (10 * 1024 * 1024 + 1)
    resp = client.post(
        "/api/v1/registrations/register",
        data={"full_name": "Tester"},
        files={"image": ("f.jpg", huge, "image/jpeg")},
    )
    assert resp.status_code == 413

def test_register_accepts_image_at_exact_limit(client, monkeypatch):
    # confirms the +1 boundary math doesn't off-by-one reject legitimate uploads
    from app.core import face_engine as fe_module
    monkeypatch.setattr(fe_module.face_engine, "embed", lambda b: [0.1] * 128)
    exactly_at_limit = b"\x00" * (10 * 1024 * 1024)
    resp = client.post(
        "/api/v1/registrations/register",
        data={"full_name": "Tester"},
        files={"image": ("f.jpg", exactly_at_limit, "image/jpeg")},
    )
    assert resp.status_code != 413
