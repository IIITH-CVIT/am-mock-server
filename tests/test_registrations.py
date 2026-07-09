import asyncio
import time 
import pytest 
from httpx import AsyncClient, ASGITransport 

@pytest.mark.anyio 
async def test_register_does_not_block_event_loop(monkeypatch):
    from app.core import face_engine as fe_module 
    from app.main import app 

    def slow_embed(image_bytes):
        time.sleep(1.0)
        return [0.1] * 512

    monkeypatch.setattr(fe_module.face_engine, "embed", slow_embed)

    transport = ASGITransport(app = app)

    async with AsyncClient(transport = transport, base_url = "http://test") as client:
        async def do_register():
            return await client.post(
                "/api/v1/registrations/register",
                data={"full_name": "Tester", "date_of_visit": "2026-07-01",
                      "timeslot": "10:00", "ticket_category": "general"},
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

def test_register_stores_both_embeddings_and_both_identify(client, monkeypatch):
    """End-to-end: one registration enrolls BOTH a mobilefacenet (512) and a dlib
    (128) vector, and either can then identify the person."""
    import json

    from app.core import face_engine as fe_module
    from app.routers import registrations as reg_module

    # Non-constant vectors: other tests enroll a constant [0.1]*512, and any two
    # constant vectors normalize to the same unit vector — a distinctive pattern
    # avoids a spurious tie in the shared test DB.
    mobile_vec = [float((i % 13) + 1) for i in range(512)]
    dlib_vec = [float((i % 5) - 2) for i in range(128)]

    # Real mobilefacenet engine is stubbed; a fake dlib engine stands in for the
    # (uncompiled-here) real one so the dual-enroll path runs exactly as it would.
    class _FakeDlib:
        model_name = "dlib"

        def embed(self, image_bytes):
            return list(dlib_vec)

    monkeypatch.setattr(fe_module.face_engine, "embed", lambda b: list(mobile_vec))
    monkeypatch.setattr(reg_module, "dlib_engine", _FakeDlib())

    resp = client.post(
        "/api/v1/registrations/register",
        data={"full_name": "Alice Kumar", "date_of_visit": "2026-07-01",
              "timeslot": "10:00", "ticket_category": "general"},
        files={"image": ("f.jpg", b"fake-bytes", "image/jpeg")},
    )
    assert resp.status_code == 200
    reg_id = resp.json()["registration_id"]

    # Both vectors are enrolled under the one registration.
    detail = client.get(f"/api/v1/registrations/{reg_id}").json()
    stored = {(v["model"], v["dim"]) for v in detail["vectors"]}
    assert ("mobilefacenet", 512) in stored
    assert ("dlib", 128) in stored

    # The dlib (128-dim) query matches.
    r_dlib = client.post("/api/v1/identify/", data={
        "type": "face", "face_vector": json.dumps(dlib_vec)})
    assert r_dlib.status_code == 200
    assert r_dlib.json()["name"] == "Alice Kumar"

    # The mobilefacenet (512-dim) query matches too.
    r_mobile = client.post("/api/v1/identify/", data={
        "type": "face", "face_vector": json.dumps(mobile_vec)})
    assert r_mobile.status_code == 200
    assert r_mobile.json()["name"] == "Alice Kumar"


def test_register_rejects_oversized_full_name(client):
    resp = client.post(
        "/api/v1/registrations/register",
        data={
            "full_name": "A" * 201,
            "date_of_visit": "2026-07-01",
            "timeslot": "10:00",
            "ticket_category": "general",
        },
        files={"image": ("f.jpg", b"fake-bytes", "image/jpeg")},
    )
    assert resp.status_code == 422

def test_register_rejects_empty_full_name(client):
    resp = client.post(
        "/api/v1/registrations/register",
        data={"full_name": "", "date_of_visit": "2026-07-01",
              "timeslot": "10:00", "ticket_category": "general"},
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
        data={"full_name": "Tester", "date_of_visit": "2026-07-01",
              "timeslot": "10:00", "ticket_category": "general"},
        files={"image": ("f.txt", b"not an image", "text/plain")},
    )
    assert resp.status_code == 415

def test_register_rejects_oversized_image(client):
    huge = b"\x00" * (10 * 1024 * 1024 + 1)
    resp = client.post(
        "/api/v1/registrations/register",
        data={"full_name": "Tester", "date_of_visit": "2026-07-01",
              "timeslot": "10:00", "ticket_category": "general"},
        files={"image": ("f.jpg", huge, "image/jpeg")},
    )
    assert resp.status_code == 413

def test_register_accepts_image_at_exact_limit(client, monkeypatch):
    # confirms the +1 boundary math doesn't off-by-one reject legitimate uploads
    from app.core import face_engine as fe_module
    monkeypatch.setattr(fe_module.face_engine, "embed", lambda b: [0.1] * 512)
    exactly_at_limit = b"\x00" * (10 * 1024 * 1024)
    resp = client.post(
        "/api/v1/registrations/register",
        data={"full_name": "Tester", "date_of_visit": "2026-07-01",
              "timeslot": "10:00", "ticket_category": "general"},
        files={"image": ("f.jpg", exactly_at_limit, "image/jpeg")},
    )
    assert resp.status_code != 413