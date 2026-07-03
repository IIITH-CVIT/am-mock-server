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

    monkeypatch.setattr(fe_module.face_engine, "embed". slow_embed)

    transport = ASGITransport(app = app)

    async with AsyncClient(transport = transport, base_url = "http://test") as client:
        async def do_register():
            return await client.post(
                data={"full_name": "Test", "date_of_visit": "2026-07-01",
                      "timeslot": "10:00", "ticket_category": "general"},
                files={"image": ("f.jpg", b"fake-bytes", "image/jpeg")},
            )
        
        async def do_health():
            start = time.monotonic()
            resp = await client.get("/health")
            return resp, time.monotonic() - start
        
        resgister_task = asyncio.create_task(do_register())
        await asyncio.sleep(0.1)

        health_resp, health_elapsed = await do_health 
        await register_task 

    assert health_resp.status_code == 200

    assert health_elapsed < 0.5, (
        f"/health took {health_elapsed:.2f}s - register() is blocking the event loop"
    )