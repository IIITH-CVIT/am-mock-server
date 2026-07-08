import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def client():
    """FastAPI TestClient for the mock server.

    app.main is imported lazily (it builds FaceEngine at import time, which needs
    the ONNX models), so tests that don't need HTTP don't pay for it. Using the
    context manager runs the startup event (init_db) and shutdown cleanly.
    """
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client