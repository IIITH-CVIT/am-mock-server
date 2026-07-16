import logging
import sys
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.core.database import init_db
from app.routers import identify, registrations

logger = logging.getLogger("mock_server")

# Resolve relative to this file (or PyInstaller's extraction dir when frozen)
# rather than the process's cwd, so static assets are found regardless of
# where the app/binary is launched from.
BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
STATIC_DIR = BASE_DIR / "app" / "static"

app = FastAPI(title="Mock AM Server", version="0.1.0")

app.include_router(registrations.router)
app.include_router(identify.router)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "internal server error"})

@app.get("/", tags=["registrations"], include_in_schema=False)
def registration_page() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/health", tags=["health"])
def health() -> dict:
    return {"status": "ok"}
