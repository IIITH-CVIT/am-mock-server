"""Entrypoint for `python -m app` and the PyInstaller build target.

Calls uvicorn.run() with the app object directly (not the "app.main:app"
import string) so a frozen binary doesn't need uvicorn's import machinery to
resolve the module by name at runtime.
"""

import uvicorn

from app.core.config import settings
from app.main import app


def main() -> None:
    uvicorn.run(app, host=settings.server.host, port=settings.server.port)


if __name__ == "__main__":
    main()
