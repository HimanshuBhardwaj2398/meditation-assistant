"""App factory for the corpus API.

Run against the database in ``.env`` (``DB_URL``)::

    poetry run uvicorn api.main:app --reload

then open http://127.0.0.1:8000/workbench/ for the reader or /docs for Swagger.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.routes import documents
from db.database import Database

_WORKBENCH_DIR = Path(__file__).resolve().parents[1] / "console"


def create_app(database: Database | None = None) -> FastAPI:
    """Build the app.

    ``database`` is the injection seam: tests pass a SQLite-backed ``Database``;
    ``None`` means the process default from settings, resolved on the first
    request rather than at import, so importing this module never connects.
    """
    app = FastAPI(title="Meditation corpus API", version="0.1.0")
    app.state.database = database
    app.include_router(documents.router)
    app.mount("/workbench", StaticFiles(directory=_WORKBENCH_DIR, html=True), name="workbench")
    return app


app = create_app()
