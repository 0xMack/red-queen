"""Writes the backend's OpenAPI schema to apps/frontend/app/types/openapi.json -- the checked-in contract the
frontend's types are generated from (`pnpm gen:api-types`). tests/test_openapi_snapshot.py fails until you
re-run this after changing a response model.

Run with: uv run python apis/backend/scripts/export_openapi.py
"""

import json
from pathlib import Path

from backend.main import app

SNAPSHOT = Path(__file__).resolve().parents[3] / "apps" / "frontend" / "app" / "types" / "openapi.json"


def schema_text() -> str:
    return json.dumps(app.openapi(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    SNAPSHOT.write_text(schema_text(), encoding="utf-8", newline="\n")
    print(f"wrote {SNAPSHOT}")
