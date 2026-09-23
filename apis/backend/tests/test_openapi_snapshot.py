import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_openapi.py"
_spec = importlib.util.spec_from_file_location("export_openapi", _SCRIPT)
export_openapi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_openapi)


def test_the_frontends_api_contract_is_current():
    # apps/frontend generates its API types from this snapshot; a response model change must reach it.
    assert export_openapi.SNAPSHOT.read_text(encoding="utf-8") == export_openapi.schema_text(), (
        "apps/frontend/app/types/openapi.json is stale: run `uv run python apis/backend/scripts/export_openapi.py`, "
        "then `pnpm gen:api-types` in apps/frontend"
    )
