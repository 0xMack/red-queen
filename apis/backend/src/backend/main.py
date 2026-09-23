from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import leaderboards, models, runs

app = FastAPI(title="red-queen backend", version="0.1.0")

# Local single-user tool (docs/design/0005: auth/deployment out of scope) -- apps/frontend runs on
# a different dev port with nothing to authenticate, so open CORS is the actual right answer here,
# not a placeholder to tighten later.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(runs.router)
app.include_router(leaderboards.router)
app.include_router(models.router)
app.include_router(models.catalog_router)
app.include_router(models.export_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
