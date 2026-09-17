import pytest
from fastapi.testclient import TestClient

from backend.dependencies import _game_session_store
from backend.game_sessions import GameSessionStore
from backend.main import app


@pytest.fixture
def client():
    # One store per test, reused across every request in it (the lambda must close over a single
    # instance -- a fresh GameSessionStore() per call would make a session created in one request
    # invisible to the next). Without this override, _game_session_store's @lru_cache would instead
    # share one store across every test for the life of the pytest process.
    store = GameSessionStore()
    app.dependency_overrides[_game_session_store] = lambda: store
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def create(client: TestClient, game: str = "snake", seed: int | None = 0) -> dict:
    resp = client.post(f"/games/{game}/sessions", json={"game": game, "seed": seed})
    assert resp.status_code == 200
    return resp.json()


def test_create_session_returns_initial_render_state(client):
    state = create(client)

    assert state["game"] == "snake"
    assert state["done"] is False
    assert state["step"] == 0
    assert state["reward"] == 0.0
    render_state = state["render_state"]
    assert render_state["width"] == 10
    assert render_state["height"] == 10
    assert render_state["alive"] is True
    # cells must be JSON-safe: a list of {x, y, label}, not a dict with tuple keys.
    assert isinstance(render_state["cells"], list)
    assert {"x", "y", "label"} <= render_state["cells"][0].keys()
    labels = {cell["label"] for cell in render_state["cells"]}
    assert labels == {"head", "body", "food"}


def test_create_session_unknown_game_404(client):
    resp = client.post("/games/no-such-game/sessions", json={"game": "no-such-game"})
    assert resp.status_code == 404


def test_create_session_mismatched_game_400(client):
    resp = client.post("/games/snake/sessions", json={"game": "reach1d"})
    assert resp.status_code == 400


def test_apply_action_updates_state(client):
    state = create(client)
    session_id = state["session_id"]

    resp = client.post(
        f"/games/snake/sessions/{session_id}/actions", json={"action": 0.0}
    )
    assert resp.status_code == 200
    next_state = resp.json()
    assert next_state["step"] == 1
    assert next_state["session_id"] == session_id


def test_apply_action_unknown_session_404(client):
    resp = client.post(
        "/games/snake/sessions/no-such-session/actions", json={"action": 0.0}
    )
    assert resp.status_code == 404


def test_apply_action_after_done_409(client):
    # a seed/action combo that runs off the edge is all that's needed here -- drive straight off
    # the right edge of the default 10x10 board.
    state = create(client, seed=0)
    session_id = state["session_id"]

    done = False
    for _ in range(30):
        resp = client.post(
            f"/games/snake/sessions/{session_id}/actions", json={"action": 0.0}
        )
        done = resp.json()["done"]
        if done:
            break
    assert done, (
        "expected the snake to die by running straight for 30 steps on a 10x10 board"
    )

    resp = client.post(
        f"/games/snake/sessions/{session_id}/actions", json={"action": 0.0}
    )
    assert resp.status_code == 409


def test_trajectory_records_every_step(client):
    state = create(client)
    session_id = state["session_id"]

    for _ in range(3):
        client.post(f"/games/snake/sessions/{session_id}/actions", json={"action": 0.0})

    resp = client.get(f"/games/snake/sessions/{session_id}/trajectory")
    assert resp.status_code == 200
    trajectory = resp.json()
    assert trajectory["run_id"] is None
    assert trajectory["game"] == "snake"
    assert len(trajectory["states"]) == 4  # initial reset state + 3 actions
    assert len(trajectory["actions"]) == 3
    assert len(trajectory["rewards"]) == 3


def test_trajectory_unknown_session_404(client):
    resp = client.get("/games/snake/sessions/no-such-session/trajectory")
    assert resp.status_code == 404
