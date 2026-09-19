import pytest
from fastapi.testclient import TestClient
from telemetry import EvaluationRecord, SqliteEvaluationStore

from backend.dependencies import _evaluation_store
from backend.main import app


def _record(entrant_id: str, mean: float, protocol: str = "snake.score.v1") -> EvaluationRecord:
    return EvaluationRecord(
        game="snake",
        protocol=protocol,
        entrant_id=entrant_id,
        entrant_kind="baseline",
        label=entrant_id,
        interface="snake/features.v1+relative3.v1",
        created_at=1.0,
        metrics={"quality": {"mean": mean}},
    )


@pytest.fixture
def client(tmp_path):
    store = SqliteEvaluationStore(tmp_path / "evaluations.db")
    store.put(_record("baseline:random", 0.1))
    store.put(_record("baseline:greedy", 18.0))
    store.put(_record("baseline:greedy", 5.0, protocol="snake.score.v0"))
    app.dependency_overrides[_evaluation_store] = lambda: store
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_leaderboard_is_sorted_by_mean_score(client):
    response = client.get("/games/snake/leaderboard", params={"protocol": "snake.score.v1"})

    assert response.status_code == 200
    assert [r["entrant_id"] for r in response.json()] == ["baseline:greedy", "baseline:random"]


def test_leaderboard_without_protocol_returns_every_protocol(client):
    protocols = {r["protocol"] for r in client.get("/games/snake/leaderboard").json()}

    assert protocols == {"snake.score.v1", "snake.score.v0"}


def test_interfaces_describe_every_snake_representation(client):
    response = client.get("/games/snake/interfaces")

    assert response.status_code == 200
    by_id = {i["id"]: i for i in response.json()}
    assert by_id["snake/features.v1+relative3.v1"]["observer"]["size"] == 11
    assert by_id["snake/grid-flat.v1+relative3.v1"]["observer"]["level"] == 1


def test_interfaces_for_unknown_game_is_404(client):
    assert client.get("/games/nope/interfaces").status_code == 404
