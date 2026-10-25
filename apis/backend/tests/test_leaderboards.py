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


def test_records_are_put_and_pruned_through_the_api(client):
    response = client.put("/games/snake/evaluations", json=_record("run:abc", 9.0).model_dump(mode="json"))
    assert response.status_code == 204
    entrants = [
        r["entrant_id"] for r in client.get("/games/snake/leaderboard", params={"protocol": "snake.score.v1"}).json()
    ]
    assert entrants == ["baseline:greedy", "run:abc", "baseline:random"]

    wrong_game = client.put("/games/checkers/evaluations", json=_record("run:abc", 9.0).model_dump(mode="json"))
    assert wrong_game.status_code == 422

    pruned = client.post("/games/snake/evaluations/prune", json={"protocol": "snake.score.v1", "keep": ["run:abc"]})
    assert sorted(pruned.json()["removed"]) == ["baseline:greedy", "baseline:random"]
    assert len(client.get("/games/snake/leaderboard", params={"protocol": "snake.score.v0"}).json()) == 1  # untouched


def test_copycat_outputs_are_served_per_image_not_on_the_leaderboard(tmp_path):
    from arena import copycat

    target = copycat.load_set()[0]
    store = SqliteEvaluationStore(tmp_path / "evaluations.db")
    image = {"id": target.id, "severity": target.severity, "fidelity": 60.0, "svg": target.clean}
    store.put(
        EvaluationRecord(
            game="copycat",
            protocol=copycat.PROTOCOL,
            entrant_id="baseline:structured",
            entrant_kind="baseline",
            label="Structured pipeline",
            interface=copycat.INTERFACE,
            created_at=1.0,
            metrics={"quality": {"mean": 60.0}, "copycat": {"images": [image]}},
        )
    )
    app.dependency_overrides[_evaluation_store] = lambda: store
    try:
        client = TestClient(app)
        board = client.get("/games/copycat/leaderboard").json()
        assert board[0]["metrics"]["copycat"]["images"] == [{k: v for k, v in image.items() if k != "svg"}]
        targets = client.get("/games/copycat/targets").json()
        assert [t["seed"] for t in targets] == list(copycat.HELD_OUT_SEEDS)
        assert targets[0]["png"].startswith("data:image/png;base64,")
        one = client.get(f"/games/copycat/targets/{target.seed}").json()
        assert one["clean"] == target.clean
        assert one["outputs"]["baseline:structured"] == {
            "svg": target.clean,
            "scores": board[0]["metrics"]["copycat"]["images"][0],
        }
        assert client.get("/games/copycat/targets/1").status_code == 404
    finally:
        app.dependency_overrides.clear()
