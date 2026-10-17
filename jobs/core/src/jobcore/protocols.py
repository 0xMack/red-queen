"""Every built-in evaluation protocol and publish target (docs/design/0018, stage 4): what an EvalSpec or PublishSpec
may name, and the params each lets vary. A leaderboard protocol's definition (seeds, caps, scoring) is fixed and
versioned in `libs/arena`; nothing here can change what its score means."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from jobcore.specs import Protocol, PublishTarget, register_protocol, register_publish_target


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoParams(_Params):
    """A leaderboard protocol: every entrant, scored exactly as the protocol says -- nothing to set."""


class SprtParams(_Params):
    a: str = Field(..., description="A fixed strategy (`material-6`) or a run id prefix, optionally `@depth`.")
    b: str
    elo0: float = Field(0.0, description="H0: A is no stronger than this.")
    elo1: float = Field(50.0, description="H1: A is at least this much stronger.")
    alpha: float = Field(0.05, gt=0, lt=1)
    beta: float = Field(0.05, gt=0, lt=1)
    out: str | None = Field(None, description="Also write the result as JSON under <data dir>/experiments/.")


class StaticRoundRobinParams(_Params):
    games: PositiveInt = Field(200, description="Games per pairing, seats alternating.")


class ExperimentReportParams(_Params):
    experiment: str = Field(..., min_length=1, description="A stored experiment (POST /experiments).")


class CheckersFieldParams(ExperimentReportParams):
    games: PositiveInt = Field(20, description="Games per field opponent (game pairs over ballot openings).")
    depth: PositiveInt = Field(3, description="Plies each final network searches.")


class CheckersH2hParams(ExperimentReportParams):
    arms: list[str] | None = Field(None, description="Only these arms (default: every arm with a baseline).")
    elo1: float = Field(50.0, description="The SPRT's H1: the arm is at least this much stronger.")
    full: bool = Field(False, description="Play the whole ballot instead of stopping when the SPRT decides.")
    depth: PositiveInt = 3


class PublishParams(_Params):
    collect_garbage: bool = Field(True, description="Then delete store files no catalog entry references any more.")


class ScaleTestPublishParams(PublishParams):
    d_model: PositiveInt = 768
    n_layers: PositiveInt = 12


for name, game, summary in (
    ("snake.score.v2", "snake", "game score on 200 held-out games, every finished champion plus the baselines"),
    ("checkers.versus.v2", "checkers", "a Bradley-Terry Elo from game pairs over the ballot, a round robin"),
    ("bandit.skill.v1", "bandit", "skill on 500 held-out games of every scenario, ranked on classic"),
):
    register_protocol(Protocol(name, game, NoParams, "leaderboard", summary))
register_protocol(
    Protocol(
        "checkers.sprt", "checkers", SprtParams, "report", "is A stronger than B: game pairs until the SPRT decides"
    )
)
register_protocol(
    Protocol(
        "checkers.static-round-robin",
        "checkers",
        StaticRoundRobinParams,
        "report",
        "the fixed strategies against each other (the numbers the Multi-Agent Games chapter cites)",
    )
)

for name, game, params, summary in (
    ("experiment.snake-report", "snake", ExperimentReportParams, "each run's final champion on the held-out games"),
    ("experiment.rl-report", "snake", ExperimentReportParams, "held-out scores, each arm against its baseline"),
    ("experiment.checkers-field", "checkers", CheckersFieldParams, "each run's network against a fixed field"),
    ("experiment.checkers-h2h", "checkers", CheckersH2hParams, "each arm against its baseline, head to head (SPRT)"),
):
    register_protocol(Protocol(name, game, params, "report", summary))


class FrontendParams(_Params):
    out: str | None = Field(None, description="Write here instead of under apps/frontend/app/data/.")


class PbtHistoryParams(FrontendParams):
    experiment: str = "selfplay-v4"
    arm: str = "g-pbt"
    seed: int = 0


class RlCurvesParams(FrontendParams):
    names: list[str] | None = Field(None, description="Recordings to write (default: all); `out` is then a directory.")


class RlRecordingParams(FrontendParams):
    run: str = Field(..., min_length=1, description="A tabular RL run (an id or a unique prefix).")
    label: str | None = Field(None, description="How the chapter describes it (default: built from its config).")


class SiteParams(_Params):
    out: str | None = Field(None, description="Where to write it (default: <data dir>/site).")
    models_base_url: str = Field(
        "https://redqueen-data.mackinations.ca/models", description="Where the uploaded model files are served from."
    )
    include_scale_test: bool = Field(False, description="Also the ~340 MB scale-test model (/dev/inference).")


register_publish_target(PublishTarget("snake", PublishParams, "the Snake leaderboard's champions, fp64 + fp32"))
register_publish_target(PublishTarget("tinylm", PublishParams, "every TinyLM checkpoint, fp32 + int8"))
register_publish_target(
    PublishTarget("scale-test", ScaleTestPublishParams, "a large random-weight model for the big-model path")
)
register_publish_target(
    PublishTarget("site", SiteParams, "the API's public GET routes as static files, for hosting with no server")
)
# Frontend data files the Learn chapters draw (publisher.frontend): written into the repo, then committed.
register_publish_target(PublishTarget("frontend.ballot", FrontendParams, "the Checkers ballot, checkersBallot.ts"))
register_publish_target(
    PublishTarget("frontend.pbt-history", PbtHistoryParams, "a PBT run's population history, pbtHistory.ts")
)
register_publish_target(
    PublishTarget("frontend.rl-curves", RlCurvesParams, "chosen RL experiment runs' curves, recordings/*.json")
)
register_publish_target(
    PublishTarget("frontend.rl-recording", RlRecordingParams, "a tabular run's curve and Q-table, the no-WASM demo")
)
