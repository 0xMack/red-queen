"""Game session endpoints -- doc 0005's "routers/games.py" group.

Server-side simulation for now (a session's Environment instance lives in GameSessionStore,
in-memory, on the backend process) -- doc 0005 step 4, proving this data contract before doc 0005
step 5 moves the actual simulation client-side into Pyodide.
"""

from fastapi import APIRouter, HTTPException

from backend.dependencies import GameSessionStoreDep
from backend.game_sessions import known_games
from backend.schemas import (
    ActionRequest,
    GameSessionCreate,
    GameSessionState,
    TrajectoryArtifact,
)

router = APIRouter(prefix="/games", tags=["games"])


@router.post("/{game}/sessions")
def create_session(
    game: str, body: GameSessionCreate, store: GameSessionStoreDep
) -> GameSessionState:
    if game != body.game:
        raise HTTPException(
            status_code=400, detail="path game and body game must match"
        )
    try:
        session_id, session = store.create(game, body.seed)
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"unknown game: {game} (known: {known_games()})"
        ) from None
    return GameSessionState(
        session_id=session_id,
        game=game,
        render_state=session.states[-1],
        reward=0.0,
        done=False,
        step=0,
    )


@router.post("/{game}/sessions/{session_id}/actions")
def apply_action(
    game: str, session_id: str, body: ActionRequest, store: GameSessionStoreDep
) -> GameSessionState:
    try:
        session, reward = store.step(game, session_id, body.action)
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"no such session: {session_id}"
        ) from None
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from None
    return GameSessionState(
        session_id=session_id,
        game=game,
        render_state=session.states[-1],
        reward=reward,
        done=session.done,
        step=session.step_count,
    )


@router.get("/{game}/sessions/{session_id}/trajectory")
def get_trajectory(
    game: str, session_id: str, store: GameSessionStoreDep
) -> TrajectoryArtifact:
    try:
        session = store.get(game, session_id)
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"no such session: {session_id}"
        ) from None
    return TrajectoryArtifact(
        run_id=None,
        game=game,
        seed=session.seed,
        states=session.states,
        actions=session.actions,
        rewards=session.rewards,
    )
