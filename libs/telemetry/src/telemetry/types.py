from typing import TypedDict


class GenerationStats(TypedDict):
    run_id: str
    island_id: str | None  # None if there's a single population (no islands)
    generation: int
    timestamp: float
    best_fitness: float
    mean_fitness: float
    worst_fitness: float
    diversity: float
    champion_ref: str  # pointer into ArtifactStore, not the program itself
