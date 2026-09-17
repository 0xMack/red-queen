from pydantic import BaseModel, Field


class GenerationStats(BaseModel):
    """Per-generation summary stats for one run (and, if using islands, one island)."""

    run_id: str = Field(..., min_length=1, description="Registry-assigned id of the run this generation belongs to.")
    island_id: str | None = Field(
        default=None,
        description="Island id, if the run uses an island model; None for a single population.",
    )
    generation: int = Field(..., ge=0, description="0-indexed generation number within the run.")
    timestamp: float = Field(..., gt=0, description="Unix timestamp when this generation's stats were recorded.")
    best_fitness: float = Field(..., description="Fitness of the best individual in this generation.")
    mean_fitness: float = Field(..., description="Mean fitness across the generation's population.")
    worst_fitness: float = Field(..., description="Fitness of the worst individual in this generation.")
    diversity: float = Field(..., ge=0, description="Population diversity metric for this generation (higher = more diverse).")
    champion_ref: str = Field(
        ..., min_length=1, description="ArtifactStore key for this generation's best individual's stored program."
    )
