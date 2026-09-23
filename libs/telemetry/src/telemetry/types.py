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
    diversity: float = Field(
        ..., ge=0, description="Population diversity metric for this generation (higher = more diverse)."
    )
    champion_ref: str = Field(
        ..., min_length=1, description="ArtifactStore key for this generation's best individual's stored program."
    )
    held_out_score: float | None = Field(
        default=None,
        description=(
            "The champion's mean *game score* on games it never trained on, if the job measured it this "
            "generation (usually every N generations -- None otherwise). Monitoring only, never used for "
            "selection: it's what shows overfitting, i.e. training fitness rising while this falls "
            "(docs/design/0007)."
        ),
    )
    extras: dict[str, float] | None = Field(
        default=None,
        description=(
            "Algorithm-specific numbers for this generation, free-form (e.g. NEAT's species count and its "
            "champion's structure size). None for algorithms with nothing extra to say. Kept generic on "
            "purpose so a new algorithm needs no telemetry schema change."
        ),
    )
