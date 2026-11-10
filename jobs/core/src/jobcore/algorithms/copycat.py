"""Copy Cat, learned (docs/design/0024): a small encoder-decoder transformer that reads an image and writes the SVG behind
it as tokens, trained by supervised learning on generated images (`copycat.svgformer`)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveFloat, PositiveInt

from jobcore.specs import Algorithm, register_algorithm


class SvgFormerParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Literal["XS", "S", "M"] = Field(
        "S", description="svgformer.model.SIZES: XS for tests, S ~5M, M ~12M params."
    )
    batch: PositiveInt = 64
    lr: PositiveFloat = 1e-3
    warmup: NonNegativeInt = Field(1000, description="Steps of linear warmup before the cosine decay.")
    weight_decay: float = Field(0.01, ge=0)
    clip: PositiveFloat = 1.0
    precision: Literal["bf16", "fp32"] = Field(
        "bf16", description="bf16 autocast on the GPU; the CPU always runs fp32."
    )
    severities: list[Literal["clean", "light", "medium", "heavy"]] = Field(
        default_factory=lambda: ["clean", "light", "medium", "heavy"], description="corrupt.v1 severities to train on."
    )
    workers: NonNegativeInt = Field(24, description="Data workers generating and corrupting images (0: in process).")
    log_every: PositiveInt = Field(500, description="Optimizer steps per recorded generation.")
    validate_images: PositiveInt = Field(400, le=400, description="synthetic/validate images in the held-out curve.")
    device: Literal["auto", "cuda", "cpu"] = "auto"
    number_sigma: float = Field(0.0, ge=0, description="Soft coordinate labels, grid steps wide (0: exact bins).")
    number_features: bool = Field(True, description="Fourier features of a number token's value in and out.")
    patch_features: bool = Field(True, description="Fourier features of each patch's place in the token grid's frame.")
    fp32_head: bool = Field(False, description="The output logits in fp32 under bf16 autocast.")


register_algorithm(
    Algorithm(
        "copycat.svgformer",
        SvgFormerParams,
        frozenset({"steps"}),
        frozenset({"copycat"}),
        "a small transformer that reads an image and writes its SVG, trained on generated images",
    )
)
