"""A small transformer language model trained by gradient descent (docs/design/0004), recorded as an ordinary run
(docs/design/0018 stage 3c answered 0004's deferred "whether it connects": it does, as a run)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt

from jobcore.specs import Algorithm, register_algorithm


class TinyLMParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint: str = Field("alice-v1", min_length=1, description="Name under <data dir>/tinylm/ (.npz + .json).")
    d_model: PositiveInt = 64
    n_heads: PositiveInt = 4
    n_layers: PositiveInt = 2
    d_hidden: PositiveInt = 128
    seq_len: PositiveInt = 32
    batch_size: PositiveInt = 32
    lr: PositiveFloat = 3e-3
    log_every: PositiveInt = Field(100, description="Training steps per recorded generation.")


register_algorithm(
    Algorithm("tinylm", TinyLMParams, frozenset({"steps"}), frozenset({"text"}), "a character-level transformer LM")
)
