"""m5.backend - the seam to a model. See `m5.backend.base` for the protocol and its contracts.

`hf` imports torch and transformers lazily, so every other part of M5 (and its tests) works on a
machine without them.
"""

from m5.backend.base import (  # noqa: F401
    Backend,
    GenRequest,
    GenResult,
    KLResult,
    ModelInfo,
    ResidualStats,
    ScoreRequest,
    ScoreResult,
    derive_seed,
    locate,
    token_index_at,
)
