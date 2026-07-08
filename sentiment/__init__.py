"""GROOPY sentiment — shared, partner-owned module.

Computes sentiment on the text flowing through either track and returns a shared
`Sentiment` (label + score). Ships a dependency-free rule-based STUB so the rest of the
system runs today; the partner swaps in a real model behind the same `analyze(text)`
interface.

IMPORTANT: what sentiment DRIVES in the output (a displayed label, signing emphasis/speed,
or avatar expression) is NOT decided yet. This module only produces the value; the wiring
seam lives in synthesis/src/pipeline.apply_sentiment and is currently a no-op.
"""
from .src.analyze import analyze  # noqa: F401
