"""Weighted scorecard — turns bake-off metrics into a single winner (CRISP-DM: Evaluation).

Each criterion is normalised to [0,1] across the candidates (min-max), with lower-is-better
metrics inverted, then combined with the weights in config.SCORECARD_WEIGHTS.

Robustness and stability are manual [0,1] scores you fill in after the Grad-CAM/bias review
and the live-webcam test — the bake-off is not purely automatic on purpose: the human check
that "the model looks at the hand and stays stable live" is part of the decision.
"""
from __future__ import annotations

from typing import Dict, List

from .config import SCORECARD_WEIGHTS

HIGHER_BETTER = {"accuracy"}
LOWER_BETTER = {"latency", "size"}
MANUAL = {"robustness", "stability"}  # already in [0,1]


def _minmax(values: List[float], higher_better: bool) -> List[float]:
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0 for _ in values]
    if higher_better:
        return [(v - lo) / (hi - lo) for v in values]
    return [(hi - v) / (hi - lo) for v in values]  # invert: smaller -> closer to 1


def score(rows: List[Dict]) -> List[Dict]:
    """rows: list of dicts with keys accuracy, latency, size, robustness, stability, model.

    Returns the rows augmented with per-criterion normalised scores and a 'total'.
    """
    out = [dict(r) for r in rows]

    for crit in ("accuracy", "latency", "size"):
        vals = [r[crit] for r in rows]
        norm = _minmax(vals, higher_better=(crit in HIGHER_BETTER))
        for r, n in zip(out, norm):
            r[f"norm_{crit}"] = round(n, 4)

    for crit in MANUAL:
        for r in out:
            r[f"norm_{crit}"] = float(r.get(crit, 0.0))

    for r in out:
        r["total"] = round(
            sum(SCORECARD_WEIGHTS[c] * r[f"norm_{c}"] for c in SCORECARD_WEIGHTS), 4
        )

    out.sort(key=lambda r: r["total"], reverse=True)
    return out


def winner(rows: List[Dict]) -> Dict:
    return score(rows)[0]
