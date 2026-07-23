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

HIGHER_BETTER = {"accuracy"}              # bigger is better -> normalise directly
LOWER_BETTER = {"latency", "size"}       # smaller is better -> invert during normalisation
MANUAL = {"robustness", "stability"}     # already in [0,1] (human-assigned), used as-is


def _minmax(values: List[float], higher_better: bool) -> List[float]:
    lo, hi = min(values), max(values)    # the range across candidates
    if hi == lo:                         # all equal -> everyone scores 1 (avoid divide-by-zero)
        return [1.0 for _ in values]
    if higher_better:
        return [(v - lo) / (hi - lo) for v in values]   # 0 for worst, 1 for best
    return [(hi - v) / (hi - lo) for v in values]  # invert: smaller -> closer to 1


def score(rows: List[Dict], weights: Dict[str, float] = None) -> List[Dict]:
    """rows: list of dicts with the criteria named in `weights` (+ 'model').

    Returns the rows augmented with per-criterion normalised scores and a 'total'. `weights`
    defaults to the CNN SCORECARD_WEIGHTS; pass a different mapping (e.g. the word bake-off's)
    to reuse this for any candidate set. Higher-is-better/lower-is-better criteria are min-max
    normalised across candidates; any other criterion is treated as an already-[0,1] manual
    score.
    """
    weights = weights or SCORECARD_WEIGHTS   # default to the CNN weights; word/sentiment pass their own
    out = [dict(r) for r in rows]            # copy so we don't mutate the caller's rows

    for crit in weights:                     # normalise each weighted criterion
        if crit in HIGHER_BETTER or crit in LOWER_BETTER:
            vals = [r[crit] for r in rows]                       # this criterion's values across candidates
            norm = _minmax(vals, higher_better=(crit in HIGHER_BETTER))   # -> [0,1]
            for r, n in zip(out, norm):
                r[f"norm_{crit}"] = round(n, 4)                  # store the normalised value
        else:  # manual criterion, already in [0,1]
            for r in out:
                r[f"norm_{crit}"] = float(r.get(crit, 0.0))      # use the manual score as-is

    for r in out:
        r["total"] = round(sum(weights[c] * r[f"norm_{c}"] for c in weights), 4)   # weighted sum = final score

    out.sort(key=lambda r: r["total"], reverse=True)   # rank best-first
    return out


def winner(rows: List[Dict], weights: Dict[str, float] = None) -> Dict:
    return score(rows, weights)[0]           # the top-ranked candidate
