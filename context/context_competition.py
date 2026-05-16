from __future__ import annotations

import math


def softmax_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    shift = max(scores)
    exps = [math.exp(s - shift) for s in scores]
    total = sum(exps) or 1.0
    return [e / total for e in exps]
