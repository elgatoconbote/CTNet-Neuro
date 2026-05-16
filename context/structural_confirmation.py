from __future__ import annotations


def is_structurally_confirmed(
    alignment: float,
    confidence: float,
    threshold: float = 0.85,
) -> bool:
    return alignment >= threshold and confidence >= 0.50
