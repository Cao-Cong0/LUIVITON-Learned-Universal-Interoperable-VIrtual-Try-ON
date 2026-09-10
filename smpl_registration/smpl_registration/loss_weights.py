"""Loss-weight schedules for body and clothing registration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


LossWeight = Callable[[Any, float], Any]

BODY_STAGE_ONE_COEFFICIENTS = {
    "b2s": 10.0,
    "s2b": 10.0,
    "correspondence": 50.0**2,
    "betas": 5.0**-1,
    "offsets": 150.0**2,
    "lap": 4000.0**2,
    "scale": 1.0,
}

BODY_STAGE_TWO_COEFFICIENTS = {
    "b2s": 25.0**2,
    "s2b": 25.0**2,
    "correspondence": 5.0**2,
    "betas": 5.0**-1,
    "offsets": 70.0**-1,
    "lap": 3000.0**2,
    "scale": 1.0,
}

CLOTHING_COEFFICIENTS = {
    "b2c": 35.0**2,
    "penetration": 24.0**2,
    "betas": 3.0**-1,
    "lap": 1.0,
    "pose": 7.0**2,
}


def body_stage_one_weights() -> dict[str, LossWeight]:
    weights = _decaying_weights(BODY_STAGE_ONE_COEFFICIENTS)
    weights["b2s"] = _growing(BODY_STAGE_ONE_COEFFICIENTS["b2s"])
    weights["pose_pr"] = _decaying(90.0**-5)
    weights["hand"] = _decaying(10.0**-5)
    return weights


def body_stage_two_weights() -> dict[str, LossWeight]:
    weights = _decaying_weights(BODY_STAGE_TWO_COEFFICIENTS)
    weights["b2s"] = _growing(BODY_STAGE_TWO_COEFFICIENTS["b2s"])
    weights["pose_pr"] = _decaying(10.0**-5)
    weights["hand"] = _decaying(10.0**-5)
    return weights


def clothing_weights() -> dict[str, LossWeight]:
    weights = _decaying_weights(CLOTHING_COEFFICIENTS)
    weights["pose_pr"] = _decaying(10.0**-5)
    weights["hand"] = _decaying(10.0**-5)
    return weights


def _decaying_weights(coefficients: dict[str, float]) -> dict[str, LossWeight]:
    return {name: _decaying(value) for name, value in coefficients.items()}


def _decaying(coefficient: float) -> LossWeight:
    return lambda loss, iteration: coefficient * loss / (1.0 + iteration)


def _growing(coefficient: float) -> LossWeight:
    return lambda loss, iteration: coefficient * loss * (1.0 + iteration)
