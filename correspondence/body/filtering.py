"""Neighborhood ring schedule for body-correspondence filtering."""

FILTER_RING_SCHEDULE = (1, 4, 9, 16)


def ring_for_iteration(iteration: int) -> int:
    if iteration < 0 or iteration >= len(FILTER_RING_SCHEDULE):
        raise ValueError(f"iteration must be in [0, {len(FILTER_RING_SCHEDULE) - 1}]")
    return FILTER_RING_SCHEDULE[iteration]
