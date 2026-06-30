"""Runtime-overhead accounting for selector evaluation."""


def apply_overhead(
    solver_is_solved: int,
    solver_runtime: float,
    overhead: float | None,
    timeout: float,
) -> tuple[int, float]:
    """Add selector overhead and convert over-budget runs to timeouts."""
    effective_runtime = solver_runtime + (overhead or 0.0)
    if effective_runtime > timeout:
        return 0, timeout
    return solver_is_solved, effective_runtime
