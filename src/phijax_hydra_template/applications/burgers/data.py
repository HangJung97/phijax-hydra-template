from collections.abc import Sequence

import numpy as np
from phijax.data import HostPool


def _validate_interval(bounds: Sequence[float], *, name: str) -> tuple[float, float]:
    """Validate one finite increasing coordinate interval.

    Args:
        bounds: Candidate lower and upper coordinate bounds.
        name: Configuration field name used in validation errors.

    Returns:
        Finite increasing `(lower, upper)` bounds.

    Raises:
        ValueError: If bounds do not contain two finite increasing values.
    """
    values = tuple(float(value) for value in bounds)
    if len(values) != 2 or not np.isfinite(values).all() or values[0] >= values[1]:
        raise ValueError(f"`{name}` must contain two finite increasing values.")
    return values


def _pool(
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    reference_shape: tuple[int, ...] | None = None,
) -> HostPool:
    """Construct one immutable Burgers host pool.

    Args:
        inputs: Rank-two coordinate rows ordered as `[t, x]`.
        targets: Rank-two target rows.
        reference_shape: Optional dense grid shape for prediction reconstruction.

    Returns:
        Indexed immutable host pool.
    """
    row_count = inputs.shape[0]
    return HostPool(
        inputs=inputs,
        targets=targets,
        aux={},
        metadata={"coordinate_names": ("t", "x"), "output_names": ("u",)},
        reference_shape=reference_shape or (row_count,),
        flat_index=np.arange(row_count, dtype=np.int64),
    )


def build_burgers_pools(
    *,
    seed: int = 42,
    initial_size: int = 256,
    pde_size: int | None = 16384,
    predict_shape: tuple[int, int] | list[int] = (100, 256),
    time_bounds: tuple[float, float] | list[float] = (0.0, 1.0),
    space_bounds: tuple[float, float] | list[float] = (-1.0, 1.0),
) -> dict[str, HostPool]:
    """Build deterministic pools for the one-dimensional Burgers benchmark.

    The initial condition is `u(t_min, x) = -sin(pi * x)`. Interior residual points are sampled uniformly over the
    configured space-time rectangle, while prediction points form a dense time-major grid.

    Args:
        seed: NumPy generator seed used for interior collocation points.
        initial_size: Number of uniformly spaced initial-condition points.
        pde_size: Number of uniformly sampled interior residual points, or `None` to omit the finite PDE pool when
            batches are generated continuously by the data module.
        predict_shape: Dense `(time, space)` prediction-grid shape.
        time_bounds: Inclusive lower and upper temporal bounds.
        space_bounds: Inclusive lower and upper spatial bounds.

    Returns:
        Named `initial` and `predict` host pools, plus `pde` when `pde_size` is defined.

    Raises:
        ValueError: If sizes, prediction shape, or coordinate bounds are invalid.
    """
    if initial_size < 1 or (pde_size is not None and pde_size < 1):
        raise ValueError("Burgers training pool sizes must be positive when defined.")
    resolved_shape = tuple(int(value) for value in predict_shape)
    if len(resolved_shape) != 2 or min(resolved_shape) < 1:
        raise ValueError("`predict_shape` must contain two positive dimensions.")
    time_min, time_max = _validate_interval(time_bounds, name="time_bounds")
    space_min, space_max = _validate_interval(space_bounds, name="space_bounds")

    initial_space = np.linspace(space_min, space_max, initial_size, dtype=np.float32)
    initial_inputs = np.column_stack((np.full(initial_size, time_min, dtype=np.float32), initial_space)).astype(
        np.float32
    )
    initial_targets = (-np.sin(np.pi * initial_space)).astype(np.float32)[:, None]

    times = np.linspace(time_min, time_max, resolved_shape[0], dtype=np.float32)
    positions = np.linspace(space_min, space_max, resolved_shape[1], dtype=np.float32)
    prediction_inputs = np.stack(np.meshgrid(times, positions, indexing="ij"), axis=-1).reshape(-1, 2)

    pools = {
        "initial": _pool(initial_inputs, initial_targets),
        "predict": _pool(
            prediction_inputs,
            np.zeros((prediction_inputs.shape[0], 0), dtype=np.float32),
            reference_shape=resolved_shape,
        ),
    }
    if pde_size is not None:
        generator = np.random.default_rng(seed)
        pde_inputs = np.column_stack(
            (
                generator.uniform(time_min, time_max, pde_size),
                generator.uniform(space_min, space_max, pde_size),
            )
        ).astype(np.float32)
        pools["pde"] = _pool(pde_inputs, np.zeros((pde_size, 0), dtype=np.float32))
    return pools
