from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from phijax.data import HostPool, get_array

# Burgers model inputs are ordered `[t, x]` and the single model output is `u`.
_COORDINATE_NAMES = ("t", "x")
_OUTPUT_NAMES = ("u",)


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


def _sample_domain(
    seed: int,
    size: int,
    time_bounds: tuple[float, float],
    space_bounds: tuple[float, float],
) -> NDArray[np.float32]:
    """Draw uniformly distributed interior coordinates from an explicitly seeded NumPy generator.

    Args:
        seed: NumPy generator seed.
        size: Number of coordinate rows.
        time_bounds: Lower and upper temporal bounds.
        space_bounds: Lower and upper spatial bounds.

    Returns:
        Coordinates with shape `[size, 2]` ordered as `[t, x]`.
    """
    # Draw every `t` before every `x` so a given seed always yields the same pool.
    generator = np.random.default_rng(seed)
    return np.column_stack(
        (generator.uniform(*time_bounds, size), generator.uniform(*space_bounds, size)),
    ).astype(np.float32)


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
    grid_inputs = np.stack(np.meshgrid(times, positions, indexing="ij"), axis=-1)
    metadata = {"coordinate_names": _COORDINATE_NAMES, "output_names": _OUTPUT_NAMES}

    pools = {
        "initial": HostPool(initial_inputs, initial_targets, metadata=metadata),
        # The analytic grid has no reference solution, so its targets are zero-width.
        "predict": HostPool.from_grid(grid_inputs, metadata=metadata),
    }
    if pde_size is not None:
        pde_inputs = _sample_domain(seed, pde_size, (time_min, time_max), (space_min, space_max))
        pools["pde"] = HostPool(pde_inputs, metadata=metadata)
    return pools


def build_burgers_file_pools(
    arrays: Mapping[str, Any],
    *,
    seed: int = 42,
    pde_size: int | None = 16384,
) -> dict[str, HostPool]:
    """Build Burgers pools from a loaded reference solution on a `[t, x]` grid.

    The `initial` pool holds the first time row of the grid and keeps the full grid as its reconstruction shape. The
    `predict` pool holds the whole grid with the reference solution as targets. Interior residual points are sampled
    uniformly between the first and last grid coordinates.

    Args:
        arrays: Loaded arrays with one-dimensional `t` and `x` coordinates and `usol` with shape `[t, x]`.
        seed: NumPy generator seed used for interior collocation points.
        pde_size: Number of uniformly sampled interior residual points, or `None` to omit the finite PDE pool when
            batches are generated continuously by the data module.

    Returns:
        Named `initial` and `predict` host pools, plus `pde` when `pde_size` is defined.

    Raises:
        ValueError: If a coordinate is not strictly increasing with at least two values, `usol` does not have shape
            `[t, x]`, or `pde_size` is not positive.
    """
    if pde_size is not None and pde_size < 1:
        raise ValueError("Burgers `pde_size` must be positive when defined.")
    # Cast before taking bounds so the sampled domain matches the float32 coordinates the model sees.
    times = np.asarray(get_array(arrays, "t"), dtype=np.float32).squeeze()
    positions = np.asarray(get_array(arrays, "x"), dtype=np.float32).squeeze()
    for name, values in (("t", times), ("x", positions)):
        if values.ndim != 1 or values.size < 2 or np.any(np.diff(values) <= 0.0):
            raise ValueError(f"Burgers coordinate `{name}` must be strictly increasing with at least two values.")
    solution = np.asarray(get_array(arrays, "usol"), dtype=np.float32)
    grid_shape = (times.size, positions.size)
    if solution.shape != grid_shape:
        raise ValueError(f"Burgers `usol` must have shape `[t, x]` = {grid_shape}; received {solution.shape}.")

    time_bounds = (float(times[0]), float(times[-1]))
    space_bounds = (float(positions[0]), float(positions[-1]))
    sampling_bounds = (("t", *time_bounds), ("x", *space_bounds))
    grid_metadata = {
        "coordinate_names": _COORDINATE_NAMES,
        "output_names": _OUTPUT_NAMES,
        "sampling_bounds": sampling_bounds,
    }
    grid_inputs = np.stack(np.meshgrid(times, positions, indexing="ij"), axis=-1)
    initial_mask = np.zeros(solution.shape, dtype=bool)
    initial_mask[0] = True

    pools = {
        "initial": HostPool.from_grid(grid_inputs, solution[..., None], mask=initial_mask, metadata=grid_metadata),
        "predict": HostPool.from_grid(grid_inputs, solution[..., None], metadata=grid_metadata),
    }
    if pde_size is not None:
        pools["pde"] = HostPool(
            _sample_domain(seed, pde_size, time_bounds, space_bounds),
            metadata={"coordinate_names": _COORDINATE_NAMES, "sampling_bounds": sampling_bounds},
        )
    return pools
