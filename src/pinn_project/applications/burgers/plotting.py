from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _scalar_field(values: Any, reference_shape: tuple[int, int], *, name: str) -> NDArray[np.float64]:
    """Validate one dense scalar field from a Burgers prediction artifact.

    Args:
        values: Candidate dense scalar values.
        reference_shape: Expected `(time, space)` grid shape.
        name: Artifact field name used in validation errors.

    Returns:
        Finite `float64` field with shape `(time, space)`.

    Raises:
        ValueError: If values are not a finite scalar field on `reference_shape`.
    """
    field = np.asarray(values, dtype=np.float64)
    if field.shape == (*reference_shape, 1):
        field = field[..., 0]
    if field.shape != reference_shape:
        raise ValueError(
            f"Burgers artifact `{name}` must have shape {reference_shape} or {(*reference_shape, 1)}, "
            f"received {field.shape}."
        )
    if not np.isfinite(field).all():
        raise ValueError(f"Burgers artifact `{name}` must contain only finite values.")
    return field


def load_burgers_prediction_fields(
    artifact_path: str | Path,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Load reference, prediction, time, and space arrays from a PhiJAX artifact.

    Args:
        artifact_path: Prediction `.npz` produced by `phijax-train predict=true` or `phijax-predict`.

    Returns:
        Reference field, predicted field, time coordinates, and spatial coordinates.

    Raises:
        FileNotFoundError: If `artifact_path` does not exist.
        KeyError: If a required prediction or reconstruction field is absent.
        ValueError: If the artifact is not a finite two-dimensional scalar Burgers grid with reference targets.
    """
    path = Path(artifact_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Burgers prediction artifact does not exist: `{path}`.")
    with np.load(path) as artifact:
        required = ("prediction", "target", "inputs", "flat_index", "reference_shape")
        missing = tuple(name for name in required if name not in artifact)
        if missing:
            raise KeyError(f"Burgers prediction artifact is missing required arrays: {missing}.")
        shape_values = np.asarray(artifact["reference_shape"], dtype=np.int64).reshape(-1)
        if shape_values.size != 2 or np.any(shape_values < 1):
            raise ValueError(
                "Burgers `reference_shape` must contain two positive dimensions ordered as `[time, space]`."
            )
        reference_shape = (int(shape_values[0]), int(shape_values[1]))
        prediction = _scalar_field(artifact["prediction"], reference_shape, name="prediction")
        reference = _scalar_field(artifact["target"], reference_shape, name="target")
        inputs = np.asarray(artifact["inputs"], dtype=np.float64)
        flat_index = np.asarray(artifact["flat_index"], dtype=np.int64).reshape(-1)

    row_count = int(np.prod(reference_shape))
    if inputs.ndim != 2 or inputs.shape[0] != row_count or inputs.shape[1] < 2:
        raise ValueError(f"Burgers artifact `inputs` must have shape `[{row_count}, >=2]` ordered as `[t, x]`.")
    if flat_index.shape != (row_count,) or not np.array_equal(np.sort(flat_index), np.arange(row_count)):
        raise ValueError("Burgers artifact `flat_index` must be a permutation of the dense grid indices.")
    dense_inputs = np.empty((row_count, inputs.shape[1]), dtype=np.float64)
    dense_inputs[flat_index] = inputs
    dense_inputs = dense_inputs.reshape(*reference_shape, inputs.shape[1])
    times = dense_inputs[:, 0, 0]
    positions = dense_inputs[0, :, 1]
    if not np.isfinite(dense_inputs).all():
        raise ValueError("Burgers artifact `inputs` must contain only finite coordinates.")
    if not np.allclose(dense_inputs[..., 0], times[:, None]) or not np.allclose(
        dense_inputs[..., 1], positions[None, :]
    ):
        raise ValueError("Burgers artifact coordinates must form a time-major Cartesian grid ordered as `[t, x]`.")
    return reference, prediction, times, positions


def relative_l2_error(reference: Any, prediction: Any) -> float:
    """Compute the relative Euclidean error of a predicted Burgers field.

    Args:
        reference: Finite nonzero reference field.
        prediction: Finite predicted field with the same shape as `reference`.

    Returns:
        Relative L2 error `||prediction - reference||_2 / ||reference||_2`.

    Raises:
        ValueError: If fields differ in shape, contain non-finite values, are empty, or the reference norm is zero.
    """
    reference_array = np.asarray(reference, dtype=np.float64)
    prediction_array = np.asarray(prediction, dtype=np.float64)
    if reference_array.shape != prediction_array.shape or reference_array.size == 0:
        raise ValueError("Reference and prediction must have the same non-empty shape.")
    if not np.isfinite(reference_array).all() or not np.isfinite(prediction_array).all():
        raise ValueError("Reference and prediction must contain only finite values.")
    reference_norm = np.linalg.norm(reference_array)
    if reference_norm == 0.0:
        raise ValueError("Relative L2 error requires a reference field with nonzero norm.")
    return float(np.linalg.norm(prediction_array - reference_array) / reference_norm)


def _import_pyplot() -> Any:
    """Import Matplotlib lazily to defer plotting-backend initialization until needed.

    Returns:
        Imported :mod:`matplotlib.pyplot` module.

    Raises:
        ModuleNotFoundError: If the default project dependencies have not been installed completely.
    """
    try:
        import matplotlib.pyplot as pyplot
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Burgers plotting requires the default Matplotlib dependency; run `uv sync`."
        ) from error
    return pyplot


def plot_burgers_predictions(
    artifact_path: str | Path,
    output_path: str | Path | None = None,
    *,
    show: bool = False,
) -> Path:
    """Plot the Burgers reference, prediction, and pointwise absolute error.

    Args:
        artifact_path: PhiJAX prediction `.npz` containing dense numerical reference targets.
        output_path: Destination figure path, or `None` to replace the artifact suffix with `.png`.
        show: Whether to display the completed figure interactively after saving it.

    Returns:
        Absolute path to the saved figure.
    """
    reference, prediction, times, positions = load_burgers_prediction_fields(artifact_path)
    destination = (
        Path(artifact_path).expanduser().resolve().with_suffix(".png")
        if output_path is None
        else Path(output_path).expanduser().resolve()
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    pyplot = _import_pyplot()
    figure = pyplot.figure(figsize=(18, 5))
    fields = (
        ("Reference solution", reference),
        ("Predicted solution", prediction),
        ("Absolute error", np.abs(prediction - reference)),
    )
    for index, (title, field) in enumerate(fields, start=1):
        pyplot.subplot(1, 3, index)
        pyplot.title(title)
        colors = pyplot.pcolor(times, positions, field.T, shading="auto", cmap="jet")
        pyplot.xlabel("t")
        pyplot.ylabel("x")
        pyplot.colorbar(colors)
    pyplot.tight_layout()
    pyplot.savefig(destination, dpi=200, bbox_inches="tight")
    if show:
        pyplot.show()
    pyplot.close(figure)
    return destination


__all__ = ["load_burgers_prediction_fields", "plot_burgers_predictions", "relative_l2_error"]
