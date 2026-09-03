import os
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from phijax.data import load_prediction_artifact


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
        ValueError: If the artifact is not a complete finite two-dimensional scalar Burgers grid with reference
            targets.
    """
    artifact = load_prediction_artifact(artifact_path)
    if len(artifact.reference_shape) != 2:
        raise ValueError("Burgers `reference_shape` must contain two dimensions ordered as `[time, space]`.")
    if artifact.coordinate_names != ("t", "x"):
        raise ValueError("Burgers artifact coordinates must be named and ordered as `[t, x]`.")
    if artifact.output_names != ("u",):
        raise ValueError("Burgers artifact outputs must contain the single field `u`.")
    if "u" not in artifact.targets:
        raise ValueError("Burgers plotting requires an artifact containing the reference target field `u`.")
    if not artifact.mask.all():
        raise ValueError("Burgers plotting requires a complete dense prediction grid.")
    prediction = np.asarray(artifact.outputs["u"], dtype=np.float64)
    reference = np.asarray(artifact.targets["u"], dtype=np.float64)
    dense_inputs = np.asarray(artifact.dense_inputs, dtype=np.float64)
    if not np.isfinite(reference).all() or not np.isfinite(prediction).all():
        raise ValueError("Burgers prediction and target fields must contain only finite values.")
    times = dense_inputs[:, 0, 0]
    positions = dense_inputs[0, :, 1]
    if not np.isfinite(dense_inputs).all():
        raise ValueError("Burgers artifact `inputs` must contain only finite coordinates.")
    if not np.allclose(dense_inputs[..., 0], times[:, None]) or not np.allclose(
        dense_inputs[..., 1], positions[None, :]
    ):
        raise ValueError("Burgers artifact coordinates must form a time-major Cartesian grid ordered as `[t, x]`.")
    return reference, prediction, times, positions


def _import_pyplot(*, show: bool) -> Any:
    """Import Matplotlib lazily to defer plotting-backend initialization until needed.

    Args:
        show: Whether the caller needs an interactive plotting backend.

    Returns:
        Imported :mod:`matplotlib.pyplot` module.

    Raises:
        ModuleNotFoundError: If the default project dependencies have not been installed completely.
    """
    previous_backend = os.environ.get("MPLBACKEND")
    if not show:
        # Notebook kernels can export an inline backend that is unavailable in the project's plotting subprocess.
        os.environ["MPLBACKEND"] = "Agg"
    try:
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Burgers plotting requires the default Matplotlib dependency; run `uv sync`."
        ) from error
    finally:
        if not show:
            if previous_backend is None:
                os.environ.pop("MPLBACKEND", None)
            else:
                os.environ["MPLBACKEND"] = previous_backend
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
    pyplot = _import_pyplot(show=show)
    figure = pyplot.figure(figsize=(18, 5))
    solution_limit = float(np.max(np.abs(reference)))
    fields = (
        ("Reference solution", reference, "coolwarm", {"vmin": -solution_limit, "vmax": solution_limit}),
        ("Predicted solution", prediction, "coolwarm", {"vmin": -solution_limit, "vmax": solution_limit}),
        ("Absolute error", np.abs(prediction - reference), "magma", {"vmin": 0.0}),
    )
    for index, (title, field, colormap, color_limits) in enumerate(fields, start=1):
        pyplot.subplot(1, 3, index)
        pyplot.title(title)
        colors = pyplot.pcolor(times, positions, field.T, shading="auto", cmap=colormap, **color_limits)
        pyplot.xlabel("t")
        pyplot.ylabel("x")
        pyplot.colorbar(colors)
    pyplot.tight_layout()
    pyplot.savefig(destination, dpi=200, bbox_inches="tight")
    if show:
        pyplot.show()
    pyplot.close(figure)
    return destination


__all__ = ["load_burgers_prediction_fields", "plot_burgers_predictions"]
