from pathlib import Path
from unittest.mock import MagicMock, call

import numpy as np
import pytest

from pinn_project.applications.burgers import plotting


def _prediction_artifact(path: Path, *, target_width: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Write one compact time-major Burgers prediction artifact.

    Args:
        path: Destination `.npz` path.
        target_width: Final target-field width, with zero representing an analytic-only prediction pool.

    Returns:
        Dense scalar reference and prediction fields.
    """
    times = np.asarray([0.0, 1.0], dtype=np.float32)
    positions = np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)
    inputs = np.stack(np.meshgrid(times, positions, indexing="ij"), axis=-1).reshape(-1, 2)
    reference = np.arange(6, dtype=np.float32).reshape(2, 3)
    prediction = reference + 0.5
    targets = reference[..., None] if target_width == 1 else np.empty((2, 3, 0), dtype=np.float32)
    np.savez_compressed(
        path,
        prediction=prediction[..., None],
        target=targets,
        inputs=inputs,
        flat_index=np.arange(6, dtype=np.int64),
        reference_shape=np.asarray([2, 3], dtype=np.int64),
    )
    return reference, prediction


def test_load_burgers_prediction_fields_reconstructs_coordinates_and_values(tmp_path: Path) -> None:
    """Verify the plotting loader returns scalar fields and ordered Cartesian coordinates.

    Args:
        tmp_path: Temporary artifact directory.
    """
    artifact_path = tmp_path / "predictions.npz"
    expected_reference, expected_prediction = _prediction_artifact(artifact_path)

    reference, prediction, times, positions = plotting.load_burgers_prediction_fields(artifact_path)

    np.testing.assert_array_equal(reference, expected_reference)
    np.testing.assert_array_equal(prediction, expected_prediction)
    np.testing.assert_array_equal(times, [0.0, 1.0])
    np.testing.assert_array_equal(positions, [-1.0, 0.0, 1.0])
    assert plotting.relative_l2_error(reference, prediction) == pytest.approx(
        np.linalg.norm(expected_prediction - expected_reference) / np.linalg.norm(expected_reference)
    )


def test_plot_burgers_predictions_uses_three_panel_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify plotting follows the reference, prediction, and absolute-error notebook layout.

    Args:
        monkeypatch: Pytest attribute patch helper used to provide a dependency-free Pyplot double.
        tmp_path: Temporary artifact and figure directory.
    """
    artifact_path = tmp_path / "predictions.npz"
    _prediction_artifact(artifact_path)
    pyplot = MagicMock()
    figure = object()
    pyplot.figure.return_value = figure
    pyplot.pcolor.side_effect = [object(), object(), object()]
    monkeypatch.setattr(plotting, "_import_pyplot", lambda: pyplot)
    output_path = tmp_path / "figures" / "burgers.png"

    result = plotting.plot_burgers_predictions(artifact_path, output_path, show=True)

    assert result == output_path.resolve()
    pyplot.figure.assert_called_once_with(figsize=(18, 5))
    assert pyplot.subplot.call_args_list == [call(1, 3, 1), call(1, 3, 2), call(1, 3, 3)]
    assert pyplot.title.call_args_list == [
        call("Reference solution"),
        call("Predicted solution"),
        call("Absolute error"),
    ]
    assert pyplot.pcolor.call_count == 3
    pyplot.savefig.assert_called_once_with(output_path.resolve(), dpi=200, bbox_inches="tight")
    pyplot.show.assert_called_once_with()
    pyplot.close.assert_called_once_with(figure)


def test_burgers_plotting_rejects_artifacts_without_reference_targets(tmp_path: Path) -> None:
    """Verify analytic-only prediction artifacts report that no numerical reference is available.

    Args:
        tmp_path: Temporary artifact directory.
    """
    artifact_path = tmp_path / "predictions.npz"
    _prediction_artifact(artifact_path, target_width=0)

    with pytest.raises(ValueError, match="target"):
        plotting.load_burgers_prediction_fields(artifact_path)
