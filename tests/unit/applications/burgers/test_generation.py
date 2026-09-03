from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.io import loadmat

from phijax_hydra_template.applications.burgers import generation


def test_generate_burgers_solution_preserves_initial_and_periodic_conditions() -> None:
    """Verify the Python spectral solver produces finite fields with the benchmark constraints."""
    solution, times, positions = generation.generate_burgers_solution(
        unique_space_points=31,
        time_steps=4,
        max_step=5.0e-3,
    )

    assert solution.shape == (5, 32)
    np.testing.assert_array_equal(times, np.linspace(0.0, 1.0, 5))
    np.testing.assert_array_equal(positions, np.linspace(-1.0, 1.0, 32))
    np.testing.assert_allclose(solution[0], -np.sin(np.pi * positions), atol=1.0e-14)
    np.testing.assert_allclose(solution[:, 0], solution[:, -1], atol=0.0)
    assert np.isfinite(solution).all()


def test_save_burgers_dataset_writes_expected_matlab_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify atomic dataset saving retains the expected variables and array layouts.

    Args:
        monkeypatch: Pytest attribute patch helper used to provide a compact generated solution.
        tmp_path: Temporary dataset output directory.
    """
    solution = np.arange(12, dtype=np.float64).reshape(3, 4)
    times = np.linspace(0.0, 1.0, 3)
    positions = np.linspace(-1.0, 1.0, 4)
    monkeypatch.setattr(generation, "generate_burgers_solution", lambda: (solution, times, positions))
    destination = tmp_path / "burgers.mat"

    result = generation.save_burgers_dataset(destination)
    artifact = loadmat(result)

    assert result == destination.resolve()
    np.testing.assert_array_equal(artifact["usol"], solution)
    np.testing.assert_array_equal(artifact["t"], times[None, :])
    np.testing.assert_array_equal(artifact["x"], positions[None, :])
    assert float(artifact["nu"].item()) == pytest.approx(0.01 / np.pi)


def test_ensure_burgers_dataset_generates_only_when_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify automatic setup reuses existing data and generates only an absent artifact.

    Args:
        monkeypatch: Pytest attribute patch helper used to observe generation calls.
        tmp_path: Temporary dataset path root.
    """
    destination = tmp_path / "burgers.mat"
    generated: list[Path] = []

    def save(path: str | Path) -> Path:
        """Record and create one simulated generated artifact.

        Args:
            path: Requested dataset destination.

        Returns:
            Resolved generated path.
        """
        resolved = Path(path).resolve()
        resolved.write_bytes(b"generated")
        generated.append(resolved)
        return resolved

    monkeypatch.setattr(generation, "save_burgers_dataset", save)

    first = generation.ensure_burgers_dataset(destination)
    repeated = generation.ensure_burgers_dataset(destination)

    assert first == destination.resolve()
    assert repeated == first
    assert generated == [destination.resolve()]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"unique_space_points": 2}, "at least three"),
        ({"time_steps": 0}, "one temporal"),
        ({"viscosity": 0.0}, "finite and positive"),
        ({"relative_tolerance": -1.0}, "finite and positive"),
        ({"absolute_tolerance": np.inf}, "finite and positive"),
        ({"max_step": 0.0}, "finite and positive"),
    ],
)
def test_generate_burgers_solution_rejects_invalid_configuration(
    kwargs: dict[str, Any],
    match: str,
) -> None:
    """Verify invalid numerical settings fail before integration.

    Args:
        kwargs: Invalid generator keyword argument.
        match: Expected validation-message fragment.
    """
    with pytest.raises(ValueError, match=match):
        generation.generate_burgers_solution(**kwargs)
