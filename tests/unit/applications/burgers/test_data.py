from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest
from phijax.data import (
    ChunkedPredictionSource,
    NamedBatchSource,
    RandomRowSampler,
    UniformDomainSampler,
    reconstruct_predictions,
)

from phijax_hydra_template.applications.burgers import (
    BurgersDataModule,
    build_burgers_file_pools,
    build_burgers_pools,
)


def _reference_arrays() -> dict[str, np.ndarray]:
    """Build a small synthetic Burgers reference solution on a `[t, x]` grid.

    Returns:
        Float64 `t` with shape `[3]`, `x` with shape `[4]`, and `usol` with shape `[3, 4]`, as stored in the file.
    """
    times = np.linspace(0.0, 0.5, 3)
    positions = np.linspace(-1.0, 1.0, 4)
    return {"t": times, "x": positions, "usol": np.add.outer(10.0 * times, positions)}


def test_burgers_pools_match_domain_initial_condition_and_prediction_layout() -> None:
    """Verify deterministic domain sampling and dense time-major prediction indexing."""
    pools = build_burgers_pools(seed=7, initial_size=5, pde_size=6, predict_shape=(3, 4))

    assert set(pools) == {"initial", "pde", "predict"}
    assert pools["initial"].inputs.shape == (5, 2)
    assert pools["pde"].inputs.shape == (6, 2)
    np.testing.assert_allclose(pools["initial"].inputs[:, 0], 0.0)
    np.testing.assert_allclose(
        pools["initial"].targets[:, 0],
        -np.sin(np.pi * pools["initial"].inputs[:, 1]),
        atol=1e-7,
    )
    assert np.all((pools["pde"].inputs[:, 0] >= 0.0) & (pools["pde"].inputs[:, 0] <= 1.0))
    assert np.all((pools["pde"].inputs[:, 1] >= -1.0) & (pools["pde"].inputs[:, 1] <= 1.0))
    assert pools["predict"].metadata["coordinate_names"] == ("t", "x")

    predictions = np.arange(12, dtype=np.float32)[:, None]
    dense = reconstruct_predictions(predictions, pools["predict"])
    assert dense.shape == (3, 4, 1)
    np.testing.assert_array_equal(dense.reshape(12, 1), predictions)


def test_burgers_interior_sampling_is_explicitly_seeded() -> None:
    """Verify identical seeds reproduce collocation points independently of global NumPy state."""
    first = build_burgers_pools(seed=11, initial_size=3, pde_size=5, predict_shape=(2, 2))
    repeated = build_burgers_pools(seed=11, initial_size=3, pde_size=5, predict_shape=(2, 2))
    changed = build_burgers_pools(seed=12, initial_size=3, pde_size=5, predict_shape=(2, 2))
    np.testing.assert_array_equal(first["pde"].inputs, repeated["pde"].inputs)
    assert not np.array_equal(first["pde"].inputs, changed["pde"].inputs)


def test_burgers_pool_builder_can_omit_finite_interior_candidates() -> None:
    """Verify continuous sampling mode can construct only finite observation and prediction pools."""
    pools = build_burgers_pools(initial_size=3, pde_size=None, predict_shape=(2, 2))

    assert set(pools) == {"initial", "predict"}


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"initial_size": 0}, "sizes"),
        ({"pde_size": 0}, "sizes"),
        ({"predict_shape": (2, 0)}, "predict_shape"),
        ({"predict_shape": (2, 3, 4)}, "predict_shape"),
        ({"time_bounds": (1.0, 0.0)}, "time_bounds"),
        ({"space_bounds": (-1.0, -1.0)}, "space_bounds"),
    ],
)
def test_burgers_pools_reject_invalid_geometry(kwargs: dict[str, Any], match: str) -> None:
    """Verify invalid pool sizes, shapes, and intervals fail eagerly.

    Args:
        kwargs: Invalid builder keyword arguments.
        match: Expected validation-message fragment.
    """
    with pytest.raises(ValueError, match=match):
        build_burgers_pools(**kwargs)


def test_burgers_data_module_owns_finite_training_and_prediction_policy() -> None:
    """Verify the application module constructs pools, samples rows, and chunks predictions."""
    data_module = BurgersDataModule(
        {"initial": "all", "pde": 3, "predict": 4},
        seed=9,
        initial_size=5,
        pde_size=7,
        predict_shape=(2, 3),
    )

    data_module.setup("fit")
    source = data_module.train_batch_source(("initial", "pde"), jax.random.key(4))

    assert isinstance(source, NamedBatchSource)
    assert data_module.pools["pde"].inputs.shape == (7, 2)
    initial_sampler = source.samplers["initial"]
    assert isinstance(initial_sampler, RandomRowSampler)
    assert isinstance(initial_sampler.pool["inputs"], np.ndarray)
    assert source(0)["initial"]["inputs"].shape == (5, 2)
    assert source(0)["pde"]["inputs"].shape == (3, 2)
    assert tuple(data_module.normalization_pools()) == ("pde",)
    prediction_source = data_module.predict_batch_source()
    assert isinstance(prediction_source, ChunkedPredictionSource)
    assert len(prediction_source) == 2


def test_burgers_data_module_generates_reproducible_fresh_uniform_pde_batches() -> None:
    """Verify continuous PDE sampling avoids a finite pool and follows global-step keys."""
    data_module = BurgersDataModule(
        {"initial": "all", "pde": 4, "predict": 4},
        pde_sampling="uniform",
        initial_size=5,
        predict_shape=(2, 3),
        time_bounds=(0.0, 2.0),
        space_bounds=(-3.0, 1.0),
    )

    data_module.setup("fit")
    source = data_module.train_batch_source(("initial", "pde"), jax.random.key(4))
    assert isinstance(source, NamedBatchSource)
    first = source(0)["pde"]
    repeated = source(0)["pde"]
    changed = source(1)["pde"]

    assert "pde" not in data_module.pools
    assert isinstance(source.samplers["pde"], UniformDomainSampler)
    assert isinstance(source.samplers["pde"].bounds, np.ndarray)
    assert first["inputs"].shape == (4, 2)
    assert first["targets"].shape == (4, 0)
    np.testing.assert_array_equal(first["inputs"], repeated["inputs"])
    assert not np.array_equal(first["inputs"], changed["inputs"])
    assert np.all((np.asarray(first["inputs"][:, 0]) >= 0.0) & (np.asarray(first["inputs"][:, 0]) < 2.0))
    assert np.all((np.asarray(first["inputs"][:, 1]) >= -3.0) & (np.asarray(first["inputs"][:, 1]) < 1.0))
    mean, std = data_module.input_statistics()
    np.testing.assert_allclose(mean, np.asarray([1.0, -1.0], dtype=np.float32))
    np.testing.assert_allclose(std, np.asarray([2.0, 4.0], dtype=np.float32) / np.sqrt(12.0))


def test_burgers_data_module_validates_stage_name_and_options() -> None:
    """Verify invalid application lifecycle configuration fails before compiled execution."""
    with pytest.raises(ValueError, match="name"):
        BurgersDataModule({"predict": 2}, name="")
    with pytest.raises(ValueError, match="pde_sampling"):
        BurgersDataModule({"predict": 2}, pde_sampling="adaptive")  # type: ignore[arg-type]
    data_module = BurgersDataModule({"initial": "all", "pde": 2}, initial_size=2, pde_size=2, predict_shape=(2, 2))
    with pytest.raises(ValueError, match="stage"):
        data_module.setup("test")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("predict_size", "match"),
    [(None, r"batch_sizes\['predict'\]"), (0, "batch"), (True, "batch")],
)
def test_burgers_prediction_requires_a_valid_chunk_size(predict_size: int | None, match: str) -> None:
    """Verify the PhiJAX default prediction source rejects a missing or invalid `predict` chunk size.

    Args:
        predict_size: Configured `predict` chunk size, or `None` to omit the entry.
        match: Expected validation-message fragment.
    """
    batch_sizes: dict[str, Any] = {"initial": "all", "pde": 2}
    if predict_size is not None:
        batch_sizes["predict"] = predict_size
    data_module = BurgersDataModule(batch_sizes, initial_size=2, pde_size=2, predict_shape=(2, 2))
    data_module.setup("predict")
    with pytest.raises(ValueError, match=match):
        data_module.predict_batch_source()


def test_burgers_prediction_accepts_one_chunk_for_the_whole_grid() -> None:
    """Verify `predict: all` evaluates the dense grid as one chunk."""
    data_module = BurgersDataModule(
        {"initial": "all", "pde": 2, "predict": "all"},
        initial_size=2,
        pde_size=2,
        predict_shape=(2, 3),
    )
    data_module.setup("predict")
    source = data_module.predict_batch_source()

    assert isinstance(source, ChunkedPredictionSource)
    assert source.pool is data_module.pools["predict"]
    assert len(source) == 1


def test_burgers_file_pools_keep_grid_layout_and_reference_targets() -> None:
    """Verify file-backed pools keep `[t, x]` order, the full reconstruction grid, and float32 storage."""
    arrays = _reference_arrays()
    pools = build_burgers_file_pools(arrays, seed=5, pde_size=6)

    assert set(pools) == {"initial", "pde", "predict"}
    for pool in pools.values():
        assert pool.inputs.dtype == np.float32
        assert pool.targets.dtype == np.float32
    grid_metadata = {
        "coordinate_names": ("t", "x"),
        "output_names": ("u",),
        "sampling_bounds": (("t", 0.0, 0.5), ("x", -1.0, 1.0)),
    }
    predict = pools["predict"]
    assert dict(predict.metadata) == grid_metadata
    assert predict.reference_shape == (3, 4)
    np.testing.assert_array_equal(predict.flat_index, np.arange(12))
    np.testing.assert_array_equal(predict.inputs[:4, 0], 0.0)
    np.testing.assert_array_equal(predict.inputs[:4, 1], arrays["x"].astype(np.float32))
    dense_targets = reconstruct_predictions(predict.targets, predict)
    np.testing.assert_array_equal(dense_targets[..., 0], arrays["usol"].astype(np.float32))

    # The initial pool is the first time row, reconstructed on the same full grid.
    initial = pools["initial"]
    assert dict(initial.metadata) == grid_metadata
    assert initial.reference_shape == (3, 4)
    np.testing.assert_array_equal(initial.flat_index, np.arange(4))
    np.testing.assert_array_equal(initial.inputs, predict.inputs[:4])
    np.testing.assert_array_equal(initial.targets, predict.targets[:4])

    pde = pools["pde"]
    assert dict(pde.metadata) == {"coordinate_names": ("t", "x"), "sampling_bounds": grid_metadata["sampling_bounds"]}
    assert pde.inputs.shape == (6, 2)
    assert pde.targets.shape == (6, 0)
    assert pde.reference_shape == (6,)
    np.testing.assert_array_equal(pde.flat_index, np.arange(6))
    assert np.all((pde.inputs[:, 0] >= 0.0) & (pde.inputs[:, 0] <= 0.5))
    assert np.all((pde.inputs[:, 1] >= -1.0) & (pde.inputs[:, 1] <= 1.0))


def test_burgers_file_pde_pool_is_explicitly_seeded_and_optional() -> None:
    """Verify file-backed PDE candidates follow the seed and are omitted for continuous sampling."""
    arrays = _reference_arrays()
    first = build_burgers_file_pools(arrays, seed=11, pde_size=5)["pde"]
    repeated = build_burgers_file_pools(arrays, seed=11, pde_size=5)["pde"]
    changed = build_burgers_file_pools(arrays, seed=12, pde_size=5)["pde"]

    np.testing.assert_array_equal(first.inputs, repeated.inputs)
    assert not np.array_equal(first.inputs, changed.inputs)
    assert set(build_burgers_file_pools(arrays, pde_size=None)) == {"initial", "predict"}


@pytest.mark.parametrize(
    ("replacements", "match"),
    [
        ({"usol": np.zeros((4, 3))}, "usol"),
        ({"t": np.asarray([0.0, 0.5, 0.25])}, "`t`"),
        ({"x": np.asarray([1.0])}, "`x`"),
        ({"pde_size": 0}, "pde_size"),
    ],
)
def test_burgers_file_pools_reject_invalid_arrays(replacements: dict[str, Any], match: str) -> None:
    """Verify transposed solutions, non-increasing coordinates, and empty PDE pools fail eagerly.

    Args:
        replacements: Array or size values replacing the valid synthetic inputs.
        match: Expected validation-message fragment.
    """
    arrays: dict[str, Any] = _reference_arrays()
    pde_size = replacements.pop("pde_size", 4)
    arrays.update(replacements)
    if "x" in replacements:
        arrays["usol"] = np.zeros((3, 1))
    with pytest.raises(ValueError, match=match):
        build_burgers_file_pools(arrays, pde_size=pde_size)


def test_burgers_data_module_samples_uniform_pde_batches_within_file_bounds(tmp_path: Path) -> None:
    """Verify a file-backed DataModule reads its uniform PDE sampling domain from the reference grid.

    Args:
        tmp_path: Temporary directory receiving the synthetic reference file.
    """
    data_path = tmp_path / "burgers.npz"
    arrays = _reference_arrays()
    np.savez(data_path, t=arrays["t"], x=arrays["x"], usol=arrays["usol"])
    data_module = BurgersDataModule(
        {"initial": "all", "pde": 8, "predict": 5},
        data_path=data_path,
        pde_sampling="uniform",
    )

    data_module.setup("fit")
    source = data_module.train_batch_source(("initial", "pde"), jax.random.key(0))

    assert set(data_module.pools) == {"initial", "predict"}
    assert isinstance(source, NamedBatchSource)
    batch = source(0)
    assert batch["initial"]["inputs"].shape == (4, 2)
    pde_inputs = np.asarray(batch["pde"]["inputs"])
    assert pde_inputs.shape == (8, 2)
    assert np.all((pde_inputs[:, 0] >= 0.0) & (pde_inputs[:, 0] <= 0.5))
    assert np.all((pde_inputs[:, 1] >= -1.0) & (pde_inputs[:, 1] <= 1.0))
    mean, std = data_module.input_statistics()
    np.testing.assert_allclose(mean, [0.25, 0.0])
    np.testing.assert_allclose(std, np.asarray([0.5, 2.0]) / np.sqrt(12.0), rtol=1e-6)
    data_module.setup("predict")
    prediction_source = data_module.predict_batch_source()
    assert isinstance(prediction_source, ChunkedPredictionSource)
    assert len(prediction_source) == 3
