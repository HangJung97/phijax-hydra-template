import jax
import numpy as np
import pytest
from phijax.data import RandomRowSampler, UniformDomainSampler, reconstruct_predictions

from pinn_project.applications.burgers import BurgersDataModule, build_burgers_pools


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
def test_burgers_pools_reject_invalid_geometry(kwargs: dict[str, object], match: str) -> None:
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

    assert data_module.pools["pde"].inputs.shape == (7, 2)
    initial_sampler = source.samplers["initial"]
    assert isinstance(initial_sampler, RandomRowSampler)
    assert isinstance(initial_sampler.pool["inputs"], np.ndarray)
    assert source(0)["initial"]["inputs"].shape == (5, 2)
    assert source(0)["pde"]["inputs"].shape == (3, 2)
    assert tuple(data_module.normalization_pools()) == ("pde",)
    assert len(data_module.predict_batch_source()) == 2


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


def test_burgers_data_module_validates_stage_name_and_prediction_batch_size() -> None:
    """Verify invalid application lifecycle configuration fails before compiled execution."""
    with pytest.raises(ValueError, match="name"):
        BurgersDataModule({"predict": 2}, name="")
    with pytest.raises(ValueError, match="pde_sampling"):
        BurgersDataModule({"predict": 2}, pde_sampling="adaptive")  # type: ignore[arg-type]
    data_module = BurgersDataModule(
        {"initial": "all", "pde": 2, "predict": "all"},
        initial_size=2,
        pde_size=2,
        predict_shape=(2, 2),
    )
    with pytest.raises(ValueError, match="stage"):
        data_module.setup("test")  # type: ignore[arg-type]
    data_module.setup("predict")
    with pytest.raises(ValueError, match="positive integer"):
        data_module.predict_batch_source()
