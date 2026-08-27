from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf
from phijax.callbacks import PredictionContext, PredictionWriter
from phijax.models import InitializedModel
from phijax.utils import register_task_finalizer

from pinn_project import predict as predict_module
from pinn_project.applications.burgers import build_burgers_pools


class _PredictionTrainer:
    """Provide a deterministic prediction runner for entrypoint artifact tests."""

    def __init__(self, callbacks: tuple[PredictionWriter, ...] = (), pool: object | None = None) -> None:
        """Initialize callbacks, reconstruction data, and an open trainer marker.

        Args:
            callbacks: Prediction callbacks dispatched by the fixture.
            pool: Optional reconstruction pool supplied to final callback contexts.
        """
        self.closed = False
        self.callbacks = callbacks
        self.prediction_writer = callbacks[0] if callbacks else None
        self.pool = pool
        self.strategy = SimpleNamespace(is_global_zero=True)
        self.precision = object()
        self.ckpt_path: str | None = None
        self.ckpt_step: int | None = None

    def predict(
        self,
        module: object,
        state: object,
        batches: object = None,
        *,
        ckpt_path: str,
        ckpt_step: int | None,
        datamodule: object,
    ) -> jax.Array:
        """Return one flat synthetic prediction array.

        Args:
            module: Ignored module placeholder.
            state: Ignored state placeholder.
            batches: Optional explicit batch source, expected to be omitted.
            ckpt_path: Checkpoint root forwarded by the entrypoint.
            ckpt_step: Optional exact checkpoint step.
            datamodule: Prediction DataModule torn down by the fixture.

        Returns:
            Six scalar prediction rows.
        """
        del module, state
        assert batches is None
        self.ckpt_path = ckpt_path
        self.ckpt_step = ckpt_step
        datamodule.prepare_stage("predict")
        datamodule.predict_batch_source()
        datamodule.teardown_stage("predict")
        predictions = jnp.arange(6, dtype=jnp.float32).reshape(6, 1)
        for callback in self.callbacks:
            callback.setup()
            callback.on_predict_end(
                PredictionContext(
                    outputs=np.asarray(predictions),
                    batch_index=None,
                    metadata={},
                    pool=self.pool,
                )
            )
            callback.teardown()
        return predictions

    def close(self) -> None:
        """Record trainer closure."""
        self.closed = True

    def initialize_state(self, *args: object) -> object:
        """Return a synthetic checkpoint-compatible state.

        Args:
            *args: Model, optimizer, balancer, and key values ignored by the fixture.

        Returns:
            Synthetic state placeholder.
        """
        del args
        return object()


def test_prediction_task_saves_flat_and_dense_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify the prediction entrypoint reconstructs and saves callback-produced model outputs.

    Args:
        monkeypatch: Pytest attribute patch helper.
        tmp_path: Temporary artifact directory.
    """
    data_module = MagicMock()
    pool = build_burgers_pools(predict_shape=(2, 3))["predict"]
    output_path = tmp_path / "prediction.npz"
    writer = PredictionWriter(
        tmp_path, save_file_name="prediction", save_mat=True, mat_field_names={"prediction": "pred"}
    )
    trainer = _PredictionTrainer((writer,), pool)
    state = object()

    def instantiate_trainer(config: object, callbacks: object) -> _PredictionTrainer:
        """Register and return the prediction trainer fixture.

        Args:
            config: Ignored trainer configuration.
            callbacks: Ignored callback collection.

        Returns:
            Prediction trainer fixture.
        """
        del config, callbacks
        register_task_finalizer(lambda status: trainer.close())
        return trainer

    monkeypatch.setattr(predict_module, "seed_everything", lambda seed: jax.random.key(seed))
    monkeypatch.setattr(predict_module, "instantiate_callbacks", MagicMock(return_value=()))
    monkeypatch.setattr(predict_module, "instantiate_trainer", instantiate_trainer)
    monkeypatch.setattr(predict_module, "instantiate_loggers", MagicMock(return_value=object()))
    monkeypatch.setattr(predict_module, "instantiate_data_module", MagicMock(return_value=data_module))
    monkeypatch.setattr(
        predict_module,
        "instantiate_model",
        MagicMock(return_value=InitializedModel(lambda model_state, inputs: inputs, state)),
    )
    monkeypatch.setattr(
        predict_module,
        "instantiate_objective",
        MagicMock(return_value=SimpleNamespace(loss_names=("loss",))),
    )
    configured_module = SimpleNamespace(loss_names=("loss",))
    monkeypatch.setattr(predict_module, "instantiate_module", MagicMock(return_value=configured_module))
    balancer = SimpleNamespace(initialize=lambda: object())
    monkeypatch.setattr(predict_module, "instantiate_balancer", MagicMock(return_value=balancer))
    monkeypatch.setattr(predict_module, "instantiate_optimizer", MagicMock(return_value=object()))
    config = OmegaConf.create(
        {
            "seed": 4,
            "bootstrap_only": False,
            "ckpt_path": str(tmp_path / "checkpoints"),
            "ckpt_step": None,
            "application": {"name": "test"},
            "data": {"_target_": "example.DataModule"},
            "model": {
                "module": {"_target_": "example.Module"},
                "net": {"_target_": "example.Model"},
                "objective": {"_target_": "example.Objective"},
                "balancer": {"factory": {"_target_": "example.Balancer"}},
                "optimizer": {"_target_": "example.Optimizer"},
            },
            "trainer": {"_target_": "example.Trainer"},
            "output_dir": str(tmp_path),
            "save_file_name": "prediction",
            "paths": {"output_dir": str(tmp_path)},
        }
    )

    result = predict_module.predict(config)

    assert result == output_path
    assert trainer.closed is True
    assert trainer.ckpt_path == str(tmp_path / "checkpoints")
    assert trainer.ckpt_step is None
    data_module.prepare_stage.assert_called_once_with("predict")
    data_module.predict_batch_source.assert_called_once_with()
    data_module.teardown_stage.assert_called_once_with("predict")
    with np.load(output_path) as artifact:
        assert artifact["flat_prediction"].shape == (6, 1)
        assert artifact["prediction"].shape == (2, 3, 1)
        assert artifact["flat_target"].shape == (6, 0)
        assert artifact["target"].shape == (2, 3, 0)
        assert artifact["dense_inputs"].shape == (2, 3, 2)
        assert artifact["mask"].shape == (2, 3)
        assert artifact["mask"].all()
        assert artifact["coordinate_names"].tolist() == ["t", "x"]
        assert artifact["output_names"].tolist() == ["u"]
        np.testing.assert_allclose(artifact["output_scales"], [1.0])
        assert artifact["artifact_schema_version"].item() == 2
        assert artifact["value_space"].item() == "physical"
    assert output_path.with_suffix(".mat").is_file()


def test_prediction_task_requires_a_checkpoint_for_executable_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify prediction reports a task-specific error instead of failing during config rendering.

    Args:
        monkeypatch: Pytest attribute patch helper.
    """
    monkeypatch.setattr(predict_module, "seed_everything", lambda seed: jax.random.key(seed))
    config = OmegaConf.create(
        {
            "seed": 4,
            "bootstrap_only": False,
            "ckpt_path": "???",
            "ckpt_step": None,
            "paths": {"output_dir": "."},
        }
    )
    with pytest.raises(ValueError, match="Prediction requires `ckpt_path`"):
        predict_module.predict(config)
