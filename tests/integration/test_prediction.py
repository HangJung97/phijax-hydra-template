from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, call

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf
from phijax import PhiModule
from phijax.callbacks import PredictionContext, PredictionWriter
from phijax.data import HostPool
from phijax.utils import register_task_finalizer

from phijax_hydra_template import predict as predict_module
from phijax_hydra_template.applications.burgers import build_burgers_pools


class _PredictionTrainer:
    """Provide a deterministic prediction runner for entrypoint artifact tests."""

    def __init__(self, callbacks: tuple[PredictionWriter, ...] = (), pool: HostPool | None = None) -> None:
        """Initialize callbacks, reconstruction data, and an open trainer marker.

        Args:
            callbacks: Prediction callbacks dispatched by the fixture.
            pool: Optional reconstruction pool supplied to final callback contexts.
        """
        self.closed = False
        self.callbacks = callbacks
        self.prediction_writer = callbacks[0] if callbacks else None
        self.pool = pool
        self.strategy = SimpleNamespace(is_global_zero=True, root_device=jax.devices()[0])
        self.precision = object()
        self.ckpt_path: str | None = None
        self.ckpt_step: int | None = None

    def predict_state(
        self,
        module: object,
        state: object,
        batches: object = None,
        *,
        ckpt_path: str,
        ckpt_step: int | None,
        datamodule: Any,
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

    def initialize_state(self, *args: object, **kwargs: object) -> object:
        """Return a synthetic checkpoint-compatible state.

        Args:
            *args: Model, optimizer, balancer, and key values ignored by the fixture.
            **kwargs: Sampling and balancing keys ignored by the fixture.

        Returns:
            Synthetic state placeholder.
        """
        del args, kwargs
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
    data_module.input_statistics.return_value = None
    pool = build_burgers_pools(predict_shape=(2, 3))["predict"]
    output_path = tmp_path / "prediction.npz"
    writer = PredictionWriter(
        tmp_path, save_file_name="prediction", save_mat=True, mat_field_names={"prediction": "pred"}
    )
    trainer = _PredictionTrainer((writer,), pool)
    state = object()

    def build_trainer(config: object) -> _PredictionTrainer:
        """Register and return the prediction trainer fixture.

        Args:
            config: Ignored root configuration.

        Returns:
            Prediction trainer fixture.
        """
        del config
        register_task_finalizer(lambda status: trainer.close())
        return trainer

    monkeypatch.setattr(predict_module, "build_trainer", build_trainer)
    monkeypatch.setattr(predict_module, "instantiate_data_module", MagicMock(return_value=data_module))
    monkeypatch.setattr(predict_module, "instantiate_model_factory", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(
        predict_module,
        "instantiate_objective",
        MagicMock(return_value=SimpleNamespace(loss_names=("loss",))),
    )
    configured_module = MagicMock(spec=PhiModule)
    configured_module.loss_names = ("loss",)
    configured_module.prepare_model.return_value = (configured_module, state)
    monkeypatch.setattr(predict_module, "instantiate_module", MagicMock(return_value=configured_module))
    balancer = SimpleNamespace(initialize=lambda: object())
    monkeypatch.setattr(predict_module, "instantiate_balancer", MagicMock(return_value=balancer))
    monkeypatch.setattr(predict_module, "instantiate_optimizer", MagicMock(return_value=object()))
    config = OmegaConf.create(
        {
            "seed": 4,
            "ckpt_path": str(tmp_path / "checkpoints"),
            "ckpt_step": None,
            "application": {"name": "test"},
            "data": {"_target_": "example.DataModule"},
            "model": {
                "module": {"_target_": "example.Module"},
                "net": {"_target_": "example.Model"},
                "objective": {"_target_": "example.Objective"},
                "balancer": {"_target_": "example.Balancer"},
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
    assert data_module.prepare_stage.call_args_list == [call("fit"), call("predict")]
    data_module.predict_batch_source.assert_called_once_with()
    assert data_module.teardown_stage.call_args_list == [call("fit"), call("predict")]
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
    config = OmegaConf.create(
        {
            "seed": 4,
            "ckpt_path": "???",
            "ckpt_step": None,
            "paths": {"output_dir": "."},
        }
    )
    with pytest.raises(ValueError, match="Prediction requires `ckpt_path`"):
        predict_module.predict(config)
