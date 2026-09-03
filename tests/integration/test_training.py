import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, call

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from phijax import FitResult, PhiModule
from phijax.utils import register_task_finalizer

from phijax_hydra_template import train as train_module


class _PostTrainingPredictionTrainer:
    """Provide deterministic fit and in-memory prediction for entrypoint orchestration tests."""

    def __init__(
        self,
        output_directory: Path,
        *,
        interrupted: bool = False,
    ) -> None:
        """Initialize trainer services around one simulated prediction artifact.

        Args:
            output_directory: Directory receiving the simulated prediction artifact.
            interrupted: Whether fitting should report a graceful interruption.
        """
        output_path = output_directory / "predictions" / "burgers_1d.npz"
        self.prediction_writer = SimpleNamespace(output_path=output_path, artifact_path=None)
        self.logger = SimpleNamespace(loggers=())
        self.precision = object()
        self.closed = False
        self.interrupted = interrupted
        self.predicted_state: object | None = None

    def fit(self, *args: object, **kwargs: object) -> FitResult:
        """Return a completed one-step fit result.

        Args:
            *args: Positional training services ignored by the fixture.
            **kwargs: Keyword training services ignored by the fixture.

        Returns:
            Deterministic completed fit result.
        """
        module = args[0]
        datamodule: Any = kwargs.pop("datamodule")
        kwargs.pop("optimizer")
        kwargs.pop("seed")
        kwargs.pop("balancer")
        datamodule.prepare_stage("fit")
        datamodule.train_batch_source(("initial",), object())
        kwargs.clear()
        datamodule.teardown_stage("fit")
        return FitResult(
            module=cast(Any, module),
            state=cast(Any, object()),
            metrics={"train/loss": 1.0},
            stopped_early=False,
            interrupted=self.interrupted,
            iterations=1,
        )

    def predict(
        self,
        result: FitResult,
        batches: object = None,
        *,
        datamodule: Any,
    ) -> np.ndarray | None:
        """Predict from the fit result after setting up the DataModule.

        Args:
            result: Final fit result whose state is recorded for assertions.
            batches: Optional explicit batches, expected to be absent.
            datamodule: DataModule supplying optional prediction data.

        Returns:
            One synthetic prediction, or `None` when the DataModule has no prediction source.
        """
        assert batches is None
        self.predicted_state = result.state
        datamodule.prepare_stage("predict")
        try:
            if datamodule.predict_batch_source() is None:
                return None
            self.prediction_writer.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.prediction_writer.output_path.touch()
            self.prediction_writer.artifact_path = self.prediction_writer.output_path
            return np.zeros((1, 1), dtype=np.float32)
        finally:
            datamodule.teardown_stage("predict")

    def close(self) -> None:
        """Record checkpoint-manager closure before prediction assembly."""
        self.closed = True


def test_optimized_metric_reads_the_final_fit_result() -> None:
    """Verify a configured sweep objective is returned as a Python float."""
    config = OmegaConf.create({"optimized_metric": "train/loss"})
    result = FitResult(
        module=cast(Any, object()),
        state=cast(Any, object()),
        metrics=cast(dict[str, float], {"train/loss": np.float32(0.25)}),
        stopped_early=False,
        interrupted=False,
        iterations=1,
    )

    objective = train_module._get_optimized_metric(config, result)
    assert type(objective) is float
    assert objective == pytest.approx(0.25)


def test_optimized_metric_is_optional_for_normal_training() -> None:
    """Verify an ordinary run does not require an optimization objective."""
    config = OmegaConf.create({"optimized_metric": None})
    result = FitResult(
        module=cast(Any, object()),
        state=cast(Any, object()),
        metrics={},
        stopped_early=False,
        interrupted=False,
        iterations=1,
    )

    assert train_module._get_optimized_metric(config, result) is None


def test_optimized_metric_reports_available_metrics() -> None:
    """Verify a misspelled sweep objective fails with useful metric names."""
    config = OmegaConf.create({"optimized_metric": "validation/loss"})
    result = FitResult(
        module=cast(Any, object()),
        state=cast(Any, object()),
        metrics={"train/loss": 0.25},
        stopped_early=False,
        interrupted=False,
        iterations=1,
    )

    with pytest.raises(KeyError, match=r"validation/loss.*train/loss"):
        train_module._get_optimized_metric(config, result)


@pytest.mark.parametrize("interrupted", [False, True])
def test_training_prediction_reuses_final_state_and_data_module(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    interrupted: bool,
) -> None:
    """Verify post-fit prediction runs after successful or gracefully interrupted fitting.

    Args:
        monkeypatch: Pytest attribute patch helper.
        tmp_path: Temporary run output directory.
        interrupted: Whether the simulated fit reports interruption.
    """
    trainer = _PostTrainingPredictionTrainer(tmp_path, interrupted=interrupted)
    data_module = MagicMock()
    data_module.predict_batch_source.return_value = object()

    def build_trainer(config: object) -> _PostTrainingPredictionTrainer:
        """Register and return the shared trainer fixture.

        Args:
            config: Root configuration unused by the fixture.

        Returns:
            Shared trainer fixture.
        """
        del config
        register_task_finalizer(lambda _: trainer.close())
        return trainer

    monkeypatch.setattr(train_module, "build_trainer", build_trainer)
    monkeypatch.setattr(train_module, "instantiate_data_module", MagicMock(return_value=data_module))
    monkeypatch.setattr(train_module, "instantiate_model_factory", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(
        train_module,
        "instantiate_objective",
        MagicMock(return_value=SimpleNamespace(loss_names=("loss",))),
    )
    configured_module = MagicMock(spec=PhiModule)
    configured_module.loss_names = ("loss",)
    monkeypatch.setattr(train_module, "instantiate_module", MagicMock(return_value=configured_module))
    balancer = SimpleNamespace(initialize=lambda: object())
    monkeypatch.setattr(train_module, "instantiate_balancer", MagicMock(return_value=balancer))
    monkeypatch.setattr(train_module, "instantiate_optimizer", MagicMock(return_value=object()))
    config = OmegaConf.create(
        {
            "seed": 9,
            "predict": True,
            "application": {
                "name": "burgers",
                "mat_field_names": {"prediction": "pred"},
            },
            "data": {
                "name": "burgers_1d",
                "builder": {"_target_": "example.build_data"},
                "batch_size": {"predict": 4},
            },
            "model": {
                "module": {"_target_": "example.Module"},
                "net": {"_target_": "example.build_model"},
                "objective": {"_target_": "example.build_objective"},
                "balancer": {"_target_": "example.build_balancer"},
                "optimizer": {"_target_": "example.build_optimizer"},
            },
            "trainer": {"_target_": "example.Trainer"},
            "callbacks": {
                "model_checkpoint": {"_target_": "example.ModelCheckpoint"},
                "rich_progress_bar": {
                    "_target_": "phijax.callbacks.RichProgressBar",
                    "total": 10,
                    "predict_description": "Predicting",
                },
                "prediction_writer": {
                    "_target_": "phijax.callbacks.PredictionWriter",
                    "output_dir": "${oc.select:output_dir,null}",
                    "save_file_name": "${oc.select:save_file_name,predictions}",
                    "save_mat": True,
                    "mat_field_names": "${application.mat_field_names}",
                },
            },
            "logger": None,
            "paths": {"output_dir": str(tmp_path)},
            "ckpt_path": None,
            "ckpt_step": None,
            "weights_only": False,
        }
    )

    result = train_module.train(config)

    expected_path = tmp_path / "predictions" / "burgers_1d.npz"
    assert isinstance(result, FitResult)
    assert result.interrupted is interrupted
    assert trainer.closed is True
    assert trainer.predicted_state is result.state
    assert data_module.prepare_stage.call_args_list == [call("fit"), call("predict")]
    assert data_module.teardown_stage.call_args_list == [call("fit"), call("predict")]
    data_module.train_batch_source.assert_called_once()
    assert expected_path.is_file()


def test_training_task_runs_a_small_composed_experiment(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify the Hydra task assembles and executes the current PhiModule API end to end.

    Args:
        caplog: Pytest fixture capturing startup lifecycle messages.
        capsys: Pytest fixture capturing the environment and model-summary displays.
        monkeypatch: Pytest environment patch helper.
        tmp_path: Temporary output root.
    """
    repository_root = Path(__file__).parents[2]
    monkeypatch.setenv("PROJECT_ROOT", str(repository_root))
    config_dir = repository_root / "src" / "phijax_hydra_template" / "configs"
    with initialize_config_dir(config_dir=str(config_dir.resolve()), version_base=None):
        config = compose(
            config_name="train",
            overrides=[
                "experiment=burgers_grad_norm_1d",
                "data=burgers_analytic_1d",
                "model/balancer=static",
                "trainer.accelerator=cpu",
                "trainer.precision=32-true",
                "trainer.max_steps=1",
                "model.net.hidden=[4]",
                "model.net.fourier_features=false",
                "data.initial_size=4",
                "data.pde_size=4",
                "data.predict_shape=[2,2]",
                "data.batch_size.initial=2",
                "data.batch_size.pde=2",
                "~callbacks.model_checkpoint",
                f"paths.output_dir={tmp_path}",
            ],
        )

    with caplog.at_level(logging.INFO):
        result = train_module.train(config)

    assert isinstance(result, FitResult)
    assert result.iterations == 1
    assert set(config.model.objective.equations) == {"initial", "pde"}
    assert "train/loss" in result.metrics
    assert result.metrics["optimizer/lr-adamw"] == pytest.approx(0.0)
    log_dir = tmp_path / "phijax_logs" / "version_0"
    assert (log_dir / "hparams.yaml").is_file()
    assert (log_dir / "metrics.csv").is_file()
    standard_output = capsys.readouterr().out
    assert "Using 32-bit true precision" in standard_output
    assert "MLP Summary" in standard_output
    assert "Total Parameters: 21" in standard_output
    lifecycle_loggers = {
        "phijax_hydra_template.train",
        "phijax.integrations.hydra.assembly",
        "phijax.integrations.hydra.factory",
    }
    messages = [record.message for record in caplog.records if record.name in lifecycle_loggers]
    assert messages == [
        "Trainer seed set to 42.",
        "Instantiating trainer <phijax.Trainer>",
        "Using freshly initialized training state.",
        "Logging hyperparameters!",
        "Instantiating data module <phijax_hydra_template.applications.burgers.BurgersDataModule>",
        "Instantiating model factory <phijax.models.build_mlp>",
        "Instantiating objective <phijax.objectives.CompositeObjective.from_equations>",
        "Instantiating module <phijax.PhiModule>",
        "Instantiating loss balancer <phijax.balancers.StaticLossBalancer>",
        "Instantiating optimizer <optax.adamw>",
        "Starting training!",
        "Starting prediction from the final in-memory training state.",
        f"Predictions saved to <{tmp_path / 'predictions' / 'burgers_analytic_1d.npz'}>.",
    ]


def test_training_task_runs_a_grad_norm_refresh(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify Hydra assembly refreshes gradient-norm weights from the current training batch.

    Args:
        monkeypatch: Pytest environment patch helper.
        tmp_path: Temporary output root.
    """
    repository_root = Path(__file__).parents[2]
    monkeypatch.setenv("PROJECT_ROOT", str(repository_root))
    config_dir = repository_root / "src" / "phijax_hydra_template" / "configs"
    with initialize_config_dir(config_dir=str(config_dir.resolve()), version_base=None):
        config = compose(
            config_name="train",
            overrides=[
                "experiment=burgers_grad_norm_1d",
                "data=burgers_analytic_1d",
                "model/balancer=grad_norm",
                "model.balancer.update_every_n_steps=1",
                "model.balancer.update_start_step=0",
                "trainer.accelerator=cpu",
                "trainer.precision=32-true",
                "trainer.max_steps=1",
                "model.net.hidden=[2]",
                "model.net.fourier_features=false",
                "data.initial_size=2",
                "data.pde_size=2",
                "data.predict_shape=[2,2]",
                "data.batch_size.initial=2",
                "data.batch_size.pde=2",
                "~callbacks.model_checkpoint",
                f"paths.output_dir={tmp_path}",
            ],
        )

    result = train_module.train(config)

    assert isinstance(result, FitResult)
    weights = np.asarray([value for name, value in result.metrics.items() if name.startswith("train/weight/")])
    assert result.iterations == 1
    assert weights.shape == (2,)
    assert np.all(np.isfinite(weights))
    assert not np.allclose(weights, 1.0)


def test_training_task_runs_a_chunked_ntk_refresh(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify the memory-bounded NTK path supports the configured Burgers equations end to end.

    Args:
        monkeypatch: Pytest environment patch helper.
        tmp_path: Temporary output root.
    """
    repository_root = Path(__file__).parents[2]
    monkeypatch.setenv("PROJECT_ROOT", str(repository_root))
    config_dir = repository_root / "src" / "phijax_hydra_template" / "configs"
    with initialize_config_dir(config_dir=str(config_dir.resolve()), version_base=None):
        config = compose(
            config_name="train",
            overrides=[
                "experiment=burgers_grad_norm_1d",
                "data=burgers_analytic_1d",
                "model/balancer=ntk",
                "model.balancer.update_every_n_steps=1",
                "model.balancer.update_start_step=0",
                "model.balancer.kernel_size=2",
                "model.balancer.kernel_chunk_size=1",
                "trainer.accelerator=cpu",
                "trainer.precision=32-true",
                "trainer.max_steps=1",
                "model.net.hidden=[2]",
                "model.net.fourier_features=false",
                "data.initial_size=2",
                "data.pde_size=2",
                "data.predict_shape=[2,2]",
                "data.batch_size.initial=2",
                "data.batch_size.pde=2",
                "~callbacks.model_checkpoint",
                f"paths.output_dir={tmp_path}",
            ],
        )

    result = train_module.train(config)

    assert isinstance(result, FitResult)
    weights = np.asarray([value for name, value in result.metrics.items() if name.startswith("train/weight/")])
    assert result.iterations == 1
    assert weights.shape == (2,)
    assert np.all(np.isfinite(weights))
    assert not np.allclose(weights, 1.0)
