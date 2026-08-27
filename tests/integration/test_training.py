import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import jax
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from phijax.models import InitializedModel
from phijax.training import FitResult, TrainingPlan
from phijax.utils import register_task_finalizer

from pinn_project import train as train_module


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

    def print_environment_info(self) -> None:
        """Skip runtime output for the orchestration fixture."""
        return None

    def initialize_state(self, *args: object) -> object:
        """Return a synthetic functional state.

        Args:
            *args: Model, optimizer, balancer, and key values ignored by the fixture.

        Returns:
            Synthetic state placeholder.
        """
        del args
        return object()

    def fit(self, *args: object, **kwargs: object) -> FitResult:
        """Return a completed one-step fit result.

        Args:
            *args: Positional training services ignored by the fixture.
            **kwargs: Keyword training services ignored by the fixture.

        Returns:
            Deterministic completed fit result.
        """
        del args
        datamodule = kwargs.pop("datamodule")
        sampling_key = kwargs.pop("sampling_key")
        datamodule.prepare_stage("fit")
        datamodule.train_batch_source(("initial",), sampling_key)
        kwargs.clear()
        datamodule.teardown_stage("fit")
        return FitResult(
            state=object(),
            metrics={"train/loss": 1.0},
            stopped_early=False,
            interrupted=self.interrupted,
            iterations=1,
        )

    def predict(
        self,
        module: object,
        state: object,
        batches: object = None,
        *,
        datamodule: object,
    ) -> np.ndarray | None:
        """Predict from the fit result after setting up the DataModule.

        Args:
            module: Configured module unused by the fixture.
            state: Final in-memory fit state recorded for assertions.
            batches: Optional explicit batches, expected to be absent.
            datamodule: DataModule supplying optional prediction data.

        Returns:
            One synthetic prediction, or `None` when the DataModule has no prediction source.
        """
        del module
        assert batches is None
        self.predicted_state = state
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


def test_training_task_applies_root_seed_before_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify the generic Hydra task globally seeds host RNGs before training assembly.

    Args:
        monkeypatch: Pytest attribute patch helper.
        tmp_path: Temporary output directory fixture.
    """
    seeded: list[int] = []
    monkeypatch.setattr(train_module, "seed_everything", lambda seed: seeded.append(seed))
    config = OmegaConf.create({"seed": 41, "paths": {"output_dir": str(tmp_path)}})
    train_module.train(config)
    assert seeded == [41]


def test_training_task_resolves_and_logs_null_seed(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify `seed: null` becomes concrete before assembly and experiment logging.

    Args:
        caplog: Pytest fixture capturing the resolved-seed lifecycle message.
        monkeypatch: Pytest attribute patch helper.
        tmp_path: Temporary output directory fixture.
    """
    seeded: list[int] = []
    monkeypatch.setattr(train_module, "resolve_seed", lambda seed: 271828 if seed is None else seed)
    monkeypatch.setattr(train_module, "seed_everything", lambda seed: seeded.append(seed))
    config = OmegaConf.create({"seed": None, "paths": {"output_dir": str(tmp_path)}})

    with caplog.at_level(logging.INFO, logger="pinn_project.train"):
        train_module.train(config)

    assert config.seed == 271828
    assert seeded == [271828]
    assert "Global seed set to 271828." in caplog.text


def test_bootstrap_description_reports_selected_application() -> None:
    """Verify the generic entrypoint identifies a composed application without knowing its implementation."""
    generic = OmegaConf.create({})
    application = OmegaConf.create({"application": {"name": "example"}})
    assert "application" not in train_module.describe_bootstrap(generic)
    assert "`example` application" in train_module.describe_bootstrap(application)


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
    train_step = MagicMock()
    state = object()

    def instantiate_trainer(config: object, callbacks: object) -> _PostTrainingPredictionTrainer:
        """Register and return the shared trainer fixture.

        Args:
            config: Trainer configuration unused by the fixture.
            callbacks: Callback instances unused by the fixture.

        Returns:
            Shared trainer fixture.
        """
        del config, callbacks
        register_task_finalizer(lambda _: trainer.close())
        return trainer

    monkeypatch.setattr(train_module, "seed_everything", lambda seed: jax.random.key(seed))
    monkeypatch.setattr(train_module, "instantiate_callbacks", MagicMock(return_value=()))
    monkeypatch.setattr(train_module, "instantiate_trainer", instantiate_trainer)
    monkeypatch.setattr(train_module, "instantiate_loggers", MagicMock(return_value=trainer.logger))
    monkeypatch.setattr(train_module, "instantiate_data_module", MagicMock(return_value=data_module))
    monkeypatch.setattr(
        train_module,
        "instantiate_model",
        MagicMock(return_value=InitializedModel(lambda model_state, inputs: inputs, state)),
    )
    monkeypatch.setattr(
        train_module,
        "instantiate_objective",
        MagicMock(return_value=SimpleNamespace(loss_names=("loss",))),
    )
    configured_module = SimpleNamespace(loss_names=("loss",))
    monkeypatch.setattr(train_module, "instantiate_module", MagicMock(return_value=configured_module))
    balancer = SimpleNamespace(initialize=lambda: object())
    monkeypatch.setattr(train_module, "instantiate_balancer", MagicMock(return_value=balancer))
    monkeypatch.setattr(train_module, "instantiate_optimizer", MagicMock(return_value=object()))
    monkeypatch.setattr(
        train_module,
        "configure_training",
        MagicMock(return_value=TrainingPlan(train_step, ("initial",))),
    )
    config = OmegaConf.create(
        {
            "seed": 9,
            "bootstrap_only": False,
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
                "balancer": {"factory": {"_target_": "example.build_balancer"}},
                "optimizer": {"_target_": "example.build_optimizer"},
            },
            "trainer": {"_target_": "example.Trainer"},
            "callbacks": {
                "model_checkpoint": {"enabled": True},
                "rich_progress_bar": {
                    "enabled": True,
                    "_target_": "phijax.callbacks.RichProgressBar",
                    "total": 10,
                    "predict_description": "Predicting",
                },
                "prediction_writer": {
                    "enabled": True,
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
    config_dir = repository_root / "src" / "pinn_project" / "configs"
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
                "callbacks.model_checkpoint.enabled=false",
                f"paths.output_dir={tmp_path}",
            ],
        )

    with caplog.at_level(logging.INFO):
        result = train_module.train(config)

    assert isinstance(result, FitResult)
    assert result.iterations == 1
    assert set(config.model.objective.terms) == {"initial", "pde"}
    assert "train/loss" in result.metrics
    assert result.metrics["train/lr"] == pytest.approx(0.0)
    standard_output = capsys.readouterr().out
    assert "Using 32-bit true precision" in standard_output
    assert "MLP Summary" in standard_output
    assert "Total Parameters: 21" in standard_output
    lifecycle_loggers = {
        "pinn_project.train",
        "phijax.integrations.hydra.assembly",
        "phijax.integrations.hydra.factory",
    }
    messages = [record.message for record in caplog.records if record.name in lifecycle_loggers]
    assert messages == [
        "Global seed set to 42.",
        "Instantiating callbacks...",
        "Instantiating trainer <phijax.training.Trainer>",
        "Instantiating loggers...",
        "Using freshly initialized training state.",
        "Instantiating data module <pinn_project.applications.burgers.BurgersDataModule>",
        "Instantiating model <phijax.models.build_mlp>",
        "Instantiating objective <phijax.objectives.CompositeObjective>",
        "Instantiating module <phijax.module.PhiModule>",
        "Instantiating loss balancer <phijax.balancers.StaticLossBalancer>",
        "Instantiating optimizer <optax.adamw>",
        "Starting training!",
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
    config_dir = repository_root / "src" / "pinn_project" / "configs"
    with initialize_config_dir(config_dir=str(config_dir.resolve()), version_base=None):
        config = compose(
            config_name="train",
            overrides=[
                "experiment=burgers_grad_norm_1d",
                "data=burgers_analytic_1d",
                "model/balancer=grad_norm",
                "model.balancer.update.every_n_steps=1",
                "model.balancer.update.skip_first_step=false",
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
                "callbacks.model_checkpoint.enabled=false",
                f"paths.output_dir={tmp_path}",
            ],
        )

    result = train_module.train(config)

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
    config_dir = repository_root / "src" / "pinn_project" / "configs"
    with initialize_config_dir(config_dir=str(config_dir.resolve()), version_base=None):
        config = compose(
            config_name="train",
            overrides=[
                "experiment=burgers_grad_norm_1d",
                "data=burgers_analytic_1d",
                "model/balancer=ntk",
                "model.balancer.update.every_n_steps=1",
                "model.balancer.update.skip_first_step=false",
                "model.balancer.update.kernel_size=2",
                "model.balancer.update.kernel_chunk_size=1",
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
                "callbacks.model_checkpoint.enabled=false",
                f"paths.output_dir={tmp_path}",
            ],
        )

    result = train_module.train(config)

    weights = np.asarray([value for name, value in result.metrics.items() if name.startswith("train/weight/")])
    assert result.iterations == 1
    assert weights.shape == (2,)
    assert np.all(np.isfinite(weights))
    assert not np.allclose(weights, 1.0)
