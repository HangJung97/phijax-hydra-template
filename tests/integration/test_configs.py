import os
from functools import partial
from pathlib import Path

import jax
import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from phijax.objectives import CompositeObjective, ResidualTerm

from phijax_hydra_template.configs import import_from_module


@pytest.fixture
def config_dir() -> Path:
    """Return the absolute project config directory.

    Returns:
        Project Hydra config directory.
    """
    return Path(__file__).parents[2] / "src" / "phijax_hydra_template" / "configs"


def test_suite_initializes_jax_on_cpu() -> None:
    """Verify local and CI tests force JAX to initialize only its CPU backend."""
    assert os.environ["JAX_PLATFORMS"] == "cpu"
    assert jax.default_backend() == "cpu"


def test_config_package_exposes_phijax_omegaconf_helpers() -> None:
    """Verify the config package exposes PhiJAX's public resolver helpers."""
    assert import_from_module("math.pi") == pytest.approx(3.141592653589793)
    assert OmegaConf.has_resolver("assert")
    assert OmegaConf.has_resolver("op")
    assert OmegaConf.has_resolver("op.ternary")
    assert OmegaConf.has_resolver("call")
    assert OmegaConf.has_resolver("tuple")
    assert OmegaConf.has_resolver("target.name")
    assert OmegaConf.has_resolver("math")


def test_burgers_experiment_composes_complete_runtime_policy(config_dir: Path) -> None:
    """Verify the reference experiment supplies every runnable project group.

    Args:
        config_dir: Project Hydra config directory.
    """
    with initialize_config_dir(config_dir=str(config_dir.resolve()), version_base=None):
        config = compose(config_name="train", overrides=["experiment=burgers_grad_norm_1d"])

    assert config.task_name == "burgers_grad_norm_1d"
    assert config.tags == ["burgers", "grad_norm"]
    assert config.application.name == "burgers"
    assert config.data._target_ == "phijax_hydra_template.applications.burgers.BurgersDataModule"
    assert config.model.net._target_ == "phijax.models.build_mlp"
    assert config.model.module._target_ == "phijax.PhiModule"
    assert config.model.objective._target_ == "phijax.objectives.CompositeObjective.from_equations"
    assert config.model.balancer._target_ == "phijax.balancers.GradNormBalancer"
    assert config.model.optimizer._target_ == "optax.adamw"
    assert config.callbacks.lr_monitor.optimizer_name == "adamw"
    assert config.callbacks.lr_monitor.log_key_prefix == "optimizer/"
    assert config.trainer.matmul_precision == "highest"
    assert "deterministic" not in config.trainer
    assert config.logger._target_ == "phijax.training.CSVLogger"
    assert set(config.callbacks) == {
        "lr_monitor",
        "rich_model_summary",
        "rich_progress_bar",
        "model_checkpoint",
        "prediction_writer",
    }
    assert all("enabled" not in callback for callback in config.callbacks.values())
    objective = instantiate(config.model.objective)
    assert isinstance(objective, CompositeObjective)
    assert isinstance(objective.terms["initial"], ResidualTerm)
    pde_term = objective.terms["pde"]
    assert isinstance(pde_term, ResidualTerm)
    assert isinstance(pde_term.residual_fn, partial)
    assert objective.loss_names == ("initial/data", "pde/burgers")


@pytest.mark.parametrize(
    ("optimizer_name", "target", "momentum", "weight_decay"),
    [
        ("adamw", "optax.adamw", 0.9, 0.01),
        ("adam", "optax.adam", 0.9, 0.0),
        ("sgd", "optax.sgd", 0.9, 0.0),
        ("soap", "soap_jax.soap", 0.9, 0.0),
    ],
)
def test_optimizer_configs_compose_with_monitor_metadata(
    config_dir: Path,
    optimizer_name: str,
    target: str,
    momentum: float,
    weight_decay: float,
) -> None:
    """Verify each optimizer composes its schedule and learning-rate monitor metadata.

    Args:
        config_dir: Project Hydra config directory.
        optimizer_name: Optimizer config selected from the model group.
        target: Expected public Optax factory target.
        momentum: Expected momentum value forwarded to the monitor.
        weight_decay: Expected weight decay forwarded to the monitor.
    """
    with initialize_config_dir(config_dir=str(config_dir.resolve()), version_base=None):
        config = compose(
            config_name="train",
            overrides=["experiment=burgers_grad_norm_1d", f"model/optimizer={optimizer_name}"],
        )

    assert config.model.optimizer._target_ == target
    assert config.callbacks.lr_monitor.optimizer_name == optimizer_name
    assert config.callbacks.lr_monitor.momentum == pytest.approx(momentum)
    assert config.callbacks.lr_monitor.weight_decay == pytest.approx(weight_decay)


@pytest.mark.parametrize("config_name", ["train", "predict", "evaluate_predictions"])
def test_every_hydra_root_config_composes(config_dir: Path, config_name: str) -> None:
    """Verify each project entrypoint has a valid root composition.

    Args:
        config_dir: Project Hydra config directory.
        config_name: Root config selected by one entrypoint.
    """
    overrides = ["predictions=/tmp/predictions.npz"] if config_name == "evaluate_predictions" else []
    with initialize_config_dir(config_dir=str(config_dir.resolve()), version_base=None):
        config = compose(config_name=config_name, overrides=overrides, return_hydra_config=True)

    raw_config = OmegaConf.to_container(config, resolve=False)
    assert isinstance(raw_config, dict)
    assert "${paths.log_dir}" in raw_config["hydra"]["run"]["dir"]


def test_prediction_callbacks_compose_progress_and_artifact_writing(config_dir: Path) -> None:
    """Verify prediction selects the project callback suite.

    Args:
        config_dir: Project Hydra config directory.
    """
    with initialize_config_dir(config_dir=str(config_dir.resolve()), version_base=None):
        config = compose(
            config_name="predict",
            overrides=["experiment=burgers_grad_norm_1d", "ckpt_path=/tmp/checkpoints"],
        )

    assert config.callbacks.rich_progress_bar._target_ == "phijax.callbacks.RichProgressBar"
    assert config.callbacks.prediction_writer._target_ == "phijax.callbacks.PredictionWriter"
    assert config.callbacks.prediction_writer.save_file_name == config.data.name


def test_burgers_optuna_search_composes_trial_policy(config_dir: Path) -> None:
    """Verify the reference search selects Optuna and avoids costly trial artifacts.

    Args:
        config_dir: Project Hydra config directory.
    """
    with initialize_config_dir(config_dir=str(config_dir.resolve()), version_base=None):
        config = compose(
            config_name="train",
            overrides=["experiment=burgers_grad_norm_1d", "hparams_search=burgers_optuna"],
            return_hydra_config=True,
        )

    assert config.optimized_metric == "train/loss"
    assert config.predict is False
    assert set(config.callbacks) == {"rich_model_summary", "rich_progress_bar"}
    assert config.get("logger") is None
    assert config.hydra.sweeper._target_ == "hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper"
    assert config.hydra.sweeper.direction == "minimize"
    assert config.hydra.sweeper.n_trials == 20
    assert set(config.hydra.sweeper.params) == {
        "model.scheduler.peak_value",
        "model.optimizer.weight_decay",
        "model.net.fourier_features_kwargs.scale",
        "model.balancer.moving_average_coefficient",
    }


def test_local_example_is_documentation_not_an_automatic_override(config_dir: Path) -> None:
    """Verify the committed local example remains opt-in and machine-neutral.

    Args:
        config_dir: Project Hydra config directory.
    """
    example = config_dir / "local" / "example.yaml"
    assert example.is_file()
    assert not (config_dir / "local" / "default.yaml").exists()
