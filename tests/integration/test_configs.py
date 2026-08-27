from functools import partial
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from phijax.objectives import CompositeObjective, ResidualTerm


@pytest.fixture
def config_dir() -> Path:
    """Return the absolute project config directory.

    Returns:
        Project Hydra config directory.
    """
    return Path(__file__).parents[2] / "src" / "pinn_project" / "configs"


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
    assert config.data._target_ == "pinn_project.applications.burgers.BurgersDataModule"
    assert config.model.net._target_ == "phijax.models.build_mlp"
    assert config.model.module._target_ == "phijax.module.PhiModule"
    assert config.model.objective._target_ == "phijax.objectives.CompositeObjective"
    assert config.model.optimizer._target_ == "optax.adamw"
    objective = instantiate(config.model.objective)
    assert isinstance(objective, CompositeObjective)
    assert isinstance(objective.terms["initial"], ResidualTerm)
    assert isinstance(objective.terms["pde"].residual_fn, partial)
    assert objective.loss_names == ("initial/u", "pde/burgers")


@pytest.mark.parametrize("config_name", ["train", "predict", "evaluate_predictions"])
def test_every_hydra_root_config_composes(config_dir: Path, config_name: str) -> None:
    """Verify each project entrypoint has a valid bootstrap composition.

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


def test_local_example_is_documentation_not_an_automatic_override(config_dir: Path) -> None:
    """Verify the committed local example remains opt-in and machine-neutral.

    Args:
        config_dir: Project Hydra config directory.
    """
    example = config_dir / "local" / "example.yaml"
    assert example.is_file()
    assert not (config_dir / "local" / "default.yaml").exists()
