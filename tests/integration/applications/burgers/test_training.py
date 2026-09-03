from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from phijax import FitResult

from phijax_hydra_template import train as train_module


def test_burgers_experiment_runs_a_compiled_training_update(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify the Burgers pools, objective, residual derivatives, and trainer execute end to end.

    Args:
        monkeypatch: Pytest environment patch helper.
        tmp_path: Temporary output root.
    """
    repository_root = Path(__file__).parents[4]
    monkeypatch.setenv("PROJECT_ROOT", str(repository_root))
    config_dir = repository_root / "src" / "phijax_hydra_template" / "configs"
    with initialize_config_dir(config_dir=str(config_dir.resolve()), version_base=None):
        config = compose(
            config_name="train",
            overrides=[
                "experiment=burgers_grad_norm_1d",
                # Keep compiled API coverage lightweight without generating the full numerical reference fixture.
                "data=burgers_analytic_1d",
                "model.balancer.update_every_n_steps=1",
                "model.balancer.update_start_step=0",
                "trainer.accelerator=cpu",
                "trainer.max_steps=2",
                "trainer.log_every_n_steps=1",
                "model.net.hidden=[4]",
                "model.net.fourier_features_kwargs.embed_dim=2",
                "data.initial_size=4",
                "data.pde_sampling=uniform",
                "data.predict_shape=[2,2]",
                "data.batch_size.pde=2",
                "~callbacks.rich_model_summary",
                "~callbacks.rich_progress_bar",
                "~callbacks.model_checkpoint",
                f"paths.output_dir={tmp_path}",
            ],
        )

    result = train_module.train(config)

    assert isinstance(result, FitResult)
    assert result.iterations == 2
    assert {"train/loss", "train/loss/initial/data", "train/loss/pde/burgers"}.issubset(result.metrics)
    assert all(np.isfinite(float(result.metrics[name])) for name in result.metrics if name.startswith("train/loss"))
    weights = np.asarray([value for name, value in result.metrics.items() if name.startswith("train/weight/")])
    assert weights.shape == (2,)
    assert np.all(np.isfinite(weights))
