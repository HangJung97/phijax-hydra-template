from pathlib import Path

import hydra
import jax
from omegaconf import DictConfig, OmegaConf
from phijax.integrations.hydra import (
    instantiate_balancer,
    instantiate_callbacks,
    instantiate_data_module,
    instantiate_loggers,
    instantiate_model,
    instantiate_module,
    instantiate_objective,
    instantiate_optimizer,
    instantiate_trainer,
)
from phijax.utils import RankedLogger, extras, pre_hydra_routine, seed_everything, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def predict(cfg: DictConfig) -> Path | None:
    """Restore a composed experiment, run prediction, and save a dense artifact.

    Args:
        cfg: Hydra prediction configuration containing an experiment, checkpoint root, and artifact destination.

    Returns:
        Saved prediction artifact path, or `None` when `bootstrap_only` is enabled.

    Raises:
        ValueError: If an executable prediction run has no checkpoint root.
    """
    key = seed_everything(int(cfg.seed))
    ckpt_path = None if OmegaConf.is_missing(cfg, "ckpt_path") else cfg.get("ckpt_path")
    if bool(cfg.get("bootstrap_only", True)):
        checkpoint_description = ckpt_path if ckpt_path else "no checkpoint"
        print(f"Prediction configured with {checkpoint_description}; enable an experiment to run.")
        return None
    if ckpt_path is None or not str(ckpt_path).strip():
        raise ValueError(
            "Prediction requires `ckpt_path` to identify a trained Orbax checkpoint root, for example "
            "`phijax-predict experiment=<name> ckpt_path=/path/to/checkpoints`.",
        )
    checkpoint_step = cfg.get("ckpt_step")
    model_key, _, _, state_key = jax.random.split(key, 4)
    log.info("Instantiating callbacks...")
    callbacks = instantiate_callbacks(cfg.get("callbacks"))
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer = instantiate_trainer(cfg.trainer, callbacks)
    log.info("Instantiating loggers...")
    trainer.logger = instantiate_loggers(cfg.get("logger"), trainer)
    writer = trainer.prediction_writer
    if writer is None:
        raise ValueError("Prediction requires one enabled `PredictionWriter` callback.")
    data_module_target = cfg.data.get("_target_", "phijax.data.PhiDataModule")
    log.info(f"Instantiating data module <{data_module_target}>")
    data_module = instantiate_data_module(cfg.data, "predict")
    log.info(f"Instantiating model <{cfg.model.net._target_}>")
    initialized_model = instantiate_model(cfg.model.net, trainer.precision, data_module, model_key)
    log.info(f"Instantiating objective <{cfg.model.objective._target_}>")
    objective = instantiate_objective(cfg.model.objective)
    log.info(f"Instantiating module <{cfg.model.module._target_}>")
    module = instantiate_module(
        cfg.model.module,
        initialized_model.apply,
        objective,
        name=str(cfg.application.get("name", "PINN")),
        model_summary=initialized_model.summary,
    )
    log.info(f"Instantiating loss balancer <{cfg.model.balancer.factory._target_}>")
    balancer = instantiate_balancer(cfg.model.balancer.factory, module.loss_names)
    log.info(f"Instantiating optimizer <{cfg.model.optimizer._target_}>")
    optimizer = instantiate_optimizer(cfg.model.optimizer)
    state = trainer.initialize_state(initialized_model.state, optimizer, balancer.initialize(), state_key)
    trainer.predict(
        module,
        state,
        ckpt_path=cfg.ckpt_path,
        ckpt_step=checkpoint_step,
        datamodule=data_module,
    )
    if writer.artifact_path is None:
        if trainer.strategy.is_global_zero or not writer.rank_zero_only:
            raise RuntimeError("PredictionWriter completed without producing an artifact.")
        return writer.output_path
    return writer.artifact_path


@hydra.main(version_base=None, config_path="configs", config_name="predict")
def hydra_main(cfg: DictConfig) -> None:
    """Run optional entrypoint utilities and the configured prediction task.

    Args:
        cfg: Configuration composed by Hydra from `configs/predict.yaml` and command-line overrides.
    """
    extras(cfg)
    predict(cfg)


def main() -> None:
    """Configure the repository environment and invoke the Hydra prediction entrypoint."""
    pre_hydra_routine()
    hydra_main()


if __name__ == "__main__":
    main()
