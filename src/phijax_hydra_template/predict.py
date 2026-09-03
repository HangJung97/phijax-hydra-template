from pathlib import Path

import hydra
import jax
from omegaconf import DictConfig, OmegaConf
from phijax import PhiModule
from phijax.integrations.hydra import (
    build_trainer,
    instantiate_balancer,
    instantiate_data_module,
    instantiate_model_factory,
    instantiate_module,
    instantiate_objective,
    instantiate_optimizer,
)
from phijax.utils import RankedLogger, extras, pre_hydra_routine, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def predict(cfg: DictConfig) -> Path:
    """Restore a composed experiment, run prediction, and save a dense artifact.

    Args:
        cfg: Hydra prediction configuration containing an experiment, checkpoint root, and artifact destination.

    Returns:
        Saved prediction artifact path.

    Raises:
        ValueError: If an executable prediction run has no checkpoint root.
    """
    ckpt_path = None if OmegaConf.is_missing(cfg, "ckpt_path") else cfg.get("ckpt_path")
    if ckpt_path is None or not str(ckpt_path).strip():
        raise ValueError(
            "Prediction requires `ckpt_path` to identify a trained Orbax checkpoint root, for example "
            "`phijax-predict experiment=<name> ckpt_path=/path/to/checkpoints`.",
        )
    checkpoint_step = cfg.get("ckpt_step")
    model_key, runtime_key, sampling_key, balancer_key = jax.random.split(jax.random.key(int(cfg.seed)), 4)
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer = build_trainer(cfg)
    writer = trainer.prediction_writer
    if writer is None:
        raise ValueError("Prediction requires one configured `PredictionWriter` callback.")
    data_module_target = cfg.data.get("_target_", "phijax.DataModule")
    log.info(f"Instantiating data module <{data_module_target}>")
    data_module = instantiate_data_module(cfg.data)
    log.info(f"Instantiating model factory <{cfg.model.net._target_}>")
    model_factory = instantiate_model_factory(cfg.model.net)
    log.info(f"Instantiating objective <{cfg.model.objective._target_}>")
    objective = instantiate_objective(cfg.model.objective)
    log.info(f"Instantiating module <{cfg.model.module._target_}>")
    module = instantiate_module(
        cfg.model.module,
        model_factory,
        objective,
        name=str(cfg.application.get("name", "PINN")),
    )
    if not isinstance(module, PhiModule):
        raise TypeError("The standard prediction entrypoint requires `phijax.PhiModule` or a subclass.")
    log.info(f"Instantiating loss balancer <{cfg.model.balancer._target_}>")
    balancer = instantiate_balancer(cfg.model.balancer, module.loss_names)
    log.info(f"Instantiating optimizer <{cfg.model.optimizer._target_}>")
    optimizer = instantiate_optimizer(cfg.model.optimizer)
    # Recreate the training-time model closure and state structure before loading checkpoint weights.
    data_module.prepare_stage("fit")
    try:
        statistics = data_module.input_statistics()
        input_mean, input_std = (None, None) if statistics is None else statistics
        with jax.default_device(trainer.strategy.root_device):
            bound_module, model_state = module.prepare_model(
                key=model_key,
                input_mean=input_mean,
                input_std=input_std,
                precision=trainer.precision,
            )
            state = trainer.initialize_state(
                model_state,
                optimizer,
                balancer.initialize(),
                runtime_key,
                sampling_key=sampling_key,
                balancer_key=balancer_key,
            )
    finally:
        data_module.teardown_stage("fit")
    trainer.predict_state(
        bound_module,
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
