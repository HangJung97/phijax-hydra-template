import hydra
from omegaconf import DictConfig
from phijax import FitResult, PhiModule
from phijax.integrations.hydra import (
    build_trainer,
    instantiate_balancer,
    instantiate_data_module,
    instantiate_model_factory,
    instantiate_module,
    instantiate_objective,
    instantiate_optimizer,
    to_hyperparameters,
)
from phijax.utils import RankedLogger, extras, pre_hydra_routine, resolve_seed, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)


def _resolve_training_seed(cfg: DictConfig) -> int:
    """Replace an optional configured seed with one concrete runtime seed.

    Args:
        cfg: Mutable composed training configuration containing `seed`.

    Returns:
        Concrete unsigned 32-bit seed stored back into `cfg.seed`.
    """
    seed = resolve_seed(cfg.get("seed"))
    cfg.seed = seed
    return seed


def _get_optimized_metric(cfg: DictConfig, result: FitResult) -> float | None:
    """Read the configured optimization objective from a completed fit.

    Args:
        cfg: Composed training configuration containing the optional `optimized_metric` name.
        result: Final PhiJAX fit result containing logged scalar metrics.

    Returns:
        Selected metric as a Python float, or `None` when no optimization metric is configured.

    Raises:
        KeyError: If the configured metric is absent from the final fit result.
    """
    metric_name = cfg.get("optimized_metric")
    if metric_name is None:
        return None
    metric_key = str(metric_name)
    if metric_key not in result.metrics:
        available_metrics = ", ".join(sorted(result.metrics)) or "none"
        raise KeyError(f"Optimized metric <{metric_key}> was not logged. Available metrics: {available_metrics}.")
    return float(result.metrics[metric_key])


@task_wrapper
def train(cfg: DictConfig) -> FitResult:
    """Run a composed training experiment.

    Args:
        cfg: Configuration composed by Hydra from `configs/train.yaml` and command-line overrides.

    Returns:
        Final fit result.
    """
    seed = _resolve_training_seed(cfg)
    log.info(f"Trainer seed set to {seed}.")
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer = build_trainer(cfg)
    if bool(cfg.get("predict", False)) and trainer.prediction_writer is None:
        raise ValueError("`predict=true` requires a configured `PredictionWriter` callback.")
    checkpoint_step = cfg.get("ckpt_step")
    checkpoint_path = cfg.get("ckpt_path")
    if checkpoint_path:
        restore_kind = "model weights" if bool(cfg.get("weights_only", False)) else "complete training state"
        step_description = "latest step" if checkpoint_step is None else f"step {checkpoint_step}"
        log.info(f"Restoring {restore_kind} from <{checkpoint_path}> at {step_description}.")
    else:
        log.info("Using freshly initialized training state.")
    hyperparameters = to_hyperparameters(cfg)
    if trainer.logger.loggers:
        log.info("Logging hyperparameters!")

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
        raise TypeError("The standard training entrypoint requires `phijax.PhiModule` or a subclass.")
    log.info(f"Instantiating loss balancer <{cfg.model.balancer._target_}>")
    balancer = instantiate_balancer(cfg.model.balancer, module.loss_names)
    log.info(f"Instantiating optimizer <{cfg.model.optimizer._target_}>")
    optimizer = instantiate_optimizer(cfg.model.optimizer)
    log.info("Starting training!")
    result = trainer.fit(
        module,
        datamodule=data_module,
        optimizer=optimizer,
        seed=seed,
        balancer=balancer,
        hyperparameters=hyperparameters,
        ckpt_path=checkpoint_path,
        ckpt_step=checkpoint_step,
        weights_only=bool(cfg.get("weights_only", False)),
    )
    if bool(cfg.get("predict", False)):
        log.info("Starting prediction from the final in-memory training state.")
        predictions = trainer.predict(result, datamodule=data_module)
        if predictions is None:
            log.info("Prediction skipped because the DataModule did not provide prediction data.")
        elif trainer.prediction_writer is not None and trainer.prediction_writer.artifact_path is not None:
            log.info(f"Predictions saved to <{trainer.prediction_writer.artifact_path}>.")
    return result


@hydra.main(version_base=None, config_path="configs", config_name="train")
def hydra_main(cfg: DictConfig) -> float | None:
    """Run optional entrypoint utilities and the configured training task.

    Args:
        cfg: Configuration composed by Hydra from `configs/train.yaml` and command-line overrides.

    Returns:
        Configured optimization metric for a Hydra sweeper, or `None` for an ordinary run.
    """
    # Resolve a random seed before printing the configuration so the concrete value is retained in the run record.
    _resolve_training_seed(cfg)
    extras(cfg)
    result = train(cfg)
    return _get_optimized_metric(cfg, result)


def main() -> None:
    """Configure the repository environment and invoke the Hydra training entrypoint."""
    pre_hydra_routine()
    hydra_main()


if __name__ == "__main__":
    main()
