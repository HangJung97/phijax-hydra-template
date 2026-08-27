import hydra
import jax
from omegaconf import DictConfig, OmegaConf
from phijax.integrations.hydra import (
    configure_training,
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
from phijax.training import FitResult
from phijax.utils import RankedLogger, extras, pre_hydra_routine, resolve_seed, seed_everything, task_wrapper

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


def describe_bootstrap(cfg: DictConfig) -> str:
    """Render the implementation-status banner for a bootstrap training run.

    Args:
        cfg: Composed Hydra training configuration.

    Returns:
        Human-readable status banner.
    """
    application = cfg.get("application")
    application_name = application.get("name") if application is not None else None
    selection = f" for the `{application_name}` application" if application_name else ""
    return f"PhiJAX configuration{selection} is valid. Select a runnable experiment to start training."


@task_wrapper
def train(cfg: DictConfig) -> FitResult | None:
    """Run a composed training experiment or an explicit bootstrap-only validation.

    Args:
        cfg: Configuration composed by Hydra from `configs/train.yaml` and command-line overrides.

    Returns:
        Final fit result, or `None` when `bootstrap_only` is enabled.
    """
    seed = _resolve_training_seed(cfg)
    log.info(f"Global seed set to {seed}.")
    key = seed_everything(seed)
    if bool(cfg.get("bootstrap_only", True)):
        print(describe_bootstrap(cfg))
        return None
    model_key, sampling_key, balancer_key, training_key = jax.random.split(key, 4)
    log.info("Instantiating callbacks...")
    callbacks = instantiate_callbacks(cfg.get("callbacks"))
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer = instantiate_trainer(cfg.trainer, callbacks)
    log.info("Instantiating loggers...")
    trainer.logger = instantiate_loggers(cfg.get("logger"), trainer)
    trainer.print_environment_info()
    if bool(cfg.get("predict", False)) and trainer.prediction_writer is None:
        raise ValueError("`predict=true` requires an enabled `PredictionWriter` callback.")
    checkpoint_step = cfg.get("ckpt_step")
    checkpoint_path = cfg.get("ckpt_path")
    if checkpoint_path:
        restore_kind = "model weights" if bool(cfg.get("weights_only", False)) else "complete training state"
        step_description = "latest step" if checkpoint_step is None else f"step {checkpoint_step}"
        log.info(f"Restoring {restore_kind} from <{checkpoint_path}> at {step_description}.")
    else:
        log.info("Using freshly initialized training state.")
    hyperparameters = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(hyperparameters, dict):
        raise TypeError("The resolved training configuration must be a mapping.")
    if trainer.logger.loggers:
        log.info("Logging hyperparameters!")

    data_module_target = cfg.data.get("_target_", "phijax.data.PhiDataModule")
    log.info(f"Instantiating data module <{data_module_target}>")
    data_module = instantiate_data_module(cfg.data, "fit")
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
    state = trainer.initialize_state(initialized_model.state, optimizer, balancer.initialize(), training_key)
    training = configure_training(
        cfg.model,
        trainer,
        module,
        balancer,
        optimizer,
    )
    log.info("Starting training!")
    result = trainer.fit(
        module,
        training,
        state,
        hyperparameters=hyperparameters,
        ckpt_path=checkpoint_path,
        ckpt_step=checkpoint_step,
        weights_only=bool(cfg.get("weights_only", False)),
        datamodule=data_module,
        sampling_key=sampling_key,
        balancer_key=balancer_key,
    )
    if bool(cfg.get("predict", False)):
        log.info("Starting prediction from the final in-memory training state.")
        predictions = trainer.predict(module, result.state, datamodule=data_module)
        if predictions is None:
            log.info("Prediction skipped because the DataModule did not provide prediction data.")
        elif trainer.prediction_writer is not None and trainer.prediction_writer.artifact_path is not None:
            log.info(f"Predictions saved to <{trainer.prediction_writer.artifact_path}>.")
    return result


@hydra.main(version_base=None, config_path="configs", config_name="train")
def hydra_main(cfg: DictConfig) -> None:
    """Run optional entrypoint utilities and the configured training task.

    Args:
        cfg: Configuration composed by Hydra from `configs/train.yaml` and command-line overrides.
    """
    # Resolve a random seed before printing the configuration so the concrete value is retained in the run record.
    _resolve_training_seed(cfg)
    extras(cfg)
    train(cfg)


def main() -> None:
    """Configure the repository environment and invoke the Hydra training entrypoint."""
    pre_hydra_routine()
    hydra_main()


if __name__ == "__main__":
    main()
