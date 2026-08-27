import json

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from phijax.evaluation import EvaluationResult, PredictionEvaluator, json_ready
from phijax.utils import extras, pre_hydra_routine, task_wrapper


@task_wrapper
def evaluate_predictions(cfg: DictConfig) -> EvaluationResult:
    """Instantiate and run the evaluator selected by Hydra.

    Args:
        cfg: Hydra configuration containing an `evaluator` object specification.

    Returns:
        Completed evaluation result.

    Raises:
        TypeError: If the configured object does not implement :class:`phijax.evaluation.PredictionEvaluator`.
    """
    evaluator = instantiate(cfg.evaluator)
    if not isinstance(evaluator, PredictionEvaluator):
        raise TypeError("Configured `evaluator` must provide an `evaluate()` method returning `EvaluationResult`.")
    result = evaluator.evaluate()
    print(json.dumps(json_ready(result.metrics), indent=2, sort_keys=True))
    return result


@hydra.main(version_base=None, config_path="configs", config_name="evaluate_predictions")
def hydra_main(cfg: DictConfig) -> None:
    """Run optional entrypoint utilities and the configured evaluation task.

    Args:
        cfg: Configuration composed by Hydra from `configs/evaluate_predictions.yaml` and command-line overrides.
    """
    extras(cfg)
    evaluate_predictions(cfg)


def main() -> None:
    """Configure the repository environment and invoke the Hydra evaluation entrypoint."""
    pre_hydra_routine()
    hydra_main()


if __name__ == "__main__":
    main()
