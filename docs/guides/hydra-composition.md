# Compose configs with Hydra

Hydra combines small YAML files into one config. Python defines application behavior. YAML selects the experiment and
run settings. The three commands then build the selected PhiJAX objects.

## Configuration groups

| Group or field                    | Responsibility                                                   |
| --------------------------------- | ---------------------------------------------------------------- |
| `application`                     | Coordinate, output, and application metadata                     |
| `data`                            | Application DataModule and sampling policy                       |
| `model.net`                       | Public PhiJAX model factory and architecture                     |
| `model.module`                    | Public `PhiModule` target                                        |
| `model.objective`                 | Objective terms, equations, and batch names                      |
| `model.balancer`                  | Loss balancer and update interval                                |
| `model.scheduler` and `optimizer` | Optax schedule and optimizer                                     |
| `callbacks`                       | Active checkpoint, progress, monitoring, and prediction services |
| `logger`                          | Local CSV by default, or console, TensorBoard, W&B, or `null`    |
| `trainer`                         | Devices, precision, step count, and logging interval             |
| `paths` and `hydra`               | Artifact paths and run directories                               |
| `experiment`                      | A complete set of choices for one experiment                     |
| `hparams_search`                  | Optional sweeper, objective metric, and search ranges            |
| `local`                           | Ignored machine-specific overrides loaded last when present      |

## Complete experiment selection

`burgers_grad_norm_1d.yaml` selects every group needed for a runnable experiment:

```yaml
# @package _global_
defaults:
  - override /application: burgers
  - override /data: burgers_1d
  - override /model: burgers_pinn_1d
  - override /model/objective: burgers_1d
  - override /model/balancer: grad_norm
  - override /model/scheduler: warmup_exp_decay
  - override /callbacks: burgers
  - _self_

task_name: burgers_grad_norm_1d
tags: [burgers, grad_norm]
seed: 42
```

Run it with:

```bash
phijax-train experiment=burgers_grad_norm_1d
```

Use command-line overrides to change a setting without editing Python:

```bash
phijax-train experiment=burgers_grad_norm_1d \
  data=burgers_analytic_1d \
  model/balancer=static \
  trainer.accelerator=cpu \
  trainer.max_steps=10
```

## Defaults order

Hydra applies later defaults after earlier ones. The root config loads the application, data, and model first. It then
loads shared services, the experiment, and optional local settings. Within one file, `_self_` sets when that file's
values are applied.

Keep `_self_` in root, model, callback-suite, and experiment configs. Use `override /group: option` inside an
experiment so readers can see which option it selects.

## Public targets

Use public package exports in Hydra targets:

```yaml
trainer:
  _target_: phijax.Trainer

model:
  module:
    _target_: phijax.PhiModule
  net:
    _target_: phijax.models.build_mlp
  objective:
    _target_: phijax.objectives.CompositeObjective.from_equations
```

Callbacks, balancers, equations, and evaluators follow the same pattern through `phijax.callbacks`,
`phijax.balancers`, `phijax.equations`, and `phijax.evaluation`.

The template provides AdamW, Adam, SGD, and SOAP optimizer configs. AdamW is the model default. Select another optimizer
without changing Python:

```bash
phijax-train experiment=burgers_grad_norm_1d model/optimizer=adam
phijax-train experiment=burgers_grad_norm_1d model/optimizer=sgd
phijax-train experiment=burgers_grad_norm_1d model/optimizer=soap
```

SOAP uses the pinned [SOAP_JAX fork](https://github.com/HangJung97/SOAP_JAX) installed by `uv sync`. Its learning
rate follows `model.scheduler`; preconditioners refresh every 10 steps by default.

The training entrypoint passes the model factory, objective, DataModule, optimizer, and balancer to `Trainer.fit()`.
PhiJAX then initializes the model and training state, finds the required batch keys, and manages DataModule setup.

Callback and logger entries are active when present. They do not use an `enabled` field. Omit an entry in a callback
suite or delete it for one run:

```bash
phijax-train experiment=burgers_grad_norm_1d '~callbacks.model_checkpoint'
```

The default logger writes to `phijax_logs/version_0/` inside the Hydra output directory. Use `logger=null` to disable
it, and remove `callbacks.lr_monitor` in the same run because that callback requires a logger.

## Resolvers

Import the project config package before composing configs in Python. Its initializer registers PhiJAX's OmegaConf
resolvers:

```python
from phijax_hydra_template.configs import register_omegaconf_resolvers

register_omegaconf_resolvers()
```

The template uses resolvers for math, choices, and Hydra run paths. Treat configs as trusted input because `_target_`
and callable resolvers can import Python objects.

## Inspect the composed config

Use Hydra's built-in flags to resolve and print the job config without running training:

```bash
phijax-train experiment=burgers_grad_norm_1d --cfg job --resolve
```

Add a programmatic test for every new experiment:

```python
from pathlib import Path

from hydra import compose, initialize_config_dir


def test_experiment_composes() -> None:
    """Verify the experiment selects a complete runtime policy."""
    config_dir = Path(__file__).parents[2] / "src" / "phijax_hydra_template" / "configs"
    with initialize_config_dir(config_dir=str(config_dir.resolve()), version_base=None):
        config = compose(config_name="train", overrides=["experiment=burgers_grad_norm_1d"])

    assert config.task_name == "burgers_grad_norm_1d"
    assert config.data._target_ == "phijax_hydra_template.applications.burgers.BurgersDataModule"
    assert config.model.module._target_ == "phijax.PhiModule"
```

## Hyperparameter search

The included Optuna config searches scalar settings in the Burgers experiment:

```bash
phijax-train -m experiment=burgers_grad_norm_1d hparams_search=burgers_optuna
```

`optimized_metric` names the final PhiJAX metric returned to Optuna. The example searches the scheduler's peak learning
rate from `1e-4` to `1e-2` on a logarithmic scale. It minimizes `train/loss`, runs trials one at a time, and omits
experiment logging, prediction, and checkpoint writing during the search. Its search ranges live in
`src/phijax_hydra_template/configs/hparams_search/burgers_optuna.yaml`.

Use overrides for a short test of the workflow:

```bash
phijax-train -m \
  experiment=burgers_grad_norm_1d \
  hparams_search=burgers_optuna \
  hydra.sweeper.n_trials=2 \
  trainer.max_steps=10
```

Optuna writes `optimization_results.yaml` in the timestamped Hydra multirun directory. Copy the best parameter values
into an experiment config before running the complete training and prediction workflow.

## Prediction and evaluation

Standalone prediction composes the same experiment used for training and restores a PhiJAX checkpoint:

```bash
phijax-predict experiment=burgers_grad_norm_1d ckpt_path=/path/to/checkpoints
```

Evaluate the resulting canonical artifact without constructing JAX training components:

```bash
phijax-evaluate predictions=/path/to/burgers_1d.npz
```

Machine-specific data roots and accelerator choices belong in ignored
`src/phijax_hydra_template/configs/local/default.yaml`, copied from `local/example.yaml`. Never put credentials in a
Hydra config because resolved configuration may be printed and saved.
