<div align="center">

# PhiJAX Hydra Template

[![Python](https://img.shields.io/badge/Python_3.12+-blue?logo=python&logoColor=white)](https://docs.python.org/3.12/)
[![PhiJAX](https://img.shields.io/pypi/v/phijax?include_prereleases&label=PhiJAX)](https://pypi.org/project/phijax/)
[![JAX](https://img.shields.io/badge/JAX-0.11.1-8A2BE2)](https://docs.jax.dev/)
[![Flax NNX](https://img.shields.io/badge/Flax-NNX-EA4B71)](https://flax.readthedocs.io/)
[![Optax](https://img.shields.io/badge/optimization-Optax-4C8BF5)](https://optax.readthedocs.io/)
[![Hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
<br>
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/Pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
<br>
[![Code Quality](https://github.com/HangJung97/phijax-hydra-template/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/HangJung97/phijax-hydra-template/actions/workflows/code-quality-main.yaml)
[![Tests](https://github.com/HangJung97/phijax-hydra-template/actions/workflows/tests.yaml/badge.svg)](https://github.com/HangJung97/phijax-hydra-template/actions/workflows/tests.yaml)
[![Codecov](https://codecov.io/gh/HangJung97/phijax-hydra-template/graph/badge.svg)](https://codecov.io/gh/HangJung97/phijax-hydra-template)
[![License](https://img.shields.io/github/license/HangJung97/phijax-hydra-template?color=blue)](LICENSE)

</div>

This template is a ready-to-use starting point for physics-informed neural network experiments with
[PhiJAX](https://github.com/HangJung97/PhiJAX) and Hydra.

PhiJAX handles the reusable training tools. This repository contains the project-specific configs, data, evaluation,
plots, and command-line entrypoints. It includes Burgers' equation as a complete example.

The project layout and experiment-config workflow are inspired by the
[Lightning Hydra Template](https://github.com/nathanpainchaud/lightning-hydra-template).

## Main technologies

- [PhiJAX](https://github.com/HangJung97/PhiJAX) - a JAX framework for organizing and running physics-informed neural
  network training.
- [JAX](https://github.com/jax-ml/jax) - a numerical computing library with automatic differentiation, compilation,
  and accelerator support.
- [Hydra](https://github.com/facebookresearch/hydra) - a framework for composing configs and overriding settings from
  the command line.
- [`uv`](https://github.com/astral-sh/uv) - a fast tool for managing Python environments, dependencies, and lockfiles.

## Why use this template?

- Save each experiment in one config while reusing model, data, callback, and Trainer settings.
- Override settings from the command line without editing Python.
- Use PhiJAX for training, checkpointing, prediction, and evaluation instead of rebuilding framework code.
- Start with a tested Burgers example that covers data generation, training, prediction, evaluation, and plots.

## Limitations

- PhiJAX `0.2.0b4` is a beta release. Review the [PhiJAX changelog](https://hangjung97.github.io/PhiJAX/changelog/)
  before upgrading.
- PhiJAX may not yet provide every framework component required by your use case.
- Burgers is the only ready-to-run application. New physics and data formats still need application code and tests.

## Quickstart

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) first if you do not already have it.

```bash
# clone the project
git clone https://github.com/HangJung97/phijax-hydra-template.git
cd phijax-hydra-template

# create `.venv` and install the project with TensorBoard support
# [OPTIONAL] add `--extra wandb` for W&B support
uv sync --extra tensorboard

# for machines with NVIDIA GPUs, add exactly one JAX CUDA extra
# uv sync --extra cuda12 --extra tensorboard
# uv sync --extra cuda13 --extra tensorboard

# activate the virtual environment
source .venv/bin/activate

# make the GPU examples repeatable under the same hardware and software environment
export XLA_FLAGS="--xla_gpu_exclude_nondeterministic_ops --xla_gpu_autotune_level=0"
```

### Command-line example

Run the Burgers example. The default `accelerator=auto` uses JAX's default backend. JAX selects an available GPU or TPU
when its runtime is installed and the device is visible. Otherwise, it uses the CPU. PhiJAX does not configure
process-wide GPU determinism through the Trainer. The exported XLA options above select deterministic GPU kernels and
disable timing-based autotuning. See [Randomness and reproducibility](https://hangjung97.github.io/PhiJAX/guides/reproducibility/#deterministic-gpu-execution)
in the PhiJAX documentation for the limits and performance trade-offs.

```bash
phijax-train experiment=burgers_grad_norm_1d logger=tensorboard
```

Try a shorter run while developing:

```bash
phijax-train experiment=burgers_grad_norm_1d trainer.max_steps=10
```

Use a saved Burgers checkpoint to make predictions. Replace `ckpt_path` with the `checkpoints` directory from your
training run:

```bash
phijax-predict \
  experiment=burgers_grad_norm_1d \
  ckpt_path=/path/to/training/run/checkpoints \
  output_dir=artifacts/burgers
```

This saves `artifacts/burgers/burgers_1d.npz`. Evaluate the predictions with:

```bash
phijax-evaluate predictions=artifacts/burgers/burgers_1d.npz
```

The command prints the Burgers regression metrics and saves them to
`artifacts/burgers/results/metrics.json`.

### Notebook example

To use the included notebook locally, install the optional Jupyter environment and open the Burgers example:

```bash
uv sync --extra jupyter
uv run --no-sync jupyter lab notebooks/burgers_example.ipynb
```

The notebook runs the complete Burgers experiment with deterministic GPU settings, then evaluates and plots its
predictions.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HangJung97/phijax-hydra-template/blob/main/notebooks/burgers_example.ipynb)

On Colab, the first code cell clones the repository and creates an isolated `uv` environment. It requests the CUDA 12
environment by default. Select a GPU runtime before running the setup cell, or set `USE_GPU = False` for CPU execution.

## Project structure

```text
phijax-hydra-template/
├── .github/                              <- GitHub workflows and repository settings
│
├── data/                                 <- Local application data
│
├── docs/
│   ├── guides/
│   │   ├── balancers.md
│   │   ├── datasets.md
│   │   ├── hydra-composition.md
│   │   └── objectives.md
│   └── getting-started.md
│
├── logs/                                 <- Hydra runs and PhiJAX logs
│
├── notebooks/                            <- Interactive examples
│   └── burgers_example.ipynb                    <- End-to-end Burgers workflow
│
├── references/
│   └── burgers/reference.json                <- Burgers reference-data metadata
│
├── scripts/
│   └── plot_burgers_predictions.py           <- Plot saved Burgers predictions
│
├── src/phijax_hydra_template/            <- Package directory
│   ├── applications/                         <- Application-specific data, generation, and plots
│   │   └── burgers/
│   │       ├── data.py                               <- Burgers data contracts
│   │       ├── datamodule.py                         <- Custom PhiJAX DataModule
│   │       ├── generation.py                         <- Analytic and reference data generation
│   │       └── plotting.py                           <- Application plots
│   │
│   ├── configs/
│   │   ├── application/                          <- Application selection
│   │   ├── callbacks/                            <- Callback suites and individual callbacks
│   │   ├── data/                                 <- Dataset and sampler settings
│   │   ├── evaluator/                            <- Evaluation metrics
│   │   ├── experiment/                           <- Complete versioned experiments
│   │   ├── extras/                               <- Optional task behavior
│   │   ├── hparams_search/                       <- Optuna search spaces
│   │   ├── hydra/                                <- Hydra run and sweep settings
│   │   ├── local/                                <- Ignored machine-specific settings
│   │   ├── logger/                               <- Local and external loggers
│   │   ├── model/                                <- Model component configs
│   │   │   ├── balancer/                             <- Loss balancing methods
│   │   │   ├── net/                                  <- Neural network settings
│   │   │   ├── objective/                            <- Physics objectives
│   │   │   ├── optimizer/                            <- Optimizer settings
│   │   │   └── scheduler/                            <- Learning-rate schedules
│   │   ├── paths/                                <- Project and output paths
│   │   ├── trainer/                              <- PhiJAX Trainer settings
│   │   │
│   │   ├── evaluate_predictions.yaml             <- Evaluation entrypoint config
│   │   ├── predict.yaml                          <- Prediction entrypoint config
│   │   └── train.yaml                            <- Training entrypoint config
│   │
│   ├── evaluate_predictions.py               <- Hydra evaluation entrypoint
│   ├── predict.py                            <- Hydra prediction entrypoint
│   └── train.py                              <- Hydra training entrypoint
│
├── tests/                                <- Unit and integration tests
│
├── .gitignore                            <- Files and directories excluded from Git
├── .pre-commit-config.yaml               <- Repository checks
├── AGENTS.md                             <- Contributor and coding guidance
├── pyproject.toml                        <- Package metadata and dependencies
├── README.md
└── uv.lock                               <- Locked dependency versions
```

Start with [Getting started](docs/getting-started.md). To add an application, read the guides for
[Hydra configs](docs/guides/hydra-composition.md), [data](docs/guides/datasets.md), and
[objectives](docs/guides/objectives.md). See the [PhiJAX docs](https://hangjung97.github.io/PhiJAX/) for framework
concepts and API details.

## Configuration and workflow

### Main config

Location: [`configs/train.yaml`](src/phijax_hydra_template/configs/train.yaml)

This is the starting config for every training run. It selects the shared callbacks, logger, Trainer, paths, and other
defaults. Choosing an experiment fills in the application, data, and model. As usual in Hydra, later entries in the
defaults list can override earlier entries.

<details>
<summary><strong>Show main project config</strong></summary>

```yaml
# Main training composition. Later defaults override earlier values, so an experiment can replace root choices.
defaults:
  - _self_
  # Application, data, and model stay unset until an experiment or command-line override selects them.
  - optional application: null
  - optional data: null
  - optional model: null
  # Host-side lifecycle services; a callback is active when its entry is present.
  - callbacks: default
  # The local default writes versioned metrics below the Hydra output directory. Override with another logger or null.
  - logger: default
  - trainer: default
  - paths: default
  - extras: default
  - hydra: default
  # Experiment configs compose a complete runnable application and intentionally load near the end.
  - optional experiment: null
  # Search configs load after the experiment so they can select an objective and disable costly trial artifacts.
  - optional hparams_search: null
  # Ignored machine-specific overrides load last when `local/default.yaml` exists.
  - optional local: default

# Root seed for model initialization, sampling, balancer updates, and training PRNG state. Set `null` to draw and log a
# random unsigned 32-bit seed before constructing the experiment.
seed: 12345
# Names the Hydra output subtree below `logs/`; experiments normally override it.
task_name: train
# Searchable experiment labels forwarded to compatible loggers such as W&B; experiments should override this list.
tags: []
# Predict from the final in-memory state after fitting or a graceful `Ctrl+C`. The Trainer asks the DataModule for an
# optional prediction source; `SIGTERM` stops the complete run without starting prediction.
predict: false
# Metric returned to a Hydra sweeper. Keep this unset for an ordinary training run.
optimized_metric: null

# Leave `ckpt_path` unset for a fresh run. Supplying a checkpoint root resumes complete training state by default;
# set `weights_only=true` to initialize a fresh optimizer, balancer, RNG, and step from pretrained model weights.
ckpt_path: null
# Restore this exact checkpoint step; `null` selects the latest committed step below `ckpt_path`.
ckpt_step: null
weights_only: false
```

</details>

### Experiment config

Location: [`configs/experiment/`](src/phijax_hydra_template/configs/experiment/)

An experiment config collects everything needed for one reproducible run. If the existing application and config
groups fit your work, copy an experiment config and edit that one file. Here is the complete Burgers experiment:

<details>
<summary><strong>Show Burgers experiment config</strong></summary>

```yaml
# @package _global_

# A reproducible experiment overrides every group required by the root training config.
defaults:
  - override /application: burgers
  - override /data: burgers_1d
  - override /model: burgers_pinn_1d
  - override /model/objective: burgers_1d
  - override /model/balancer: grad_norm
  - override /model/scheduler: warmup_exp_decay
  - override /callbacks: burgers
  - _self_

# Stable run-family name used below `logs/` and as experiment metadata.
task_name: burgers_grad_norm_1d
# Searchable labels forwarded to experiment loggers such as W&B.
tags: [burgers, grad_norm]
# Root seed shared by data construction and all explicit JAX key splits.
seed: 42
predict: true

trainer:
  # Reference optimization length and host metric cadence.
  max_steps: 5000
  log_every_n_steps: 100
  # Prefer the best available JAX backend while using a single device from the trainer defaults.
  accelerator: auto
  # Keep parameters, forward arithmetic, derivatives, and losses in float32.
  precision: 32-true
  # Use JAX's highest dot-product precision to reduce rounding differences in the dense network.
  matmul_precision: highest
  derivative_dtype: float32
```

</details>

Values below the defaults list override settings for that experiment, such as `seed`, `tags`, `trainer.max_steps`, and
precision.

### Basic workflow

1. Add `applications/<name>/` when your application needs new Python code. Follow the Burgers example:

   - Define how the application loads or creates its data, including its contracts and pools. See the Burgers
     [data handling](src/phijax_hydra_template/applications/burgers/data.py) and
     [generation](src/phijax_hydra_template/applications/burgers/generation.py) examples.
   - Create a DataModule for sampling and batching. See
     [`datamodule.py`](src/phijax_hydra_template/applications/burgers/datamodule.py).
   - Reuse the public `PhiModule` when possible. See the
     [Burgers model config](src/phijax_hydra_template/configs/model/burgers_pinn_1d.yaml), and create a subclass only
     when different module behavior is required.
   - Add application plots and views of saved results as needed. See the
     [Burgers plotting example](src/phijax_hydra_template/applications/burgers/plotting.py).
   - Select existing PhiJAX components through Hydra configs, or create new ones by following the PhiJAX guides for
     [data](https://hangjung97.github.io/PhiJAX/guides/datasets/),
     [models](https://hangjung97.github.io/PhiJAX/guides/models/),
     [objectives](https://hangjung97.github.io/PhiJAX/guides/objectives/), and
     [balancers](https://hangjung97.github.io/PhiJAX/guides/balancers/). If a new feature would benefit other users,
     you are welcome to contribute it to PhiJAX.

2. Choose or add configs for the data, model, objective, balancer, callbacks, and Trainer.

3. Create an experiment config that combines those choices and records the settings for the run.

4. Check the final config, then run the experiment:

   ```bash
   phijax-train experiment=your_experiment --cfg job --resolve
   phijax-train experiment=your_experiment
   ```

<details>
<summary><strong>Hydra workflows and command overrides</strong></summary>

Hydra lets you change, resume, or repeat an experiment without editing Python.

Override settings for one run:

```bash
phijax-train experiment=burgers_grad_norm_1d \
  model/balancer=static \
  trainer.accelerator=cpu \
  trainer.max_steps=10
```

Resume the full training state from the latest checkpoint:

```bash
phijax-train experiment=burgers_grad_norm_1d ckpt_path=/path/to/checkpoints
```

Run the same experiment with several random seeds:

```bash
phijax-train -m experiment=burgers_grad_norm_1d seed=1,2,3
```

</details>

<details>
<summary><strong>Hyperparameter search</strong></summary>

Run the included Optuna search with the Burgers experiment:

```bash
phijax-train -m experiment=burgers_grad_norm_1d hparams_search=burgers_optuna
```

By default, the search runs 20 trials and minimizes the final `train/loss`. It searches the peak learning rate from
`1e-4` to `1e-2` on a logarithmic scale, along with weight decay, Fourier feature scale, and GradNorm smoothing. It
skips prediction and checkpoints during the search, then saves `optimization_results.yaml` in the Hydra multirun
directory. Start with a small test sweep:

```bash
phijax-train -m \
  experiment=burgers_grad_norm_1d \
  hparams_search=burgers_optuna \
  hydra.sweeper.n_trials=2 \
  trainer.max_steps=10
```

Edit `src/phijax_hydra_template/configs/hparams_search/burgers_optuna.yaml` to change the metric, ranges, sampler, or
number of trials. Copy the best settings into an experiment config for the final run. See the
[Hydra config guide](docs/guides/hydra-composition.md) for more details.

</details>

<details>
<summary><strong>Logs and experiment tracking</strong></summary>

### Logs and artifacts

Hydra creates a new output directory for every run under `logs/<task_name>/runs/<timestamp>/`. Multiruns use
`logs/<task_name>/multiruns/<timestamp>/<job_number>/`.

| Path                               | Contents                                                              |
| ---------------------------------- | --------------------------------------------------------------------- |
| `.hydra/`                          | Resolved config, Hydra config, and command-line overrides             |
| `<task_name>.log`                  | Application and lifecycle messages                                    |
| `phijax_logs/version_0/`           | Default PhiJAX hyperparameters and scalar metrics                     |
| `checkpoints/`                     | Versioned PhiJAX training state saved by Orbax                        |
| `predictions/*.npz`                | Model outputs, coordinates, targets, masks, and artifact metadata     |
| `predictions/results/metrics.json` | Evaluation metrics when evaluation uses the run's prediction artifact |

Keep machine-specific paths in an ignored `local/default.yaml`, copied from
`src/phijax_hydra_template/configs/local/example.yaml`. Do not put credentials in this file because Hydra may print and
save the full config.

### Experiment tracking

The default local logger saves `phijax_logs/version_0/hparams.yaml` and `metrics.csv` inside each Hydra run directory.
The default callbacks also log the learning rate. Turn experiment logging off with:

```bash
phijax-train experiment=burgers_grad_norm_1d logger=null '~callbacks.lr_monitor'
```

Choose another logger from the command line:

```bash
phijax-train experiment=burgers_grad_norm_1d logger=console
phijax-train experiment=burgers_grad_norm_1d logger=csv
```

TensorBoard and W&B need their optional dependencies:

```bash
uv sync --extra tensorboard
phijax-train experiment=burgers_grad_norm_1d logger=tensorboard

uv sync --extra wandb
phijax-train experiment=burgers_grad_norm_1d logger=wandb
```

The explicit CSV logger saves `metrics.csv` in the Hydra run directory. TensorBoard files go under `tensorboard/`.
W&B receives the final config, scalar metrics, and tags. Set W&B credentials with its CLI or environment variables,
not in Hydra configs.

</details>

<details>
<summary><strong>Environment management with uv</strong></summary>

[`uv`](https://docs.astral.sh/uv/) manages Python, the virtual environment, dependencies, the project package, and the
lockfile. Install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create or update `.venv` from `pyproject.toml` and `uv.lock`:

```bash
uv sync
```

Add only one CUDA extra, together with any integrations you need:

```bash
uv sync --extra cuda12 --extra wandb
```

Install the optional notebook environment with:

```bash
uv sync --extra jupyter
```

Run a command without activating the environment:

```bash
uv run phijax-train experiment=burgers_grad_norm_1d
```

Or activate the environment and use the commands directly:

```bash
source .venv/bin/activate
phijax-train experiment=burgers_grad_norm_1d
```

Commands in this README assume `.venv` is active. Otherwise, prefix them with `uv run`. Commit `uv.lock` so local
development and CI use the same dependency versions.

</details>

______________________________________________________________________

## **DELETE EVERYTHING ABOVE FOR YOUR PROJECT**

Delete the template guide above this line when you start your own project. Keep and edit the project README below.

______________________________________________________________________

<div align="center">

# Your Project Name

[![Python](https://img.shields.io/badge/Python_3.12+-blue?logo=python&logoColor=white)](https://docs.python.org/3.12/)
[![PhiJAX](https://img.shields.io/pypi/v/phijax?include_prereleases&label=PhiJAX)](https://pypi.org/project/phijax/)
[![JAX](https://img.shields.io/badge/JAX-0.11.1-8A2BE2)](https://docs.jax.dev/)
[![Flax NNX](https://img.shields.io/badge/Flax-NNX-EA4B71)](https://flax.readthedocs.io/)
[![Optax](https://img.shields.io/badge/optimization-Optax-4C8BF5)](https://optax.readthedocs.io/)
<br>
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/Pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![phijax-hydra-template](https://img.shields.io/badge/-Phi--JAX--Template-017F2F?style=flat&logo=github&labelColor=gray)](https://github.com/HangJung97/phijax-hydra-template)

</div>

<!--
## Publications

Add journal, conference, or preprint links here when your project has associated publications.
-->

## Description

Describe what the project does, what data it uses, and what results it produces.

This project uses [PhiJAX](https://github.com/HangJung97/PhiJAX) with Hydra and was created from the
[PhiJAX Hydra Template](https://github.com/HangJung97/phijax-hydra-template).

> [!IMPORTANT]
> You should have a basic understanding of [JAX](https://docs.jax.dev/),
> [PhiJAX](https://hangjung97.github.io/PhiJAX/), and [Hydra](https://hydra.cc/docs/intro/). If these tools are new to
> you, read their getting-started guides first.

## Installation

### uv (recommended)

> [!NOTE]
> [`uv`](https://docs.astral.sh/uv/) manages Python, the virtual environment, the project package, dependencies, and
> the lockfile. If you do not have `uv`, follow its
> [installation guide](https://docs.astral.sh/uv/getting-started/installation/).

1. Clone the repository:

   ```bash
   git clone https://github.com/YourGithubName/your-repo-name.git
   cd your-repo-name
   ```

2. Create `.venv` and install the project. The base environment uses the CPU. For NVIDIA, add exactly one CUDA extra:

   ```bash
   # install the CPU environment with TensorBoard support
   uv sync --extra tensorboard

   # for NVIDIA, choose one JAX CUDA runtime
   uv sync --extra cuda12 --extra tensorboard
   # or
   uv sync --extra cuda13 --extra tensorboard
   ```

   Add W&B as another optional extra:

   ```bash
   # add W&B to the CPU and TensorBoard environment
   uv sync --extra tensorboard --extra wandb
   ```

3. Activate the virtual environment:

   ```bash
   source .venv/bin/activate

   # make GPU runs repeatable under the same hardware and software environment
   export XLA_FLAGS="--xla_gpu_exclude_nondeterministic_ops --xla_gpu_autotune_level=0"
   ```

## How to run

The commands below assume `.venv` is active. Otherwise, prefix them with `uv run`.

### Training

Choose an experiment from `src/phijax_hydra_template/configs/experiment/`:

```bash
phijax-train experiment=your_experiment
```

By default, `accelerator=auto` uses JAX's default backend. JAX selects an available GPU or TPU when its runtime is
installed and the device is visible, and otherwise uses the CPU. PhiJAX does not configure process-wide GPU
determinism through the Trainer. See
[Randomness and reproducibility](https://hangjung97.github.io/PhiJAX/guides/reproducibility/#deterministic-gpu-execution)
for the XLA options used above and their trade-offs.

You can inspect the final config or override any setting from the command line:

```bash
phijax-train experiment=your_experiment --cfg job --resolve
phijax-train experiment=your_experiment trainer.max_steps=1000 data.batch_size.pde=256
```

### Hyperparameter search

Add a config under `src/phijax_hydra_template/configs/hparams_search/`, then run it with an experiment:

```bash
phijax-train -m experiment=your_experiment hparams_search=your_search
```

### Prediction

```bash
phijax-predict experiment=your_experiment ckpt_path=/path/to/checkpoints
```

### Evaluation

```bash
phijax-evaluate predictions=/path/to/predictions.npz
```

## Project structure

- `src/phijax_hydra_template/applications/`: application data, evaluation, and plots
- `src/phijax_hydra_template/configs/experiment/`: versioned experiment settings
- `tests/`: unit and integration tests
