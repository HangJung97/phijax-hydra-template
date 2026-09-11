# Getting started

This page trains the included Burgers example, then evaluates and plots its predictions.

## Install

Clone the repository and install the locked environment:

```bash
uv sync --locked
source .venv/bin/activate
```

The project requires Python 3.12 or newer and uses `phijax==0.2.0b5`.

For an NVIDIA GPU, install exactly one JAX CUDA runtime before activating the environment:

```bash
uv sync --locked --extra cuda12
# or
uv sync --locked --extra cuda13

source .venv/bin/activate
```

TPU support is experimental. On a TPU machine, install the JAX TPU runtime instead of a CUDA runtime:

```bash
uv sync --locked --extra tpu
source .venv/bin/activate
```

Install at most one of `cuda12`, `cuda13`, and `tpu` in an environment.

## Run the notebook

Install the optional Jupyter environment and open the guided Burgers example from the project root:

```bash
uv sync --locked --extra jupyter
uv run --no-sync jupyter lab notebooks/burgers_example.ipynb
```

The notebook uses the same Hydra entrypoints and 5,000-step Burgers experiment as the command-line workflow. It enables
deterministic XLA options, saves the final checkpoint, and writes each run under `logs/`.

You can also open it directly in Google Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HangJung97/phijax-hydra-template/blob/main/notebooks/burgers_example.ipynb)

The first code cell detects Colab, clones the repository, installs `uv`, and creates an isolated project environment.
It requests the CUDA 12 environment by default. Select a GPU runtime before running the setup cell, or set
`USE_GPU = False` for CPU execution. Colab storage is temporary, so download results or copy them to Google Drive
before the [runtime ends](https://research.google.com/colaboratory/faq.html#resource-limits).

## Control the JAX backend

The Trainer uses `accelerator=auto` by default. This follows JAX's default backend selection: an available GPU or TPU
when its runtime is installed and the device is visible, and the CPU otherwise.

Set `JAX_PLATFORMS` before the command to force JAX to initialize only the CPU backend:

```bash
JAX_PLATFORMS=cpu phijax-train experiment=burgers_grad_norm_1d
```

The test suite also forces CPU initialization. The environment variable can be made explicit when running tests:

```bash
JAX_PLATFORMS=cpu uv run --no-sync pytest
```

### Deterministic GPU execution

A seed makes model initialization and batch sampling repeatable, but it does not control XLA's process-wide GPU
kernels or compile-time autotuning. Set both options before starting PhiJAX when comparing GPU runs:

```bash
XLA_FLAGS="--xla_gpu_exclude_nondeterministic_ops --xla_gpu_autotune_level=0" \
  phijax-train experiment=burgers_grad_norm_1d
```

PhiJAX 0.2.0b4 removes the old inert `Trainer.deterministic` option. See PhiJAX's
[Randomness and reproducibility](https://hangjung97.github.io/PhiJAX/guides/reproducibility/#deterministic-gpu-execution)
guide for seed requirements, limitations, and performance trade-offs.

### GPU memory allocation errors

JAX normally reserves most GPU memory when it initializes. On a shared or display GPU, disabling preallocation may
help with startup allocation errors:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false phijax-train experiment=burgers_grad_norm_1d
```

Keep JAX's default preallocation for ordinary runs. Use this override only when allocation fails, and keep the setting
fixed when comparing experiments because it can affect GPU autotuning. See PhiJAX's
[troubleshooting guide](https://hangjung97.github.io/PhiJAX/guides/troubleshooting/#cuda-runs-out-of-memory-during-startup)
for alternatives. Environment variables that control JAX must be set before JAX initializes.

## Check the configuration

Compose and print the experiment without training:

```bash
phijax-train experiment=burgers_grad_norm_1d --cfg job --resolve
```

Use this command after changing a config group.

## Run a short training job

Run ten steps on CPU:

```bash
phijax-train experiment=burgers_grad_norm_1d \
  data=burgers_analytic_1d \
  trainer.accelerator=cpu \
  trainer.max_steps=10
```

The first run may pause while JAX compiles the training step. Hydra writes run files under `logs/`. PhiJAX writes
hyperparameters and scalar metrics below the run's `phijax_logs/version_0/` directory.

## Run the full reference experiment

```bash
XLA_FLAGS="--xla_gpu_exclude_nondeterministic_ops --xla_gpu_autotune_level=0" \
  phijax-train experiment=burgers_grad_norm_1d
```

The DataModule creates the Burgers reference file when it is missing. The experiment uses the settings in
`configs/experiment/burgers_grad_norm_1d.yaml`.

## Predict from a checkpoint

```bash
phijax-predict \
  experiment=burgers_grad_norm_1d \
  ckpt_path=/path/to/checkpoints
```

This writes a PhiJAX `.npz` prediction artifact.

## Evaluate predictions

```bash
phijax-evaluate predictions=/path/to/burgers_1d.npz
```

The evaluator prints regression metrics and writes `metrics.json`.

## Plot predictions

Use the included script to plot the reference solution, prediction, and absolute error:

```bash
python scripts/plot_burgers_predictions.py \
  /path/to/burgers_1d.npz \
  --output /path/to/burgers_1d.png
```

If `--output` is omitted, the script saves a `.png` beside the prediction artifact. Add `--show` to display the figure
after saving it. The command also prints the relative L2 error.

## Next steps

- [Hydra composition](guides/hydra-composition.md) explains how config groups form an experiment.
- [Application data](guides/datasets.md) explains pools, sampling, and prediction chunks.
- [Equations and objectives](guides/objectives.md) explains the Burgers loss terms.
- [Loss balancers](guides/balancers.md) explains fixed and adaptive loss weights.
- [PhiJAX documentation](https://hangjung97.github.io/PhiJAX/) covers the framework API and core concepts.
