# Connecting an application with Hydra

PhiJAX's entrypoints assemble applications from Hydra config groups. Python code defines reusable behavior; YAML
selects implementations and records experiment policy.

This guide connects the heat dataset and objective from the preceding guides into a runnable experiment.

## What the entrypoint reads

The training entrypoint, factory, and compiled-step assembly consume these paths:

| Config path                      | Runtime use                                                            |
| -------------------------------- | ---------------------------------------------------------------------- |
| `application.name`               | Human-readable module display identity                                 |
| `data`                           | Instantiated into an application-defined `PhiDataModule`               |
| `data.<application fields>`      | Passed directly to that application's DataModule                       |
| `data.batch_size`                | Application sampling sizes and deterministic prediction chunk size     |
| `model.net`                      | Instantiated with an explicit model key and trainer precision          |
| `model.objective`                | Instantiated into the objective owned by `PhiModule`                   |
| `model.module`                   | Instantiated with runtime model and objective objects                  |
| `model.balancer.factory`         | Instantiated with objective `loss_names` injected by the factory       |
| `model.balancer.update`          | Optional host schedule for NTK or gradient-norm refreshes              |
| `model.scheduler`                | Instantiated before the optimizer through recursive Hydra construction |
| `model.optimizer`                | Instantiated into an Optax gradient transformation                     |
| `trainer`, `callbacks`, `logger` | Instantiated into host-side runtime services                           |

The most important cross-group invariant is:

```text
objective term batch_key == returned training pool name == data.batch_size key
```

The application DataModule chooses how each training pool is sampled. Network `input_dim` must match every pool's
input width. Output indices used by equations must be smaller than `model.net.output_dim`.

## 1. Describe the application

Create `src/pinn_project/configs/application/heat.yaml`:

```yaml
name: heat
description: One-dimensional heat equation with homogeneous Dirichlet boundaries.
# These names document exact array order; they do not reorder arrays automatically.
coordinate_names: [t, x]
output_names: [u]
```

## 2. Select model components

Create `src/pinn_project/configs/model/heat_pinn_1d.yaml`:

```yaml
defaults:
  - module: phi_module
  - net: mlp
  - objective: heat_1d
  - balancer: static
  - scheduler: constant
  - optimizer: adamw
  - _self_

net:
  input_dim: 2
  output_dim: 1
  output_names: [u]
  hidden: [128, 128, 128, 128]
  activation: tanh
  input_norm: true
```

Each entry in this defaults list is packaged below `model`. Therefore the composed paths are
`model.module`, `model.net`, `model.objective`, `model.balancer`, `model.scheduler`, and `model.optimizer`.

The base MLP config contains required `???` values for dimensions and hidden widths. The parent model config resolves
them after composition.

## 3. Create the experiment

Create `src/pinn_project/configs/experiment/heat_static_1d.yaml`:

```yaml
# @package _global_

defaults:
  - override /application: heat
  - override /data: heat_1d
  - override /model: heat_pinn_1d
  - override /model/objective: heat_1d
  - override /model/balancer: static
  - override /model/scheduler: constant
  - _self_

task_name: heat_static_1d
seed: 42
bootstrap_only: false

trainer:
  max_steps: 20000
  log_every_n_steps: 100
  accelerator: auto
  precision: 32-true
  # Optional JAX dot/convolution policy; `null` preserves the external default.
  matmul_precision: null

model:
  balancer:
    factory:
      weights:
        initial/u: 1.0
        boundary/u: 1.0
        pde/heat: 1.0

callbacks:
  model_checkpoint:
    every_n_steps: 5000
```

`# @package _global_` makes these values override the root training configuration. Absolute defaults paths beginning
with `/` avoid resolving relative to the `experiment` group.

The experiment repeats important component selections even when the model config already has defaults. This makes the
reproducible numerical policy visible from one file and lets Hydra reject missing groups early.

## Defaults ordering

The root `train.yaml` loads:

1. its own bootstrap values;
2. optional application, data, and model groups;
3. callbacks, logger, trainer, paths, extras, and Hydra policy;
4. the selected experiment; and
5. an optional local machine override.

Because the experiment loads near the end, its global overrides turn bootstrap mode into a complete run. A private
`local/default.yaml` can still change machine-specific paths or accelerator selection without modifying the recorded
experiment.

Within one config, `_self_` controls when that file's own values are applied relative to its defaults. PhiJAX model
and experiment configs put `_self_` last so local values override the selected reusable group defaults.

## Hydra instantiation keywords

PhiJAX uses standard Hydra object construction:

| Key                   | Meaning                                                              |
| --------------------- | -------------------------------------------------------------------- |
| `_target_`            | Import path of the class or function to construct                    |
| `_partial_: true`     | Return a partial callable instead of invoking the target immediately |
| `???`                 | Mandatory value that must be supplied by composition or an override  |
| `null`                | Python `None`; often disables an optional behavior                   |
| `${seed}`             | Interpolate another config value                                     |
| `${paths.output_dir}` | Use the current Hydra run directory                                  |

Do not configure `loss_names` inside a balancer factory. The training factory injects names from the instantiated
objective so weights and diagnostics always use the same ordering.

## OmegaConf resolvers

PhiJAX entrypoints register a small set of reusable OmegaConf resolvers before Hydra composes the configuration:

| Resolver     | Purpose                                       | Example                                      |
| ------------ | --------------------------------------------- | -------------------------------------------- |
| `math`       | Read or call a public Python `math` attribute | `${math:pi}`, `${math:sqrt,4}`               |
| `op`         | Apply a public Python `operator` function     | `${op:truediv,0.01,${math:pi}}`              |
| `op.ternary` | Select between two resolved values            | `${op.ternary:${debug},small,large}`         |
| `tuple`      | Construct a tuple instead of a list           | `${tuple:1,2,3}`                             |
| `call`       | Invoke a callable by dotted import path       | `${call:os.path.basename,${paths.root_dir}}` |
| `assert`     | Raise for a false configuration condition     | `${assert:${op:gt,${trainer.max_steps},0}}`  |

For example, a Burgers objective can retain the physical expression instead of embedding a decimal approximation:

```yaml
viscosity_coefficient: ${op:truediv,0.01,${math:pi}}
```

The `call` resolver imports and executes Python code, just like Hydra's `_target_` mechanism. Treat configuration files
as trusted executable project inputs.

The `pinn-train`, `pinn-predict`, and `pinn-evaluate` entrypoints register these resolvers
automatically. Code that calls Hydra's programmatic composition API directly must register them first:

```python
from phijax.configs import register_omegaconf_resolvers

register_omegaconf_resolvers()
```

## Validate composition

Inspect the fully composed tree without starting training:

```bash
pinn-train experiment=heat_static_1d bootstrap_only=true
```

The experiment normally sets `bootstrap_only=false`; the command-line override restores validation-only behavior.
Check that the Rich config tree contains:

```text
application.name: heat
data.name: heat_1d
model.net.input_dim: 2
model.objective.terms.initial.batch_key: initial
model.objective.terms.boundary.batch_key: boundary
model.objective.terms.pde.batch_key: pde
model.balancer.name: static
```

Add a composition test:

```python
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


def test_heat_experiment_composes() -> None:
    """Verify the heat experiment connects data, model, and objective groups."""
    config_dir = Path(__file__).parents[2] / "src" / "phijax" / "configs"
    with initialize_config_dir(config_dir=str(config_dir.resolve()), version_base=None):
        config = compose(config_name="train", overrides=["experiment=heat_static_1d"])

    objective = instantiate(config.model.objective)
    data_module = instantiate(config.data)
    data_module.prepare_data()
    data_module.setup("fit")
    pools = data_module.pools

    assert config.bootstrap_only is False
    assert config.model.net.input_dim == pools["predict"].inputs.shape[1]
    assert objective.loss_names == ("initial/u", "boundary/u", "pde/heat")
    assert {term.batch_key for term in objective.terms.values()} <= set(pools)
    assert {term.batch_key for term in objective.terms.values()} <= set(config.data.batch_size)
    assert "predict" in pools
    assert int(config.data.batch_size.predict) > 0
```

## Run and override the experiment

Train with the recorded defaults:

```bash
pinn-train experiment=heat_static_1d
```

Use command-line overrides for temporary comparisons:

```bash
# Change the PDE coefficient in both generated reference targets and the residual.
pinn-train experiment=heat_static_1d \
  data.diffusivity=0.05 \
  model.objective.terms.pde.residual_fn.diffusivity=0.05

# Replace fixed weights with gradient-norm balancing.
pinn-train experiment=heat_static_1d \
  model/balancer=grad_norm

# Train and immediately predict from the final in-memory state.
pinn-train experiment=heat_static_1d predict=true
```

When one physical parameter appears in several groups, consider defining it once at the experiment root and
interpolating it from both data and objective configs. This prevents a reference generator and residual function from
silently using different values.

## Standalone prediction

Prediction must compose the same application, data, network, objective, optimizer, and balancer structure used to
create the full-state Orbax checkpoint:

```bash
pinn-predict experiment=heat_static_1d \
  ckpt_path=/path/to/training/run/checkpoints \
  output_dir=/path/to/predictions \
  save_file_name=heat_solution
```

PhiJAX constructs this compatible restore template without building a training batch source, compiled optimizer step,
or adaptive-balancer diagnostic batches. Changing parameter-tree, optimizer-state, or balancer-state structure makes
the checkpoint template incompatible. Changing only prediction chunk size is safe:

```bash
pinn-predict experiment=heat_static_1d \
  ckpt_path=/path/to/checkpoints \
  data.batch_size.predict=1024
```

An optional MATLAB sidecar can be enabled without changing the checkpoint or prediction batches:

```bash
pinn-predict experiment=heat_static_1d \
  ckpt_path=/path/to/checkpoints \
  callbacks.prediction_writer.save_mat=true
```

The `.mat` file is written below `output_dir` with the same suffix-free `save_file_name`. Applications may define
`application.mat_field_names` to translate generic keys such as `prediction`, named output channels, or nested metadata
fields into downstream MATLAB conventions. The prediction writer applies application-provided `output_scales` to both
NPZ and MATLAB predictions and references, and both formats retain the applied scales. MATLAB groups structural and
application metadata inside a `metadata` struct.

Saving is performed by the standard prediction writer callback:

```yaml
callbacks:
  prediction_writer:
    save_mat: true
```

The default fitting suite contains the same writer under `callbacks.prediction_writer`, enabled only when
`predict=true`. After fitting, the existing Trainer asks the DataModule to set up its prediction stage and reuses the
final in-memory model state. Returning `None` from `predict_batch_source()` skips prediction. A graceful `Ctrl+C` stops
fitting at the last completed state and continues into prediction; `SIGTERM` terminates the complete run.

## Common composition failures

| Symptom                            | Likely cause                                                       |
| ---------------------------------- | ------------------------------------------------------------------ |
| Missing mandatory value            | A required MLP field or prediction checkpoint remains `???`        |
| Missing pool or batch-size key     | A term's `batch_key` is absent from pools or `data.batch_size`     |
| Residual-group count error         | Equation outer groups do not align with inferred or explicit names |
| Unknown Hydra override             | Wrong config-group path or a missing `+` for a new field           |
| Checkpoint restore structure error | Prediction composed a different model architecture                 |
| Repeated JAX compilation           | Batch shapes or PyTree structure change between iterations         |

Prefer adding a focused Hydra composition test whenever a new config group or experiment is introduced.
