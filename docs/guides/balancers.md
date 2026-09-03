# Choose a loss balancer

A loss balancer sets the weight of each named loss from `PhiModule`. This project selects a balancer from PhiJAX and
configures it with Hydra. See the [PhiJAX balancer guide](https://hangjung97.github.io/PhiJAX/guides/balancers/)
for details about each algorithm.

## Built-in policies

The template includes three model config options:

| Config                     | PhiJAX implementation                 | Update policy                          |
| -------------------------- | ------------------------------------- | -------------------------------------- |
| `model/balancer/static`    | `phijax.balancers.StaticLossBalancer` | Fixed weights                          |
| `model/balancer/grad_norm` | `phijax.balancers.GradNormBalancer`   | Periodic gradient-norm updates         |
| `model/balancer/ntk`       | `phijax.balancers.ExactNTKBalancer`   | Periodic neural tangent kernel updates |

Select a policy in an experiment defaults list:

```yaml
defaults:
  - override /model/balancer: grad_norm
```

Or override it from the command line:

```bash
phijax-train experiment=burgers_grad_norm_1d model/balancer=static
```

## Static weights

Static balancing accepts exact objective loss names. The Burgers objective produces `initial/data` and `pde/burgers`, so
an experiment can configure them directly:

```yaml
model:
  balancer:
    _target_: phijax.balancers.StaticLossBalancer
    weights:
      initial/data: 10.0
      pde/burgers: 1.0
```

Names not present in `weights` retain the PhiJAX default weight of `1.0`.

## Set the update interval

Adaptive balancers update their weights during training:

```yaml
_target_: phijax.balancers.GradNormBalancer
update_every_n_steps: 1000
update_start_step: null
eps: 1.0e-5
moving_average_coefficient: 0.9
```

`update_every_n_steps` sets how often the weights change. With `update_start_step: null`, the first update happens
after one full interval. Use `update_start_step: 0` to update before the first optimizer step. Exact NTK balancing also
accepts `kernel_size` and `kernel_chunk_size` at the same level.

The Trainer gives each random update a key based on the training seed. It saves balancer state in the same checkpoint
as the model and optimizer.

## Adding a new balancer

The [PhiJAX balancer guide](https://hangjung97.github.io/PhiJAX/guides/balancers/) shows how to create a new balancer
with the public contract. Add a matching Hydra config under
`src/phijax_hydra_template/configs/model/balancer/` and select it in an experiment.

Test the config with a small synthetic objective before running a full experiment. PhiJAX tests cover the shared
balancer behavior and checkpoint support.
