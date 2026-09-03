# Configure equations and objectives

PhiJAX separates physical residual calculations from loss reduction:

- An equation computes one or more named residual arrays from a model, explicit model state, and one batch.
- `CompositeObjective.from_equations()` turns named equations into routed loss terms.
- Equation metadata supplies default residual names and the NTK input stream.

This project configures these public APIs. The
[PhiJAX objective guide](https://hangjung97.github.io/PhiJAX/guides/objectives/) shows how to create new equations and
objectives.

## Burgers objective

The reference application declares its complete objective in
`src/phijax_hydra_template/configs/model/objective/burgers_1d.yaml`.

The complete objective uses the simplified equation mapping:

```yaml
_target_: phijax.objectives.CompositeObjective.from_equations
equations:
  initial:
    _target_: phijax.equations.base_data_fidelity
    _partial_: true
    output_indices: [0]
    target_indices: [0]
  pde:
    _target_: phijax.equations.burgers_1d
    _partial_: true
    viscosity_coefficient: ${op:truediv,0.01,${math:pi}}
    output_index: 0
```

Each mapping key is also the batch key. `initial` compares output `u` with the target column. `pde` applies PhiJAX's
public one-dimensional Burgers equation.

The application contract is explicit:

| Property         | Burgers value                       |
| ---------------- | ----------------------------------- |
| Coordinates      | `[t, x]`                            |
| Model outputs    | `[u]`                               |
| Initial batch    | `inputs: [N, 2]`, `targets: [N, 1]` |
| PDE batch        | `inputs: [N, 2]`, `targets: [N, 0]` |
| Objective losses | `initial/data`, `pde/burgers`       |

These names are also the keys used by loss balancers and training metrics.

## Choose the NTK input

NTK balancers can use physical residuals or selected model outputs. Equation metadata selects the stream when
`CompositeObjective.from_equations()` builds the terms. Data-fidelity equations use model outputs. PDE equations use
physical residuals.

Keep batch keys stable. `Trainer.fit()` reads the required names from the objective. The application DataModule must
provide a sampler for each name.

## Adding application physics

Create project-specific equations with the public contracts shown in the
[PhiJAX objective guide](https://hangjung97.github.io/PhiJAX/guides/objectives/). Equations can depend on the
application's coordinate convention, measured fields, units, or artifact contract.

A local physics function must work with JAX transformations. Pass model state explicitly and use fixed-shape JAX
arrays. Do not log, read files, convert to NumPy, read configs, or change Python state inside transformed code.

Export local physics from the application package and reference it with a Hydra `_target_`. Keep coordinate and output
indices next to the config that selects the function.

## Testing

Test objective configuration at two levels:

1. Compose the experiment and assert the exact term types, batch keys, and loss names.
2. Run a tiny CPU-only compiled update with synthetic data and assert finite named losses.

The Burgers tests show both levels. PhiJAX tests cover shared equations, objective reduction, derivatives, and NTK
inputs.
