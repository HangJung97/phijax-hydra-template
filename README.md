# PhiJAX Hydra Template

A configuration-first starting point for reproducible physics-informed neural network projects built with
[PhiJAX](https://github.com/HangJung97/PhiJAX), Hydra, and OmegaConf.

This repository owns project concerns: experiments, application DataModules, plotting, evaluation, local paths, and
command-line entrypoints. PhiJAX supplies the reusable Trainer, PhiModule, callbacks, equations, objectives, models,
balancers, data contracts, and persistence APIs.

## Use this template

Create a repository from the GitHub template, then rename the neutral `pinn_project` package and the three console
entrypoints in `pyproject.toml`. Search the repository for `pinn_project` so imports, Hydra targets, and tests move
together.

Install the CPU environment:

```bash
uv sync
```

For NVIDIA systems, select exactly one JAX runtime:

```bash
uv sync --extra cuda12
# or
uv sync --extra cuda13
```

Run the included Burgers experiment:

```bash
pinn-train experiment=burgers_grad_norm_1d
```

Shorten it while developing:

```bash
pinn-train experiment=burgers_grad_norm_1d trainer.max_steps=10 trainer.accelerator=cpu
```

Hydra writes runs below `logs/`. Machine-specific paths belong in an ignored `local/default.yaml`, copied from
`src/pinn_project/configs/local/example.yaml`; never store credentials there because composed configs are printed and
saved.

## Project structure

- `src/pinn_project/configs/`: root configs and composable application, data, model, callback, logger, and trainer groups
- `src/pinn_project/applications/`: application-owned DataModules, generation, plotting, and evaluation code
- `src/pinn_project/{train,predict,evaluate_predictions}.py`: thin Hydra entrypoints
- `tests/`: config composition, application, and shortened workflow tests
- `scripts/`: user-facing plotting and utility commands

Start with [Hydra composition](docs/guides/hydra-composition.md), [datasets](docs/guides/datasets.md), and
[objectives](docs/guides/objectives.md) when adding an application.
