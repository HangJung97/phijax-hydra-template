# AGENTS.md

These instructions apply to the entire generated project.

This repository is a Hydra project built on PhiJAX. Keep project-specific applications, DataModules, experiments,
plots, and evaluation here. Contribute reusable trainers, callbacks, equations, objectives, models, balancers, and
data contracts to PhiJAX instead of duplicating framework code.

## Architecture

- `src/pinn_project/{train,predict,evaluate_predictions}.py` are thin Hydra entrypoints.
- `src/pinn_project/configs/` owns project composition and machine-independent defaults.
- `src/pinn_project/applications/` owns application data preparation, DataModules, evaluation, and plotting.
- Application packages may import PhiJAX. PhiJAX must never import this project.
- Keep coordinate order, output order, units, transforms, and artifact semantics explicit in each application.

## Engineering standards

- Use Python 3.12 or newer and `uv` for environment and command execution.
- Keep JAX PRNG keys explicit and transformed computations free from logging, filesystem IO, NumPy conversion, and
  Python-side mutation.
- Construct host data with NumPy and let the PhiJAX Trainer place batches on devices.
- Prefer config composition over hard-coded experiment switches.
- Add complete Google-style docstrings to every class, function, method, and private helper; omit module docstrings.
- Keep line length at 120 characters and use ASCII in newly written comments.
- Prefer concise, explicit state and clear abstractions. Avoid dynamic attributes and one-use trivial helpers.
- Do not store credentials in Hydra configs because resolved configs may be printed and saved.
- Keep ordinary tests CPU-only, synthetic, and independent of network services or external research datasets.

## Validation

Run:

```bash
JAX_PLATFORMS=cpu uv run --no-sync pytest
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pyright
uv run --no-sync mkdocs build --strict
```

Do not commit, push, or modify remote state unless the user explicitly requests it.
