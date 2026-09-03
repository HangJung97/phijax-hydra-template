# AGENTS.md

These instructions apply to the entire generated project. They adapt the coding, documentation, testing, and tooling
standards from [PhiJAX's `AGENTS.md`](https://github.com/HangJung97/PhiJAX/blob/main/AGENTS.md) to a downstream Hydra
project.

When working in this repository, act as an expert Python engineer. Use PhiJAX as the reusable JAX framework and keep
executable project configuration and domain applications here. The PhiJAX guides show how to create project-specific
[data](https://hangjung97.github.io/PhiJAX/guides/datasets/),
[models](https://hangjung97.github.io/PhiJAX/guides/models/),
[objectives](https://hangjung97.github.io/PhiJAX/guides/objectives/), and
[balancers](https://hangjung97.github.io/PhiJAX/guides/balancers/). If a feature would benefit other PhiJAX users,
consider contribute it to PhiJAX. Follow the upstream repository instructions when doing so.

## Architecture

- `src/phijax_hydra_template/{train,predict,evaluate_predictions}.py` are thin Hydra entrypoints.
- `src/phijax_hydra_template/configs/` owns project composition and machine-independent defaults.
- `src/phijax_hydra_template/configs/hparams_search/` owns project-specific search metrics, samplers, and ranges.
- `src/phijax_hydra_template/applications/` owns application data preparation, DataModules, evaluation, and plotting.
- Burgers is the complete reference application. Keep its coordinate order, output order, units, transforms, loss
  names, and artifact semantics explicit.
- Application packages may import public PhiJAX APIs. PhiJAX must never import this project.
- Keep framework callback, logger, optimizer, scheduler, and balancer options in their project config groups.

## Environment and validation

- Use Python 3.12 or newer and `uv` for environment and command execution.
- Install the CPU environment with `uv sync`. Select at most one CUDA extra when an accelerator is required.
- Keep ordinary validation CPU-only, synthetic, and independent of networks, external services, and research datasets.
- Run the complete project validation suite:

```bash
JAX_PLATFORMS=cpu uv run --no-sync pytest
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pyright
```

- Respect `.pre-commit-config.yaml`, including Ruff, docformatter, YAML formatting, shellcheck, mdformat, codespell,
  nbstripout, and Renovate validation.

## Implementation standards

- Import Trainer, PhiModule, DataModule, models, objectives, balancers, callbacks, persistence, and evaluation through
  supported public PhiJAX exports.
- Keep JAX PRNG keys explicit. Split keys deliberately for model initialization, sampling, balancing, and training.
- Keep transformed computations functional and their PyTree structures, shapes, and dtypes stable.
- Do not perform logging, filesystem IO, NumPy conversion, configuration access, or Python-side mutation inside
  `jax.jit`, `jax.grad`, `jax.vmap`, or `jax.lax` transformations.
- Avoid data-dependent array shapes and Python control flow over traced values. Keep static configuration outside JAX
  transformations.
- Construct host data with NumPy. Let the PhiJAX Trainer and Strategy place batches on devices.
- Preserve `float32` parameters, optimizer state, derivatives, losses, and NTK traces unless an explicit precision
  policy says otherwise.
- Keep evaluation independent of JAX training code when the public PhiJAX evaluator supports the workflow.
- Prefer Hydra config composition over hard-coded experiment switches.
- Do not store credentials in Hydra configs because resolved configs may be printed and saved.
- Restrict output, external logging, artifact writes, and checkpoint commits to global rank zero unless an API provides
  safe distributed semantics.
- Ensure resource-owning callbacks, loggers, DataModules, and checkpoint backends release resources after success,
  interruption, early stopping, and exceptions.

## Coding style

- Keep line length at 120 characters and use ASCII in newly written comments.
- Prefer concise, self-explanatory code, concrete names, explicit state, and clear abstractions.
- Match existing naming, typing, configuration, and architectural patterns before adding an abstraction.
- Declare class attributes and PyTree structure deliberately. Avoid hidden runtime state through dynamic `setattr` or
  `getattr` patterns.
- Do not introduce a trivial one- or two-line helper used once unless it clarifies a meaningful lifecycle, tracing,
  transformation, or reusable contract boundary.
- Use comments for non-local context such as coordinate or output semantics, mathematical assumptions, PRNG behavior,
  device placement, and JAX retracing constraints. Do not restate nearby code.
- When the line limit causes awkward wrapping, first shorten a name or introduce a descriptive local variable.
- When several implementations are equally correct, choose the simpler and more concise one.

## Documentation guide

- Use the [PhiJAX docs](https://hangjung97.github.io/PhiJAX/) as the source for framework terms and API
  behavior. Link to them instead of repeating framework reference material.
- Follow the broad documentation patterns used by PyTorch, Lightning, and MONAI: separate first steps, task guides,
  examples, and API reference; start task pages with the result the reader will achieve; keep domain terms close to
  the example that uses them.
- Write for readers who know Python and basic JAX but may be new to PhiJAX. Explain a new term when it first appears.
- Keep one main task per guide. State the result first, then show the shortest working command or example.
- Prefer common words, concrete verbs, and short sentences. Avoid terms such as "instantiate", "orchestration",
  "lifecycle policy", and "cadence" unless they are exact API terms. Explain them when they are required.
- Use task guides to explain how to do work. Use API pages to list exact classes, functions, parameters, and return
  values.
- Use descriptive headings and executable examples. Use tables for exact mappings and option comparisons. Write
  troubleshooting items as symptom, cause, and fix.
- Preserve established JAX, PINN, mathematical, and public API terminology. Avoid filler, promotional language, vague
  wording, and unnecessary qualifiers.
- Keep examples aligned with the supported public PhiJAX API. Explain non-obvious state ownership, array shapes,
  coordinate order, PRNG behavior, and device placement close to the relevant example.
- Do not add module-level docstrings to Python files.
- Add complete Google-style docstrings to every class, function, method, private helper, and private method.
- In docstrings, use single backticks for variables, parameters, attributes, config keys, array shapes, and literals.
- Use Sphinx/reST roles such as `:class:`, `:func:`, `:meth:`, `:mod:`, `:attr:`, and `:paramref:` for documented
  cross-references.
- Add short inline comments only for non-obvious array semantics, physics, PRNG behavior, device placement, or tracing
  constraints.

## Testing requirements

- Add or update tests for every changed class, function, method, and module.
- Put focused application array, generation, plotting, and evaluation checks under `tests/unit/`.
- Put Hydra composition and cross-component runtime workflows under `tests/integration/`.
- Test array shapes, dtypes, finite values and gradients, singleton dimensions, empty targets or masks, padded final
  chunks, and deterministic PRNG behavior when relevant.
- Treat PDE equations, boundary conditions, objective terms, coordinate order, output semantics, and NTK streams as
  correctness-sensitive.
- Do not compare independent random initializations for parity. Reuse identical parameters, keys, and inputs.

## Change and tooling discipline

- Treat public imports, Hydra targets, config paths, and artifact contracts as user-facing APIs. Update exports,
  documentation, tests, and downstream usage together when any of them changes.
- Preserve unrelated user changes in a dirty worktree.
- Put temporary scripts, extracted diagnostics, and throwaway benchmarks under `/tmp`.
- Avoid modifying generated outputs, experiment logs, local data, notebook outputs, or machine-specific configs.
- Do not commit, amend, push, open a pull request, or modify remote state unless the user explicitly requests it.
- If validation cannot run, report why and provide the exact remaining commands.

## Final response

- Summarize intent, user-visible behavior, validation commands, and any configuration migration notes.
- Call out renamed package paths, changed defaults, backward-incompatible behavior, and CPU-only validation.
