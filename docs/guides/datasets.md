# Configure application data

An application DataModule prepares data for training and prediction. It also defines how PhiJAX samples each batch.

This project includes `BurgersDataModule` as a working example. For the full data API, see the
[PhiJAX data guide](https://hangjung97.github.io/PhiJAX/guides/datasets/) and
[data API](https://hangjung97.github.io/PhiJAX/api/data/).

## What the DataModule owns

The Burgers DataModule decides:

- where the data comes from;
- which arrays belong to each training pool;
- how each pool is sampled;
- how many rows are used in one step; and
- how prediction points are split into chunks.

The PhiJAX Trainer moves batches to the selected device. The DataModule builds host data with NumPy.

Input normalization is optional. The base `DataModule.input_statistics()` returns `None`. The Burgers DataModule opts
in by calling the public `phijax.data.input_statistics()` helper for its fixed training pools.

## Burgers data options

The main data config is `configs/data/burgers_1d.yaml`.

| Option         | Meaning                                             |
| -------------- | --------------------------------------------------- |
| `data_path`    | Path to the generated Burgers reference file        |
| `seed`         | Seed used to create fixed PDE points                |
| `pde_sampling` | `fixed` reuses a pool; `uniform` creates new points |
| `pde_size`     | Number of points in the fixed PDE pool              |
| `batch_size`   | Rows used in one step or prediction chunk           |

Use the small analytic config when you do not need reference targets:

```bash
phijax-train experiment=burgers_grad_norm_1d data=burgers_analytic_1d
```

## Data pools

The Burgers application uses three named pools:

| Pool      | Inputs   | Targets | Purpose                             |
| --------- | -------- | ------- | ----------------------------------- |
| `initial` | `[t, x]` | `[u]`   | Enforce the initial condition       |
| `pde`     | `[t, x]` | none    | Evaluate the Burgers equation       |
| `predict` | `[t, x]` | `[u]`   | Create and evaluate the output grid |

The pool names match the equation keys in the objective config. If you change a pool name, update the objective too.

PhiJAX stores each pool in a `HostPool`. A pool contains:

- `inputs`: a two-dimensional NumPy array;
- `targets`: a two-dimensional NumPy array, including zero-width targets for PDE-only points;
- `aux`: optional sample data, such as weights or normals;
- `metadata`: coordinate and output names; and
- `reference_shape` and `flat_index`: values used to rebuild a prediction grid.

## Fixed and uniform sampling

Fixed sampling creates `pde_size` points once. Each step selects a batch from this pool:

```yaml
pde_sampling: fixed
pde_size: 16384
batch_size:
  pde: 4096
```

Uniform sampling creates a new batch inside the domain at each step:

```bash
phijax-train experiment=burgers_grad_norm_1d data.pde_sampling=uniform
```

Both modes use explicit JAX keys. The same seed and training step produce the same points.

## Prediction chunks

`predict_shape` sets the full prediction grid. `batch_size.predict` sets how many rows are evaluated at once.

```yaml
predict_shape: [100, 256]
batch_size:
  predict: 4096
```

This grid has 25,600 points. PhiJAX evaluates it in chunks of at most 4,096 rows.

## Add a new application

1. Create `applications/<name>/data.py` for NumPy data loading and pool construction.
2. Create `applications/<name>/datamodule.py` and subclass `phijax.DataModule`.
3. Implement `prepare_data()`, `setup()`, and `train_batch_source()`.
4. Add prediction hooks when the application supports prediction.
5. Add a config under `configs/data/`.
6. Match every objective `batch_key` with a sampler and batch size.

Keep file keys, coordinate order, output order, units, and transforms in the application package. The
[PhiJAX data guide](https://hangjung97.github.io/PhiJAX/guides/datasets/) shows how to create data contracts and
samplers.

## Test the DataModule

Use small synthetic arrays. Test:

- pool names and shapes;
- coordinate and output order;
- deterministic sampling;
- fixed and uniform sampling;
- prediction chunking; and
- invalid sizes and options.

The tests under `tests/unit/applications/burgers/` show this pattern.
