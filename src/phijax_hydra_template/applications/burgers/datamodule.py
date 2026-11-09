from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

import jax
import numpy as np
from numpy.typing import NDArray
from phijax import DataModule
from phijax.data import (
    BatchSize,
    DataStage,
    HostPool,
    NamedBatchSource,
    RandomRowSampler,
    TrainingBatchSource,
    UniformDomainSampler,
    input_statistics,
    load_arrays,
)

from phijax_hydra_template.applications.burgers.data import build_burgers_file_pools, build_burgers_pools
from phijax_hydra_template.applications.burgers.generation import ensure_burgers_dataset


class BurgersDataModule(DataModule):
    """Own Burgers dataset preparation, host pools, and PDE sampling policy.

    The PhiJAX base class chunks the `predict` pool by `batch_sizes["predict"]` and samples finite pools by name.

    Args:
        batch_sizes: Per-device positive integer or `all` batch policies keyed by pool name. The `predict` entry sets
            the prediction chunk size.
        seed: NumPy seed controlling the finite PDE candidate pool.
        data_path: Optional reference artifact. When absent, analytic initial, PDE, and prediction pools are generated
            directly without reference prediction targets.
        initial_size: Number of analytic initial-condition coordinates used when `data_path` is absent.
        pde_size: Number of finite uniformly distributed PDE candidate coordinates.
        pde_sampling: `fixed` to sample rows from a finite candidate pool or `uniform` to generate fresh interior
            coordinates at every optimizer step.
        predict_shape: Dense analytic prediction shape used when `data_path` is absent.
        time_bounds: Analytic temporal bounds used when `data_path` is absent.
        space_bounds: Analytic spatial bounds used when `data_path` is absent.
        name: Filename-safe dataset identifier.

    Attributes:
        name: Filename-safe dataset identifier used by prediction artifacts.
    """

    def __init__(
        self,
        batch_sizes: Mapping[str, BatchSize],
        *,
        seed: int = 42,
        data_path: str | Path | None = None,
        initial_size: int = 256,
        pde_size: int = 16384,
        pde_sampling: Literal["fixed", "uniform"] = "fixed",
        predict_shape: Sequence[int] = (100, 256),
        time_bounds: Sequence[float] = (0.0, 1.0),
        space_bounds: Sequence[float] = (-1.0, 1.0),
        name: str = "burgers_1d",
    ) -> None:
        """Store application data policy without constructing arrays or initializing JAX.

        Args:
            batch_sizes: Per-device training batch and prediction chunk sizes keyed by pool name.
            seed: Finite PDE candidate-pool seed.
            data_path: Optional reference artifact path.
            initial_size: Analytic initial-condition pool size.
            pde_size: Finite PDE candidate-pool size.
            pde_sampling: Interior collocation sampling policy.
            predict_shape: Analytic dense prediction shape.
            time_bounds: Analytic temporal bounds.
            space_bounds: Analytic spatial bounds.
            name: Filename-safe dataset identifier.

        Raises:
            ValueError: If `name` is empty or `pde_sampling` is unsupported.
        """
        super().__init__(batch_sizes=batch_sizes)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Burgers dataset `name` must be a non-empty string.")
        if pde_sampling not in ("fixed", "uniform"):
            raise ValueError("Burgers `pde_sampling` must be `fixed` or `uniform`.")
        self.seed = seed
        self.data_path = None if data_path is None else Path(data_path).expanduser().resolve()
        self.initial_size = initial_size
        self.pde_size = pde_size
        self.pde_sampling = pde_sampling
        self.predict_shape = cast(tuple[int, int], tuple(int(value) for value in predict_shape))
        self.time_bounds = cast(tuple[float, float], tuple(float(value) for value in time_bounds))
        self.space_bounds = cast(tuple[float, float], tuple(float(value) for value in space_bounds))
        self.name = name

    def prepare_data(self) -> None:
        """Generate the configured reference artifact when it is absent."""
        if self.data_path is not None:
            ensure_burgers_dataset(self.data_path)

    def setup(self, stage: DataStage) -> None:
        """Construct and store finite Burgers pools for fitting or prediction.

        Args:
            stage: Requested `fit` or `predict` lifecycle stage.

        Raises:
            ValueError: If `stage` is unsupported.
        """
        if stage not in ("fit", "predict"):
            raise ValueError("Burgers data stage must be `fit` or `predict`.")
        # `uniform` mode draws fresh PDE coordinates per step, so it builds no finite PDE pool.
        pde_size = self.pde_size if self.pde_sampling == "fixed" else None
        if self.data_path is None:
            self.pools = build_burgers_pools(
                seed=self.seed,
                initial_size=self.initial_size,
                pde_size=pde_size,
                predict_shape=self.predict_shape,
                time_bounds=self.time_bounds,
                space_bounds=self.space_bounds,
            )
        else:
            self.pools = build_burgers_file_pools(load_arrays(self.data_path), seed=self.seed, pde_size=pde_size)

    def train_batch_source(
        self,
        batch_keys: tuple[str, ...],
        key: jax.Array,
    ) -> TrainingBatchSource:
        """Sample finite pools by name, and draw fresh PDE coordinates in `uniform` mode.

        Args:
            batch_keys: Objective batch keys in stable declaration order.
            key: Explicit root sampling key.

        Returns:
            Unprepared deterministic source that the Trainer places before sampling.
        """
        if self.pde_sampling == "fixed":
            return super().train_batch_source(batch_keys, key)
        pools = self._require_setup("fit")
        samplers = {}
        for batch_key in batch_keys:
            if batch_key == "pde":
                # Preserve the zero-width target field so both PDE policies expose the same batch structure.
                templates = {"targets": np.empty((0,), dtype=np.float32)}
                samplers[batch_key] = UniformDomainSampler(
                    bounds=self._domain_bounds(pools["predict"]),
                    templates=templates,
                )
            else:
                samplers[batch_key] = RandomRowSampler(pools[batch_key].fields())
        batch_sizes = cast(Mapping[str, BatchSize], self.batch_sizes)
        return NamedBatchSource(samplers, {batch_key: batch_sizes[batch_key] for batch_key in batch_keys}, key)

    def normalization_pools(self) -> Mapping[str, HostPool]:
        """Select a finite domain-representative pool when one is requested directly.

        Returns:
            Finite PDE candidates in `fixed` mode or the dense prediction grid in `uniform` mode.
        """
        pools = self._require_setup("fit")
        name = "pde" if self.pde_sampling == "fixed" else "predict"
        return {name: pools[name]}

    def input_statistics(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Resolve empirical or exact uniform-domain input statistics.

        Returns:
            Per-coordinate means and standard deviations in model input order.
        """
        if self.pde_sampling == "fixed":
            mean, std = input_statistics(self.normalization_pools())
            return mean.astype(np.float32), std.astype(np.float32)
        pools = self._require_setup("fit")
        bounds = self._domain_bounds(pools["predict"])
        mean = ((bounds[:, 0] + bounds[:, 1]) / 2.0).astype(np.float32)
        std = ((bounds[:, 1] - bounds[:, 0]) / np.sqrt(12.0)).astype(np.float32)
        return mean, std

    def _domain_bounds(self, prediction_pool: HostPool) -> NDArray[np.float32]:
        """Resolve continuous coordinate bounds in Burgers model-input order.

        Args:
            prediction_pool: Dense prediction pool carrying file-derived coordinate metadata when available.

        Returns:
            Array with shape `[2, 2]` ordered as `[t, x]` and `[lower, upper]`.

        Raises:
            ValueError: If file-derived bounds are absent or inconsistent with the Burgers coordinate order.
        """
        if self.data_path is None:
            return np.asarray((self.time_bounds, self.space_bounds), dtype=np.float32)
        coordinate_names = tuple(str(name) for name in prediction_pool.metadata.get("coordinate_names", ()))
        raw_bounds = prediction_pool.metadata.get("sampling_bounds")
        if coordinate_names != ("t", "x") or raw_bounds is None:
            raise ValueError("Burgers uniform PDE sampling requires prediction bounds ordered as `t` and `x`.")
        named_bounds = {str(name): (float(lower), float(upper)) for name, lower, upper in raw_bounds}
        return np.asarray([named_bounds[name] for name in coordinate_names], dtype=np.float32)


__all__ = ["BurgersDataModule"]
