from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


def test_soap_config_updates_nnx_parameters() -> None:
    """Verify the scheduled SOAP config preserves NNX state across preconditioner refreshes."""
    config_dir = Path(__file__).parents[2] / "src" / "phijax_hydra_template" / "configs"
    with initialize_config_dir(config_dir=str(config_dir.resolve()), version_base=None):
        config = compose(
            config_name="train",
            overrides=[
                "experiment=burgers_grad_norm_1d",
                "model/optimizer=soap",
                "model/scheduler=constant",
                "model.scheduler.value=0.001",
            ],
        )
    optimizer = instantiate(config.model.optimizer)
    _, params = nnx.split(nnx.Linear(2, 1, rngs=nnx.Rngs(0)))
    original_params = params
    structure = jax.tree.structure(params)
    state = optimizer.init(params)
    update = jax.jit(optimizer.update)
    for _ in range(24):
        gradients = jax.tree.map(jnp.ones_like, params)
        updates, state = update(gradients, state, params)
        assert jax.tree.structure(updates) == structure
        params = optax.apply_updates(params, updates)
        for leaf in jax.tree.leaves((params, state)):
            assert bool(jnp.isfinite(leaf).all())
            if jnp.issubdtype(leaf.dtype, jnp.floating):
                assert leaf.dtype == jnp.float32
    assert any(
        bool(jnp.any(before != after))
        for before, after in zip(jax.tree.leaves(original_params), jax.tree.leaves(params), strict=True)
    )
