import os

import pytest

os.environ["JAX_PLATFORMS"] = "cpu"


@pytest.fixture(scope="session", autouse=True)
def register_test_omegaconf_resolvers() -> None:
    """Register PhiJAX resolvers before tests compose or resolve Hydra configurations."""
    from phijax_hydra_template.configs import register_omegaconf_resolvers

    register_omegaconf_resolvers()
