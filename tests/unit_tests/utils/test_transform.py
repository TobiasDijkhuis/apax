from apax.utils.transform import create_energy_fn
from apax.utils.data import make_minimal_input
import jax
import numpy as np


def test_create_energy_fn(dummy_calculator):
    R, Z, idx, box, offsets = make_minimal_input()
    energy_fn = create_energy_fn(dummy_calculator)
    energy, grad = jax.value_and_grad(energy_fn)(R, None, Z, box, None, None)
    assert np.allclose(grad, -dummy_calculator.results["forces"])
    assert np.allclose(energy, dummy_calculator.results["energy"])
