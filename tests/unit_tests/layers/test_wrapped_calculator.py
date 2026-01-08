from functools import partial

import jax.numpy as jnp
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from jax import value_and_grad

from apax.layers.empirical import WrappedCalculator
from apax.utils.data import make_minimal_padded_input


@pytest.mark.parametrize("n_pad", (0, 1, 7))
def test_wrapped_calculator_from_ase_calc(dummy_calculator, n_pad):
    R, Z, idx, box, offsets = make_minimal_padded_input(n_pad=n_pad)

    ase_calc = dummy_calculator
    calc = WrappedCalculator(calculator=ase_calc)
    ef_func = partial(value_and_grad(calc.apply, argnums=1), {})

    energy, grad = ef_func(R, None, Z, idx, box, None)

    assert energy == ase_calc.results["energy"]
    assert jnp.allclose(grad[:2], -ase_calc.results["forces"])
