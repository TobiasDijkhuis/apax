from functools import partial

import jax.numpy as jnp
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT
from jax import value_and_grad

from apax.layers.empirical import WrappedCalculator
from apax.utils.data import make_minimal_padded_input, make_minimal_input

test_n_pad = [0, 1, 7]


@pytest.mark.parametrize("n_pad", test_n_pad)
def test_wrapped_calculator_from_ase_calc(n_pad):
    R, Z, idx, box, offsets = make_minimal_padded_input(n_pad=n_pad)

    calc = EMT()
    wrapped_calc = WrappedCalculator(dtype="fp64", calculator=calc)
    ef_func = partial(value_and_grad(wrapped_calc.apply, argnums=1), {})

    energy, grad = ef_func(R, None, Z, idx, box, None)

    assert energy == calc.results["energy"]
    assert jnp.allclose(grad[:2], -calc.results["forces"])


@pytest.mark.parametrize("n_pad", test_n_pad)
def test_wrapped_calculator_from_name(n_pad):
    calculator_name = "ase.calculators.emt.EMT"
    calculator_kwargs = {}
    wrapped_calc = WrappedCalculator(
        dtype="fp64", calculator_name=calculator_name, calculator_kwargs=calculator_kwargs
    )
    ef_func = partial(value_and_grad(wrapped_calc.apply, argnums=1), {})

    R, Z, idx, box, offsets = make_minimal_padded_input(n_pad=n_pad)
    energy, grad = ef_func(R, None, Z, idx, box, None)

    R, Z, idx, box, offsets = make_minimal_input()
    calc = EMT()
    calc.calculate(atoms=Atoms(positions=R, numbers=Z, cell=box))

    assert energy == calc.results["energy"]
    assert jnp.allclose(grad[:2], -calc.results["forces"])
