from functools import partial

import jax.numpy as jnp
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from jax import value_and_grad

from apax.layers.empirical import WrappedCalculator
from apax.utils.data import make_minimal_padded_input

test_data = [(1.0, 0), (2.0, 0), (-0.5, 0), (6.5, 10)]


class DummyCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, spring_constant: float = 1.0, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.spring_constant = spring_constant

    def calculate(
        self, atoms: Atoms, properties: list[str] = ["energy"], system_changes=all_changes
    ) -> None:
        Calculator.calculate(
            self, atoms=atoms, properties=properties, system_changes=system_changes
        )

        positions = atoms.get_positions()

        def harmonic_potential(x1, x2) -> float:
            return self.spring_constant * 0.5 * jnp.linalg.norm(x1 - x2) ** 2

        energy, neg_force = value_and_grad(harmonic_potential, argnums=(0, 1))(
            positions[0], positions[1]
        )

        self.results["energy"] = energy
        self.results["forces"] = -jnp.stack(neg_force)


@pytest.mark.parametrize("spring_constant, n_pad", test_data)
def test_delta_ml(spring_constant, n_pad):
    R, Z, idx, box, offsets = make_minimal_padded_input(n_pad=n_pad)

    ase_calc = DummyCalculator(spring_constant=spring_constant)
    calc = WrappedCalculator(calculator=ase_calc)
    ef_func = partial(value_and_grad(calc.apply, argnums=1), {})

    energy, grad = ef_func(R, None, Z, idx, box, None)

    assert energy == ase_calc.results["energy"]
    assert jnp.allclose(grad[:2], -ase_calc.results["forces"])
