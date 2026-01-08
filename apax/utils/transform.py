from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator


def make_energy_only_model(energy_properties_model):
    energy_model = lambda *args, **kwargs: energy_properties_model(*args, **kwargs)[0]
    return energy_model


def create_energy_fn(
    calculator: Calculator,
) -> Callable[[jax.Array, jax.Array, jax.Array], float]:
    """Transform an ASE calculator to be differentiable by JAX.

    Args:
        calculator (Calculator): ASE calculator

    Returns:
        energy_fn (Callable): function that calculates the energy of a system.
        This function takes R, Z, and box and returns the energy in eV.
    """

    def base_energy_fn(R: jax.Array, Z: jax.Array, box) -> float:
        """Calculate the energy of the system using an ASE calculator

        Args:
            R (np.ndarray): postition vectors
            Z (np.ndarray): atomic numbers

        Returns:
            energy (float): total energy of the system.
                Units are ASE internal units, so eV
        """

        atoms = Atoms(positions=R, numbers=Z, cell=box)
        energy = calculator.get_potential_energy(atoms=atoms)
        return energy

    def base_force_fn(R: jax.Array, Z: jax.Array, box) -> np.ndarray:
        """Calculate the forces using an ASE calculator

        Args:
            R (np.ndarray): postition vectors
            Z (np.ndarray): atomic numbers

        Returns:
            forces (np.ndarray): force vectors, with same shape as R.
                Units are ASE internal units, so eV/Angstrom
        """
        atoms = Atoms(positions=R, numbers=Z, cell=box)
        forces = calculator.get_forces(atoms=atoms)
        return forces

    @jax.custom_vjp
    def energy_fn(
        R: jax.Array,
        dr_vec: jax.Array,
        Z: jax.Array,
        idx: list,
        box: jax.Array,
        properties: dict,
    ) -> float:
        """Calculate the energy of the system using an ASE calculator.
        This was made to be auto-differentiable, such that jax.grad on this
        function returns the gradient of the energy (i.e. the force with
        opposite sign) in eV/Angstrom.

        Args:
            R (jax.Array): position vectors. Can be padded
            dr_vec (jax.Array): not used
            Z (jax.Array): atomic numbers. Can be padded
            idx (jax.Array): not used
            box (jax.Array): box of periodic boundary conditions
            properties (dict): not used

        Returns:
            float: energy in eV (ASE internal units)
        """
        n_nonpadded = jnp.count_nonzero(Z)
        return jax.pure_callback(
            base_energy_fn,
            jax.ShapeDtypeStruct((), float),
            R[:n_nonpadded],
            Z[:n_nonpadded],
            box,
        )

    def energy_fn_fwd(R, dr_vec, Z, idx, box, properties) -> tuple[float, jax.Array]:
        energy = energy_fn(R, dr_vec, Z, idx, box, properties)

        n_nonpadded = jnp.count_nonzero(Z)
        # The gradient of the energy has opposite sign of the force.
        energy_grad = -jax.pure_callback(
            base_force_fn,
            jax.ShapeDtypeStruct((n_nonpadded, 3), float),
            R[:n_nonpadded],
            Z[:n_nonpadded],
            box,
        )

        zeros_to_add = Z.shape[0] - n_nonpadded
        energy_grad = jnp.pad(energy_grad, ((0, zeros_to_add), (0, 0)), "constant")
        return energy, (energy_grad, None, None, None, None, None)

    def energy_fn_bwd(res, g):
        return res

    energy_fn.defvjp(energy_fn_fwd, energy_fn_bwd)

    return energy_fn
