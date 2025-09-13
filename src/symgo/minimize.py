"""Functions used for minimize energy."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from symfc.utils.utils import SymfcAtoms

from symgo.interface.base import PropertyCalculator


@dataclasses.dataclass
class MinimizeFunctionParams:
    """Arguments for scipy.optimize.minimize function.

    args: tuple[ArgsForMinimizeFunction]

    Used as follows:

    res = minimize(
        fun,
        x0,
        (args,),
        method=method,
        jac=jac,
        options=options,
        constraints=[nlc],
    )

    """

    prop: PropertyCalculator
    size_pos: int
    orig_x: NDArray
    orig_structure: SymfcAtoms
    basis_axis: NDArray | None
    position_relaxation: bool
    lattice_relaxation: bool
    volume_relaxation: bool
    basis_frac: NDArray | None
    pressure: float


def function_fix_cell(x: NDArray, params: MinimizeFunctionParams) -> ArrayLike:
    """Target function when performing no cell optimization.

    args is a dummy variable for scipy.optimize.minimize.

    """
    structure = to_structure(
        x,
        params.size_pos,
        params.orig_x,
        params.orig_structure,
        params.basis_axis,
        params.lattice_relaxation,
        params.volume_relaxation,
        params.basis_frac,
    )
    params.prop.eval(structure)
    energy = params.prop.energy

    if energy < -1e3 * len(structure):
        print("Energy =", energy)
        print("Axis :")
        print(structure.cell)
        print("Fractional coordinates:")
        print(structure.scaled_positions)
        raise ValueError("Geometry optimization failed: Huge negative energy value.")

    energy += params.pressure * np.linalg.det(structure.cell)
    return energy


def jacobian_fix_cell(x: NDArray, params: MinimizeFunctionParams) -> ArrayLike:
    """Target Jacobian function when performing no cell optimization."""
    if params.basis_frac is not None:
        structure = to_structure(
            x,
            params.size_pos,
            params.orig_x,
            params.orig_structure,
            params.basis_axis,
            params.lattice_relaxation,
            params.volume_relaxation,
            params.basis_frac,
        )
        prod = -(structure.cell @ params.prop.force).T
        derivatives = params.basis_frac.T @ prod.ravel()
        return derivatives
    return []


def function_relax_cell(x: NDArray, params: MinimizeFunctionParams) -> ArrayLike:
    """Target function when performing cell optimization."""
    structure = to_structure(
        x,
        params.size_pos,
        params.orig_x,
        params.orig_structure,
        params.basis_axis,
        params.lattice_relaxation,
        params.volume_relaxation,
        params.basis_frac,
    )
    params.prop.eval(structure)
    energy = params.prop.energy

    if (
        energy < -1e3 * len(structure)
        or abs(np.linalg.det(structure.cell)) / len(structure) > 1000
    ):
        print("Energy =", energy)
        print("Lattice vectors:")
        print(structure.cell)
        print("Fractional coordinates:")
        print(structure.scaled_positions)
        raise ValueError(
            "Geometry optimization failed: Huge negative energy value"
            "or huge volume value."
        )

    energy += params.pressure * np.linalg.det(structure.cell)
    return energy


def jacobian_relax_cell(x: NDArray, params: MinimizeFunctionParams) -> ArrayLike:
    """Target Jacobian function when performing cell optimization.

    args is a dummy variable for scipy.optimize.minimize.

    """
    partition1 = params.size_pos
    derivatives = np.zeros(len(x))
    if params.position_relaxation:
        derivatives[:partition1] = jacobian_fix_cell(x, params)
    structure = to_structure(
        x,
        params.size_pos,
        params.orig_x,
        params.orig_structure,
        params.basis_axis,
        params.lattice_relaxation,
        params.volume_relaxation,
        params.basis_frac,
    )
    derivatives[partition1:] = _derivatives_by_axis(
        structure,
        params.prop.stress,
        params.pressure,
        params.basis_axis,
        params.lattice_relaxation,
    )
    return derivatives


def to_volume(size_pos: int, basis_axis: NDArray | None) -> Callable:
    """Return function to compute volume from x."""

    def _to_volume(x: NDArray) -> float:
        _, x_cells = _split(x, size_pos)
        axis = basis_axis @ x_cells
        return np.linalg.det(axis.reshape((3, 3)))

    return _to_volume


def refine_positions(positions: NDArray, tol=1e-13):
    """Refine atomic positions to be within [0, 1)."""
    positions -= np.floor(positions)
    positions[np.where(positions > 1 - tol)] -= 1.0
    return positions


def to_structure(
    x: NDArray,
    size_pos: int,
    orig_x: NDArray,
    orig_structure: SymfcAtoms,
    basis_axis: NDArray | None,
    lattice_relaxation: bool,
    volume_relaxation: bool,
    basis_frac: NDArray | None,
) -> SymfcAtoms:
    """Convert x to structure."""
    x_positions, x_cells = _split(x, size_pos)
    if lattice_relaxation:
        axis = basis_axis @ x_cells
        cell = axis.reshape((3, 3)).T
    elif volume_relaxation:
        x_cells0 = _split(orig_x, size_pos)[1][0]
        scale = (x_cells[0] / x_cells0) ** (1 / 3)
        cell = orig_structure.cell * scale
    else:
        cell = orig_structure.cell

    return SymfcAtoms(
        cell=cell,
        numbers=orig_structure.numbers,
        scaled_positions=_to_relaxed_positions(x_positions, basis_frac, orig_structure),
    )


def _split(x: NDArray, size_pos: int):
    """Split coefficients."""
    partition1 = size_pos
    x_pos = x[:partition1]
    x_axis = x[partition1:]
    return x_pos, x_axis


def _to_relaxed_positions(
    x: NDArray, basis_frac: NDArray | None, orig_structure: SymfcAtoms
) -> NDArray:
    """Convert x to structure."""
    if basis_frac is not None:
        disps_f = (basis_frac @ x).reshape(-1, 3)
        return refine_positions(orig_structure.scaled_positions + disps_f)
    return orig_structure.scaled_positions


def _derivatives_by_axis(
    structure: SymfcAtoms,
    stress: NDArray,
    pressure: float,
    basis_axis: NDArray | None,
    lattice_relaxation: bool,
) -> NDArray | float:
    """Compute derivatives with respect to axis elements.

    PV @ axis_inv.T is exactly the same as the derivatives of PV term
    with respect to axis components.

    Under the constraint of a fixed cell shape, the mean normal stress
    serves as an approximation to the derivative of the enthalpy
    with respect to volume.
    """
    pv = pressure * np.linalg.det(structure.cell)
    sigma = [
        [stress[0] - pv, stress[3], stress[5]],
        [stress[3], stress[1] - pv, stress[4]],
        [stress[5], stress[4], stress[2] - pv],
    ]
    if lattice_relaxation:
        """derivatives_s: In the order of ax, bx, cx, ay, by, cy, az, bz, cz"""
        derivatives_s = -np.array(sigma) @ np.linalg.inv(structure.cell)
        assert basis_axis is not None
        derivatives_s = basis_axis.T @ derivatives_s.ravel()
    else:
        derivatives_s = -np.trace(np.array(sigma)) / 3

    return derivatives_s
