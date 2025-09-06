"""Class for geometry optimization with symmetric constraint."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
import spglib
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import NonlinearConstraint, minimize
from symfc.basis_sets import FCBasisSetO1
from symfc.utils.utils import SymfcAtoms


def print_structure(structure: SymfcAtoms):
    """Print structure."""
    print("basis vectors:")
    for a in structure.cell:
        print(f" - {a}")
    print("Fractional coordinates:")
    for p, e in zip(structure.scaled_positions, structure.numbers):
        print(f" - {e} {p}")


def _refine_positions(cell: SymfcAtoms, tol=1e-13):
    """Refine atomic positions to be within [0, 1)."""
    positions = cell.scaled_positions
    positions -= np.floor(positions)
    positions[np.where(positions > 1 - tol)] -= 1.0
    return SymfcAtoms(cell=cell.cell, numbers=cell.numbers, scaled_positions=positions)


def _construct_basis_fractional_coordinates(cell: SymfcAtoms) -> NDArray | None:
    """Generate a basis set for atomic positions in fractional coordinates."""
    basis_cart = _construct_basis_cartesian(cell)
    if basis_cart is None or basis_cart.size == 0:
        return None
    basis_f = _basis_cartesian_to_fractional_coordinates(basis_cart, cell)
    return basis_f


def _construct_basis_cartesian(cell: SymfcAtoms) -> NDArray | None:
    """Generate a basis set for atomic positions in Cartesian coordinates."""
    try:
        fc_basis = FCBasisSetO1(cell).run()
    except ValueError:
        return None
    assert fc_basis.full_basis_set is not None
    return fc_basis.full_basis_set.toarray()


def _basis_cartesian_to_fractional_coordinates(
    basis_cart: NDArray, unitcell: SymfcAtoms
) -> NDArray:
    """Convert basis set in Cartesian coord. to basis set in fractional coordinates."""
    n_basis = basis_cart.shape[1]
    n_atom = len(unitcell)
    inv_lattice = np.linalg.inv(unitcell.cell)

    basis_cart = np.array([b.reshape((n_atom, 3)) for b in basis_cart.T])
    basis_cart = basis_cart.transpose((2, 1, 0))
    basis_cart = basis_cart.reshape(3, -1)
    basis_frac = inv_lattice.T @ basis_cart
    basis_frac = basis_frac.reshape((3, n_atom, n_basis))
    basis_frac = basis_frac.transpose((1, 0, 2)).reshape(-1, n_basis)
    basis_frac, _, _ = np.linalg.svd(basis_frac, full_matrices=False)
    return basis_frac


def _construct_basis_cell(
    cell: SymfcAtoms, verbose: bool = False
) -> tuple[NDArray, SymfcAtoms]:
    """Generate a basis set for basis vectors.

    basis (row): In the order of ax, bx, cx, ay, by, cy, az, bz, cz

    """
    cell_copy = _standardize_cell(cell)
    spg_info = _get_symmetry_dataset(cell_copy)
    assert spg_info is not None
    spg_num = spg_info.number
    if verbose:
        if len(cell_copy.numbers) != len(cell.numbers):
            print("Number of atoms changed by standardization.")
        print("Space group:", spg_info.international, spg_num)
    if spg_num >= 195:
        if verbose:
            print("Crystal system: Cubic")
        basis = np.zeros((9, 1))
        basis[:, 0] = _normalize_vector([1, 0, 0, 0, 1, 0, 0, 0, 1])
    elif spg_num >= 168 and spg_num <= 194:
        if verbose:
            print("Crystal system: Hexagonal")
        basis = np.zeros((9, 2))
        basis[:, 0] = _normalize_vector([1, -0.5, 0, 0, np.sqrt(3) / 2, 0, 0, 0, 0])
        basis[8, 1] = 1.0
    elif spg_num >= 143 and spg_num <= 167:
        if "P" in spg_info["international"]:
            if verbose:
                print("Crystal system: Trigonal (Hexagonal)")
            basis = np.zeros((9, 2))
            basis[:, 0] = _normalize_vector([1, -0.5, 0, 0, np.sqrt(3) / 2, 0, 0, 0, 0])
            basis[8, 1] = 1.0
        else:
            if verbose:
                print("Crystal system: Trigonal (Rhombohedral)")
            basis = np.zeros((9, 2))
            basis[:, 0] = _normalize_vector([1, -0.5, 0, 0, np.sqrt(3) / 2, 0, 0, 0, 0])
            basis[8, 1] = 1.0
    elif spg_num >= 75 and spg_num <= 142:
        if verbose:
            print("Crystal system: Tetragonal")
        basis = np.zeros((9, 2))
        basis[:, 0] = _normalize_vector([1, 0, 0, 0, 1, 0, 0, 0, 0])
        basis[8, 1] = 1.0
    elif spg_num >= 16 and spg_num <= 74:
        if verbose:
            print("Crystal system: Orthorhombic")
        basis = np.zeros((9, 3))
        basis[0, 0] = 1.0
        basis[4, 1] = 1.0
        basis[8, 2] = 1.0
    elif spg_num >= 3 and spg_num <= 15:
        if verbose:
            print("Crystal system: Monoclinic")
        basis = np.zeros((9, 4))
        basis[0, 0] = 1.0
        basis[4, 1] = 1.0
        basis[8, 2] = 1.0
        basis[2, 3] = 1.0
    else:
        if verbose:
            print("Crystal system: Triclinic")
        basis = np.eye(9)
    return basis, cell_copy


def _normalize_vector(vec: ArrayLike) -> NDArray:
    """Normalize a vector."""
    _vec = np.array(vec)
    return _vec / np.linalg.norm(_vec)


def _standardize_cell(cell: SymfcAtoms) -> SymfcAtoms:
    """Standardize cell for constructing cell basis."""
    lattice, scaled_positions, numbers = spglib.standardize_cell(  # type: ignore
        cell.totuple(),  # type: ignore
        to_primitive=False,
    )

    n_atoms, scaled_positions_reorder, numbers_reorder = [], [], []
    for i in sorted(set(numbers)):
        ids = np.where(numbers == i)[0]
        n_atoms.append(len(ids))
        scaled_positions_reorder += scaled_positions[ids].tolist()  # type: ignore
        numbers_reorder += numbers[ids].tolist()  # type: ignore

    cell_standardized = SymfcAtoms(
        numbers=numbers_reorder, cell=lattice, scaled_positions=scaled_positions_reorder
    )
    return cell_standardized


def _get_symmetry_dataset(cell: SymfcAtoms) -> spglib.SpglibDataset | None:  # type: ignore
    """Return symmetry dataset."""
    spg_info = spglib.get_symmetry_dataset(cell.totuple())  # type: ignore
    assert spg_info is not None
    return spg_info


class PropertyCalculator(ABC):
    """Calculator class to compute properties."""

    def __init__(self):
        """Init method."""
        self._energy: float
        self._force: NDArray
        self._stress: NDArray

    @abstractmethod
    def eval(self, cell: SymfcAtoms):
        """Evaluate property."""
        pass

    @property
    def energy(self) -> float:
        """Return energy in a specific energy unit."""
        return self._energy

    @property
    def force(self) -> NDArray:
        """Return forces in a specific energy unit.

        shape=(3, len(cell))

        """
        return self._force

    @property
    def stress(self) -> NDArray:
        """Return stress in a specific energy unit.

        shape=(6,)

        """
        return self._stress

    @property
    def GPa_to_energy(self) -> float:
        """Return conversion factor from GPa to energy unit."""
        EVtoGPa = 160.21766208
        return 1.0 / EVtoGPa


class GeometryOptimization:
    """Class for geometry optimization."""

    def __init__(
        self,
        cell: SymfcAtoms,
        prop: PropertyCalculator,
        relax_cell: bool = False,
        relax_volume: bool = False,
        relax_positions: bool = True,
        with_symmetry: bool = True,
        pressure: float = 0.0,
        verbose: bool = False,
    ):
        """Init method.

        Parameters
        ----------
        cell: Initial structure.
        relax_cell: Optimize cell shape.
        relax_volume: Optimize volume.
        relax_positions: Optimize atomic positions.
        with_symmetry: Consider symmetric properties.
        pressure: Pressure in GPa.

        Any one of pot, (params, coeffs), and properties is needed.
        """
        if not relax_cell and not relax_volume and not relax_positions:
            raise ValueError("No degree of freedom to be optimized.")

        self._prop = prop

        self._relax_cell = relax_cell
        self._relax_volume = relax_volume
        self._relax_positions = relax_positions
        self._with_sym = with_symmetry
        self._pressure = pressure * prop.GPa_to_energy
        self._verbose = verbose

        self._structure: SymfcAtoms
        self._basis_axis: NDArray | None
        self._positions_f0: NDArray
        self._set_basis_axis(cell)
        self._basis_frac: NDArray | None
        self._set_basis_positions(cell)

        if not self._relax_cell and not self._relax_volume:
            if not self._relax_positions:
                raise ValueError("No degree of freedom to be optimized.")

        self._x0: NDArray
        self._size_pos: int
        self._set_initial_coefficients()

        if not relax_volume:
            self._v0 = np.linalg.det(self._structure.cell)

        self._energy: float
        self._force: NDArray
        self._stress: NDArray
        self._res: Any
        self._n_atom = len(self._structure)

    @property
    def relax_cell(self) -> bool:
        """Return whether cell shape is optimized or not."""
        return self._relax_cell

    @property
    def relax_volume(self) -> bool:
        """Return whether volume is optimized or not."""
        return self._relax_volume

    @property
    def relax_positions(self) -> bool:
        """Return whether atomic positions are optimized or not."""
        return self._relax_positions

    @property
    def structure(self):
        """Return optimized structure."""
        return self._structure

    @structure.setter
    def structure(self, st: SymfcAtoms):
        self._structure = _refine_positions(st)

    @property
    def energy(self):
        """Return energy at final iteration."""
        return self._res.fun  # type: ignore

    @property
    def n_iter(self):
        """Return number of iterations."""
        return self._res.nit  # type: ignore

    @property
    def success(self):
        """Return whether optimization is successful or not."""
        if self._res is None:
            return False
        return self._res.success

    @property
    def residual_forces(self):
        """Return residual forces and stress represented in basis sets."""
        if self._relax_cell or self._relax_volume:
            residual_f = -self._res.jac[: self._size_pos]  # type: ignore
            residual_s = -self._res.jac[self._size_pos :]  # type: ignore
            return residual_f, residual_s
        return -self._res.jac  # type: ignore

    def _set_basis_axis(self, cell: SymfcAtoms):
        """Set basis vectors for axis components."""
        if self._relax_cell:
            if self._with_sym:
                self._basis_axis, cell_update = _construct_basis_cell(
                    cell,
                    verbose=self._verbose,
                )
            else:
                self._basis_axis = np.eye(9)
                cell_update = cell
        else:
            self._basis_axis = None
            cell_update = cell
        self._structure = _refine_positions(cell_update)
        self._positions_f0 = self._structure.scaled_positions

    def _set_basis_positions(self, cell: SymfcAtoms):
        """Set basis vectors for atomic positions."""
        if self._relax_positions:
            if self._with_sym:
                self._basis_frac = _construct_basis_fractional_coordinates(cell)
                if self._basis_frac is None:
                    self._relax_positions = False
            else:
                N3 = cell.scaled_positions.shape[0] * cell.scaled_positions.shape[1]
                self._basis_frac = np.eye(N3)
        else:
            self._basis_frac = None

    def _set_initial_coefficients(self):
        """Set initial coefficients representing structure."""
        xf, xs = [], []
        if self._relax_positions:
            assert self._basis_frac is not None
            xf = np.zeros(self._basis_frac.shape[1])
        if self._relax_cell:
            xs = self._structure.cell.ravel() @ self._basis_axis
        elif self._relax_volume and not self.relax_cell:
            xs = [np.linalg.det(self._structure.cell)]

        self._x0 = np.concatenate([xf, xs], 0)
        self._size_pos = 0 if self._basis_frac is None else self._basis_frac.shape[1]

    def _split(self, x: NDArray):
        """Split coefficients."""
        partition1 = self._size_pos
        x_pos = x[:partition1]
        x_axis = x[partition1:]
        return x_pos, x_axis

    def function_fix_cell(self, x: NDArray, args: tuple = ()) -> ArrayLike:
        """Target function when performing no cell optimization.

        args is a dummy variable for scipy.optimize.minimize.

        """
        structure = self._to_structure_fix_cell(x)
        self._prop.eval(structure)
        energy = self._prop.energy

        if energy < -1e3 * self._n_atom:
            print("Energy =", energy)
            print("Axis :")
            print(structure.cell)
            print("Fractional coordinates:")
            print(structure.scaled_positions)
            raise ValueError(
                "Geometry optimization failed: Huge negative energy value."
            )

        energy += self._pressure * np.linalg.det(structure.cell)
        return energy

    def jacobian_fix_cell(self, x: NDArray, args: tuple = ()) -> ArrayLike:
        """Target Jacobian function when performing no cell optimization.

        x and args are a dummy variable for scipy.optimize.minimize.

        """
        if self._basis_frac is not None:
            structure = self._to_structure_fix_cell(x)
            prod = -(structure.cell @ self._prop.force).T
            derivatives = self._basis_frac.T @ prod.ravel()
            return derivatives
        return []

    def function_relax_cell(self, x: NDArray, args: tuple = ()) -> ArrayLike:
        """Target function when performing cell optimization.

        args is a dummy variable for scipy.optimize.minimize.

        """
        structure = self._to_structure_relax_cell(x)
        self._prop.eval(structure)
        energy = self._prop.energy

        if (
            energy < -1e3 * self._n_atom
            or abs(np.linalg.det(structure.cell)) / self._n_atom > 1000
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

        energy += self._pressure * np.linalg.det(structure.cell)
        return energy

    def jacobian_relax_cell(self, x: NDArray, args: tuple = ()) -> ArrayLike:
        """Target Jacobian function when performing cell optimization.

        args is a dummy variable for scipy.optimize.minimize.

        """
        partition1 = self._size_pos
        derivatives = np.zeros(len(x))
        if self._relax_positions:
            derivatives[:partition1] = self.jacobian_fix_cell(x[:partition1])
        structure = self._to_structure_relax_cell(x)
        derivatives[partition1:] = self._derivatives_by_axis(
            structure, self._prop.stress
        )
        return derivatives

    def _update_positions(
        self, positions_frac: NDArray, structure: SymfcAtoms
    ) -> SymfcAtoms:
        """Update atomic positions."""
        return _refine_positions(
            SymfcAtoms(
                cell=structure.cell,
                numbers=structure.numbers,
                scaled_positions=positions_frac,
            )
        )

    def _to_structure_fix_cell(self, x: NDArray) -> SymfcAtoms:
        """Convert x to structure."""
        if self._basis_frac is not None:
            disps_f = (self._basis_frac @ x).reshape(-1, 3)
            return self._update_positions(self._positions_f0 + disps_f, self._structure)
        return self._structure

    def _to_structure_relax_cell(self, x: NDArray) -> SymfcAtoms:
        """Convert x to structure."""
        x_positions, x_cells = self._split(x)
        if self._relax_cell:
            axis = self._basis_axis @ x_cells
            cell = axis.reshape((3, 3)).T
        else:
            x_cells0 = self._split(self._x0)[1][0]
            scale = (x_cells[0] / x_cells0) ** (1 / 3)
            cell = self._structure.cell * scale

        structure = SymfcAtoms(
            cell=cell,
            numbers=self._structure.numbers,
            scaled_positions=self._structure.scaled_positions,
        )

        if self._relax_positions:
            return self._to_structure_fix_cell(x_positions)

        return structure

    def _to_volume(self, x: NDArray) -> float:
        _, x_cells = self._split(x)
        axis = self._basis_axis @ x_cells
        return np.linalg.det(axis.reshape((3, 3)))

    def _derivatives_by_axis(
        self, structure: SymfcAtoms, stress: NDArray
    ) -> NDArray | float:
        """Compute derivatives with respect to axis elements.

        PV @ axis_inv.T is exactly the same as the derivatives of PV term
        with respect to axis components.

        Under the constraint of a fixed cell shape, the mean normal stress
        serves as an approximation to the derivative of the enthalpy
        with respect to volume.
        """
        pv = self._pressure * np.linalg.det(structure.cell)
        sigma = [
            [stress[0] - pv, stress[3], stress[5]],
            [stress[3], stress[1] - pv, stress[4]],
            [stress[5], stress[4], stress[2] - pv],
        ]
        if self._relax_cell:
            """derivatives_s: In the order of ax, bx, cx, ay, by, cy, az, bz, cz"""
            derivatives_s = -np.array(sigma) @ np.linalg.inv(structure.cell)
            assert self._basis_axis is not None
            derivatives_s = self._basis_axis.T @ derivatives_s.ravel()
        else:
            derivatives_s = -np.trace(np.array(sigma)) / 3

        return derivatives_s

    def run(
        self,
        method: Literal["BFGS", "CG", "L-BFGS-B", "SLSQP"] = "BFGS",
        gtol: float = 1e-4,
        maxiter: int = 1000,
        c1: float | None = None,
        c2: float | None = None,
    ):
        """Run geometry optimization.

        Parameters
        ----------
        method: Optimization method, CG, BFGS, L-BFGS-B or SLSQP.
                If relax_volume = False, SLSQP is automatically used.
        gtol: Tolerance for gradients.
        maxiter: Maximum iteration in scipy optimization.
        c1: c1 parameter in scipy optimization. c1=1e-4 is the default in scipy.
        c2: c2 parameter in scipy optimization. c2=0.0 is the default in scipy.
        """
        if self._relax_cell and not self._relax_volume:
            method = "SLSQP"

        if self._verbose:
            print("Using", method, "method", flush=True)
            print("Relax cell shape:       ", self._relax_cell, flush=True)
            print("Relax volume:           ", self._relax_volume, flush=True)
            print("Relax atomic positionss:", self._relax_positions, flush=True)

        if method == "SLSQP":
            options = {"ftol": gtol, "disp": self._verbose}
        else:
            options = {"gtol": gtol, "disp": self._verbose}
            if maxiter is not None:
                options["maxiter"] = maxiter
            if c1 is not None:
                options["c1"] = c1
            if c2 is not None:
                options["c2"] = c2

        if self._relax_cell or self._relax_volume:
            fun = self.function_relax_cell
            jac = self.jacobian_relax_cell
        else:
            fun = self.function_fix_cell
            jac = self.jacobian_fix_cell

        if self._verbose:
            print("Number of degrees of freedom:", len(self._x0), flush=True)

        if self._relax_cell and not self._relax_volume:
            nlc = NonlinearConstraint(
                self._to_volume,
                self._v0 - 1e-15,
                self._v0 + 1e-15,
                jac="2-point",
            )
            self._res = minimize(
                fun,
                self._x0,
                method=method,
                jac=jac,
                options=options,
                constraints=[nlc],
            )
        else:
            self._res = minimize(fun, self._x0, method=method, jac=jac, options=options)

        self._x0 = self._res.x

        if self._relax_cell or self._relax_volume:
            self._structure = self._to_structure_relax_cell(self._res.x)
        else:
            self._structure = self._to_structure_fix_cell(self._res.x)

        return self
