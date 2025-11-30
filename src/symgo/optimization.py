"""Class for geometry optimization with symmetric constraint."""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import spglib
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import NonlinearConstraint, OptimizeResult, minimize
from symfc.basis_sets import FCBasisSetO1
from symfc.utils.utils import SymfcAtoms

from symgo.interface.base import PropertyCalculator
from symgo.minimize import (
    MinimizeFunctionParams,
    function_fix_cell,
    function_relax_cell,
    jacobian_fix_cell,
    jacobian_relax_cell,
    refine_positions,
    to_structure,
    to_volume,
)


def _construct_basis_fractional_coordinates(cell: SymfcAtoms) -> NDArray | None:
    """Generate a basis set for atomic positions in fractional coordinates."""
    try:
        fc_basis = FCBasisSetO1(cell).run()
    except ValueError:
        return None
    assert fc_basis.full_basis_set is not None
    basis_cart = fc_basis.full_basis_set.toarray()
    if basis_cart is None or basis_cart.size == 0:
        return None
    basis_f = _basis_cartesian_to_fractional_coordinates(basis_cart, cell)
    return basis_f


def _basis_cartesian_to_fractional_coordinates(
    basis_cart: NDArray, cell: SymfcAtoms
) -> NDArray:
    """Convert basis set in Cartesian coord. to basis set in fractional coordinates."""
    n_basis = basis_cart.shape[1]
    n_atom = len(cell)
    inv_lattice = np.linalg.inv(cell.cell)

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
) -> tuple[NDArray, NDArray]:
    """Generate a basis set for basis vectors.

    basis (row): In the order of ax, bx, cx, ay, by, cy, az, bz, cz

    """
    spg_info = spglib.get_symmetry_dataset(cell.totuple())  # type: ignore
    assert spg_info is not None
    ptg_symbol, ptg_number, tmat = spglib.get_pointgroup(spg_info.rotations)  # type: ignore
    print(f"Point group of input lattice: {ptg_symbol}")
    if ptg_number >= 28:
        if verbose:
            print("Crystal system: Cubic")
        basis = np.zeros((9, 1))
        basis[:, 0] = _normalize_vector([1, 0, 0, 0, 1, 0, 0, 0, 1])
    elif 16 <= ptg_number <= 27:
        if verbose:
            print("Crystal system: Hexagonal")
        basis = np.zeros((9, 2))
        basis[:, 0] = _normalize_vector([1, -0.5, 0, 0, np.sqrt(3) / 2, 0, 0, 0, 0])
        basis[8, 1] = 1.0
    elif 9 <= ptg_number <= 15:
        if verbose:
            print("Crystal system: Tetragonal")
        basis = np.zeros((9, 2))
        basis[:, 0] = _normalize_vector([1, 0, 0, 0, 1, 0, 0, 0, 0])
        basis[8, 1] = 1.0
    elif 6 <= ptg_number <= 8:
        if verbose:
            print("Crystal system: Orthorhombic")
        basis = np.zeros((9, 3))
        basis[0, 0] = 1.0
        basis[4, 1] = 1.0
        basis[8, 2] = 1.0
    elif 3 <= ptg_number <= 5:
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
    return basis, np.array(tmat)


def _normalize_vector(vec: ArrayLike) -> NDArray:
    """Normalize a vector."""
    _vec = np.array(vec)
    return _vec / np.linalg.norm(_vec)


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
        self._prop = prop

        self._relax_cell = relax_cell
        self._relax_volume = relax_volume
        self._relax_positions = relax_positions
        self._with_symmetry_constraints = with_symmetry
        self._pressure = pressure * prop.GPa_to_energy
        self._verbose = verbose
        self._structure: SymfcAtoms
        self._basis_axis: NDArray | None
        self._transformation_matrix: NDArray
        self._set_basis_axis(cell)
        self._basis_frac: NDArray | None
        self._set_basis_positions()
        self._res: OptimizeResult | None = None

        if (
            not self._relax_cell
            and not self._relax_volume
            and not self._relax_positions
        ):
            warnings.warn(
                "No degree of freedom to be optimized.", UserWarning, stacklevel=2
            )
            self._params = None
        else:
            self._x0: NDArray
            self._size_pos: int
            self._set_initial_coefficients()
            self._params = MinimizeFunctionParams(
                prop=self._prop,
                size_pos=self._size_pos,
                orig_x=self._x0,
                orig_structure=self._structure,
                basis_axis=self._basis_axis,
                position_relaxation=self._relax_positions,
                lattice_relaxation=self._relax_cell,
                volume_relaxation=self._relax_volume,
                basis_frac=self._basis_frac,
                pressure=self._pressure,
            )

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
    def structure(self) -> SymfcAtoms:
        """Return optimized structure."""
        lattice = np.linalg.inv(self._transformation_matrix).T @ self._structure.cell
        return SymfcAtoms(
            cell=lattice,
            numbers=self._structure.numbers,
            scaled_positions=self._structure.scaled_positions,
        )

    @property
    def relaxed(self) -> bool:
        """Return whether optimization has been run or not."""
        return self._res is not None

    @property
    def energy(self) -> NDArray:
        """Return energy at final iteration."""
        if self._res is None:
            raise ValueError("Optimization has not been run yet.")
        return self._res.fun  # type: ignore

    @property
    def n_iter(self) -> int:
        """Return number of iterations."""
        if self._res is None:
            raise ValueError("Optimization has not been run yet.")
        return self._res.nit  # type: ignore

    @property
    def success(self) -> bool:
        """Return whether optimization is successful or not."""
        if self._res is None:
            raise ValueError("Optimization has not been run yet.")
        return self._res.success

    @property
    def residual_forces(self) -> NDArray | tuple[NDArray, NDArray]:
        """Return residual forces and stress represented in basis sets."""
        if self._res is None:
            raise ValueError("Optimization has not been run yet.")

        if self._relax_cell or self._relax_volume:
            residual_f = -self._res.jac[: self._size_pos]  # type: ignore
            residual_s = -self._res.jac[self._size_pos :]  # type: ignore
            return residual_f, residual_s
        return -self._res.jac  # type: ignore

    def _set_basis_axis(self, cell: SymfcAtoms):
        """Set basis vectors for axis components."""
        self._transformation_matrix = np.eye(3)
        if self._relax_cell:
            if self._with_symmetry_constraints:
                self._basis_axis, self._transformation_matrix = _construct_basis_cell(
                    cell,
                    verbose=self._verbose,
                )
            else:
                self._basis_axis = np.eye(9)
        else:
            self._basis_axis = None

        self._structure = SymfcAtoms(
            cell=self._transformation_matrix.T @ cell.cell,
            numbers=cell.numbers,
            scaled_positions=refine_positions(cell.scaled_positions),
        )

    def _set_basis_positions(self):
        """Set basis vectors for atomic positions."""
        if self._relax_positions:
            if self._with_symmetry_constraints:
                self._basis_frac = _construct_basis_fractional_coordinates(
                    self._structure
                )
                if self._basis_frac is None:
                    self._relax_positions = False
            else:
                N3 = int(np.prod(self._structure.scaled_positions.shape))
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

    def run(
        self,
        method: Literal["BFGS", "CG", "TNC", "L-BFGS-B", "SLSQP"] = "BFGS",
        gtol: float = 1e-4,
        xtol: float | None = None,
        ftol: float | None = None,
        maxiter: int = 1000,
        c1: float | None = None,
        c2: float | None = None,
    ) -> GeometryOptimization:
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
        if self._params is None:
            return self

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
            if xtol is not None:
                options["xtol"] = xtol
            if ftol is not None:
                options["ftol"] = ftol

        if self._relax_cell or self._relax_volume:
            fun = function_relax_cell
            jac = jacobian_relax_cell
        else:
            fun = function_fix_cell
            jac = jacobian_fix_cell

        if self._verbose:
            print("Number of degrees of freedom:", len(self._x0), flush=True)

        if self._relax_cell and not self._relax_volume:
            v0 = np.linalg.det(self._structure.cell)
            nlc = NonlinearConstraint(
                to_volume(self._params.size_pos, self._params.basis_axis),
                v0 - 1e-15,
                v0 + 1e-15,
                jac="2-point",
            )
            self._res = minimize(
                fun,
                self._x0,
                (self._params,),
                method=method,
                jac=jac,
                options=options,
                constraints=[nlc],
            )
        else:
            self._res = minimize(
                fun, self._x0, (self._params,), method=method, jac=jac, options=options
            )

        self._structure = to_structure(
            self._res.x,
            self._size_pos,
            self._x0,
            self._structure,
            self._basis_axis,
            self._relax_cell,
            self._relax_volume,
            self._basis_frac,
        )

        return self
