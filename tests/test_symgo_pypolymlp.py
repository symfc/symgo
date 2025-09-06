import os
import pathlib

import phonopy
from numpy.typing import NDArray
from phonopy.structure.atoms import PhonopyAtoms
from pypolymlp.calculator.properties import Properties
from pypolymlp.mlp_dev.pypolymlp import Pypolymlp
from pypolymlp.utils.phonopy_utils import phonopy_cell_to_structure
from symfc.utils.utils import SymfcAtoms

from symgo.optimization import GeometryOptimization, PropertyCalculator, print_structure

cwd = pathlib.Path(__file__).parent


class PypolymlpPropertyCalculator(PropertyCalculator):
    """Property class using pypolymlp."""

    def __init__(
        self,
        polymlp_filename: str | os.PathLike = "polymlp.yaml",
        verbose: bool = False,
    ):
        """Init method."""
        self._verbose = verbose
        mlp = Pypolymlp()
        mlp.load_mlp(str(polymlp_filename))
        self._prop = Properties(params=mlp.parameters, coeffs=mlp.coeffs)  # type: ignore

        self._energy: float
        self._force: NDArray
        self._stress: NDArray

    def eval(self, cell: SymfcAtoms):
        """Evaluate property."""
        phpy_cell = PhonopyAtoms(
            cell=cell.cell,
            scaled_positions=cell.scaled_positions,
            numbers=cell.numbers,
        )
        self._energy, self._force, self._stress = self._prop.eval(
            phonopy_cell_to_structure(phpy_cell)
        )
        self._energy = float(self._energy)

        if self._verbose:
            print("Energy:", self._energy)
            print("Forces:\n", self._force.T)
            print("Stress:\n", self._stress)
            print_structure(cell)


def test_symgo_relax_positions():
    """Test relax positions."""
    ph = phonopy.load(cwd / "phonopy_cells.yaml")
    ph.generate_displacements(distance=0.1)
    assert ph.supercells_with_displacements is not None
    scell = ph.supercells_with_displacements[0]
    cell = SymfcAtoms(
        cell=scell.cell,
        numbers=scell.numbers,
        scaled_positions=scell.scaled_positions,
    )
    prop = PypolymlpPropertyCalculator(cwd / "polymlp.yaml", verbose=True)
    go = GeometryOptimization(cell, prop, verbose=True)
    go.run()
