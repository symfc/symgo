import os
import pathlib

import numpy as np
import phonopy
from numpy.typing import NDArray
from phonopy.structure.atoms import PhonopyAtoms
from pypolymlp.calculator.properties import Properties
from pypolymlp.mlp_dev.pypolymlp import Pypolymlp
from pypolymlp.utils.phonopy_utils import phonopy_cell_to_structure
from symfc.utils.utils import SymfcAtoms

from symgo.optimization import GeometryOptimization, Property

cwd = pathlib.Path(__file__).parent


class PypolymlpProperty(Property):
    """Property class using pypolymlp."""

    def __init__(self, polymlp_filename: str | os.PathLike = "polymlp.yaml"):
        """Init method."""
        mlp = Pypolymlp()
        mlp.load_mlp(polymlp_filename)
        self._prop = Properties(params=mlp.parameters, coeffs=mlp.coeffs)

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
        print(phpy_cell)
        self._energy, self._force, self._stress = self._prop.eval(
            phonopy_cell_to_structure(phpy_cell)
        )

    @property
    def energy(self) -> float:
        """Return energy in a specific energy unit."""
        return self._energy

    @property
    def force(self) -> NDArray:
        """Return forces in a specific energy unit."""
        return self._force

    @property
    def stress(self) -> NDArray:
        """Return stresses in a specific energy unit."""
        return self._stress

    @property
    def GPa_to_energy(self) -> float:
        """Return conversion factor from GPa to energy unit."""
        return 1.0 / 160.21766208


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
    prop = PypolymlpProperty(cwd / "polymlp.yaml")
    go = GeometryOptimization(cell, prop, verbose=True)
    go.run()
