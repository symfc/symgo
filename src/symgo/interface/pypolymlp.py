import os

from numpy.typing import NDArray
from phonopy.structure.atoms import PhonopyAtoms
from symfc.utils.utils import SymfcAtoms

from symgo.optimization import PropertyCalculator, print_structure


class PypolymlpPropertyCalculator(PropertyCalculator):
    """Property class using pypolymlp."""

    def __init__(
        self,
        polymlp_filename: str | os.PathLike = "polymlp.yaml",
        verbose: bool = False,
    ):
        """Init method."""
        from pypolymlp.calculator.properties import Properties
        from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

        self._verbose = verbose
        mlp = Pypolymlp()
        mlp.load_mlp(str(polymlp_filename))
        self._prop = Properties(params=mlp.parameters, coeffs=mlp.coeffs)  # type: ignore

        self._energy: float
        self._force: NDArray
        self._stress: NDArray

    def eval(self, cell: SymfcAtoms):
        """Evaluate property."""
        from pypolymlp.utils.phonopy_utils import phonopy_cell_to_structure

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
