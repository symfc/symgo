import pathlib

import numpy as np
import phonopy
from ase import Atoms
from mattersim.forcefield import MatterSimCalculator
from numpy.typing import NDArray
from symfc.utils.utils import SymfcAtoms

from symgo.optimization import GeometryOptimization, Property, print_structure

cwd = pathlib.Path(__file__).parent


class MattersimProperty(Property):
    """Property class using pypolymlp."""

    def __init__(self, cell: SymfcAtoms, verbose: bool = False):
        """Init method."""
        self._verbose = verbose
        self._cell = Atoms(
            numbers=cell.numbers,
            scaled_positions=cell.scaled_positions,
            cell=cell.cell,
            pbc=True,
        )
        self._cell.calc = MatterSimCalculator(
            load_path="MatterSim-v1.0.0-5M.pth", device="cpu"
        )

        self._energy: float
        self._force: NDArray
        self._stress: NDArray

    def eval(self, cell: SymfcAtoms):
        """Evaluate property.

        Note
        ----
        Sign convention of stress is chosen to be same as pressure.

        """
        self._cell.set_cell(cell.cell)
        self._cell.set_scaled_positions(cell.scaled_positions)
        self._energy = float(self._cell.get_potential_energy())
        self._force = self._cell.get_forces().T
        self._stress = -self._cell.get_stress()[[0, 1, 2, 5, 3, 4]]

        if self._verbose:
            print("Energy:", self._energy)
            print("Forces:\n", self._force.T)
            print("Stress:\n", self._stress)
            print_structure(cell)

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
    prop = MattersimProperty(cell, verbose=True)
    go = GeometryOptimization(cell, prop, verbose=True)
    go.run()


def test_symgo_relax_cell():
    """Test relax cell."""
    cell = SymfcAtoms(
        cell=np.eye(3) * 5.9,
        numbers=[11] * 4 + [17] * 4,
        scaled_positions=[
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, 0.5000000000000000, 0.5000000000000000],
            [0.5000000000000000, 0.0000000000000000, 0.5000000000000000],
            [0.5000000000000000, 0.5000000000000000, 0.0000000000000000],
            [0.5000000000000000, 0.5000000000000000, 0.5000000000000000],
            [0.5000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, 0.5000000000000000, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, 0.5000000000000000],
        ],
    )
    prop = MattersimProperty(cell, verbose=True)
    go = GeometryOptimization(
        cell, prop, relax_cell=True, relax_volume=True, verbose=True
    )
    go.run(gtol=1e-10)
