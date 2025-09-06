from numpy.typing import NDArray
from symfc.utils.utils import SymfcAtoms

from symgo.optimization import PropertyCalculator, print_structure


class MattersimPropertyCalculator(PropertyCalculator):
    """Property class using pypolymlp."""

    def __init__(self, cell: SymfcAtoms, verbose: bool = False):
        """Init method."""
        from ase import Atoms
        from mattersim.forcefield import MatterSimCalculator

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
