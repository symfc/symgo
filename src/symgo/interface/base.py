from __future__ import annotations

from abc import ABC, abstractmethod

from numpy.typing import NDArray
from symfc.utils.utils import SymfcAtoms


def print_structure(structure: SymfcAtoms):
    """Print structure."""
    print("basis vectors:")
    for a in structure.cell:
        print(f" - {a}")
    print("Fractional coordinates:")
    for p, e in zip(structure.scaled_positions, structure.numbers):
        print(f" - {e} {p}")


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
