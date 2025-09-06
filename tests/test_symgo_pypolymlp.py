import pathlib

import phonopy
from symfc.utils.utils import SymfcAtoms

from symgo.interface.pypolymlp import PypolymlpPropertyCalculator
from symgo.optimization import GeometryOptimization

cwd = pathlib.Path(__file__).parent


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
