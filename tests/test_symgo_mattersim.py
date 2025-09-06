import pathlib

import numpy as np
import phonopy
from symfc.utils.utils import SymfcAtoms

from symgo.interface.mattersim import MattersimPropertyCalculator
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
    prop = MattersimPropertyCalculator(cell)
    go = GeometryOptimization(cell, prop, verbose=True)
    go.run(gtol=1e-4)
    print(go.structure.scaled_positions[0])
    np.testing.assert_allclose(
        go.structure.scaled_positions[0], [0.99858583, 0.0, 0.0], atol=0.001
    )


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
    prop = MattersimPropertyCalculator(cell)
    go = GeometryOptimization(
        cell, prop, relax_cell=True, relax_volume=True, verbose=True
    )
    go.run(gtol=1e-5)
    np.testing.assert_allclose(go.structure.cell, np.eye(3) * 5.70141633)
