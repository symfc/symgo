import phonopy
from phonopy.interface.pypolymlp import relax_atomic_positions

ph = phonopy.load("phonopy_cells.yaml")
ph.load_mlp()
ph.generate_displacements(distance=0.1)
relax_atomic_positions(ph.supercells_with_displacements[0], ph.mlp.mlp, verbose=True)
