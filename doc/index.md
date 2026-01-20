# Symgo

Symgo is a Python-based geometry optimization code under constraint of crystal
symmetry. Currently pypolymlp and mattersim interfaces are provided.

## Algorithm

Crystal symmetry is used to determine the degrees of freedom for atomic
positions and the basis vectors of the crystal structure. The crystal geometry
optimization is performed with respect to these degrees of freedom; that is,
structural changes are represented as linear combinations of the basis vectors
spanning these degrees of freedom.

## Usage

```{note}
The usage can change in the future.
```

The `GeometryOptimization` class is the core of this code. The first and second
arguments are the crystal structure and
crystal structure is represented by a `SymfcAtoms` class instance, and the
latter should have `energy`, `force`, and `stress` attributes as well as an
`eval` method that takes a `SymfcAtoms` instance and stores the energy, force,
and stress in it. The following is an example using pypolymlp as the force
calculator.

```python
from symfc.utils.utils import SymfcAtoms
from symgo.interface.pypolymlp import PypolymlpPropertyCalculator
from symgo.optimization import GeometryOptimization

def relax(
    cell: SymfcAtoms, mlp: PypolymlpPropertyCalculator
) -> SymfcAtoms:
    try:
        go = GeometryOptimization(cell, mlp, verbose=True)
        go.run()
    except ValueError as e:
        if "Geometry optimization failed" in str(e):
            print("Geometry optimization by BFGS failed. Try TNC.")
            go = GeometryOptimization(cell, mlp, verbose=True)
            go.run(method="TNC", xtol=1e-10, ftol=1e-10)
        else:
            raise ValueError(e)

    if not go.relaxed:
        print(
            "Geometry optimization was not performed because of no degree of freedoms."
        )
    return go.structure
```


## History

The initial code was forked from `optimization.py` in pypolymlp (see
{ref}`changelog_0_1_0`).

## License

BSD-3-Clause.

## Contributors

- {user}`Atsuto Seko <sekocha>` (Kyoto university)
- {user}`Atsushi Togo <atztogo>` (National Institute for Materials Science)

```{toctree}
:hidden:
install
changelog
```
