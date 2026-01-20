(instasll)=
# Installation

PyPI package is available.

- https://pypi.org/project/symgo/

## Requirement

- symfc
- scipy
- numpy
- spglib

## Installation from source code

A simplest installation using conda-forge packages:

```bash
% conda create -n symgo -c conda-forge
% conda activate symgo
% conda install -c conda-forge numpy scipy spglib
% git clone https://github.com/symfc/symgo.git
% cd symgo
% pip install -e .
```
