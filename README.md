# Iterative block solvers for PorePy using PETSc
This repository implements linear solvers for the open-source software
[PorePy](https://github.com/pmgbergen/porepy), using [PETSc](https://petsc.org/) as the
linear algebra backend.

This code is under active development and cannot be considered stable. Use with care.

The code contained herein is based on initial work by Yury Zabegaev in 
[this repository](https://github.com/yuriyzabegaev/FTHM-Solver/).
Whereas the upstream repository is used for prototyping of solvers and production of
papers, this repository aims to make the solvers robust and easily applicable in
general PorePy models.

# Installation
This package can be installed with

    pip install -e .

It is assumed that working installations of PorePy and PETSc are available.


