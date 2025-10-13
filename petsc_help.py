"""This script serves a quick access to the PETSc command-line arguments reference.

Usage: edit the `args` variable with the options you are interested in. Run with python.
"""

import petsc4py


args = "-help -pc_type hypre -pc_hypre_type boomeramg"
# Some more examples:
# args = '-help -pc_type gamg'
# args = '-help -ksp_type fgmres'

petsc4py.init(args)

from petsc4py import PETSc

ksp = PETSc.KSP().create()
ksp.setFromOptions()

pc = ksp.getPC()
pc.setFromOptions()
