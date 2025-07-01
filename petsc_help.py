import sys
import petsc4py
import scipy.sparse
import numpy as np


# args = '-help'
# args = '-pc_type hypre -pc_hypre_type pilut -help'
# args = '-pc_type hypre -pc_hypre_type boomeramg -help'
args = "-help"
# args = '-pc_type bjacobi -sub_pc_type ilu -sub_pc_factor_levels 0 -sub_ksp_type preonly'
# args = '-pc_type gamg -pc_gamg_threshold 0.01 -mg_levels_ksp_max_it 5 -pc_gamg_agg_nsmooths 1'

petsc4py.init(args)

from petsc4py import PETSc

ksp = PETSc.KSP().create()

mat = scipy.sparse.csr_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

petsc_mat = PETSc.Mat().createAIJ(
    size=mat.shape,
    csr=(mat.indptr, mat.indices, mat.data),
    bsize=1,
)

ksp.setFromOptions()
ksp.setOperators(petsc_mat)
ksp.setUp()
ksp.view()

pc = ksp.getPC()

pc.setType("fieldsplit")
is_0 = PETSc.IS().createGeneral([0, 1])
is_1 = PETSc.IS().createGeneral([2])

pc.setFieldSplitIS(("field_0", is_0), ("field_1", is_1))
pc.setUp()

pc_0 = pc.getFieldSplitSubKSP()[0].getPC()
pc_0.setType("sor")
pc_1 = pc.getFieldSplitSubKSP()[1].getPC()


pc_1.setType("composite")
pc_1.setCompositeType(PETSc.PC.CompositeType.MULTIPLICATIVE)
pc_1.setUp()
pc_1.addCompositePCType(PETSc.PC.Type.ILU)
pc_sub_0 = pc_1.getCompositePC(0)


# pc_1.addCompositePCType(PETSc.PC.Type.SOR)
# pc_sub_1 = pc_1.getCompositePC(1)
# # pc_sub_1.setType("sor")
# pc_1.setUp()
pc_sub_0.setOperators(*pc_1.getOperators())
pc_sub_0.setUp()

# pc_sub_1.setOperators(*pc_1.getOperators())
# pc_sub_1.setUp()

x = PETSc.Vec().createSeq(3)
x.set(1.0)
b = PETSc.Vec().createSeq(3)
b.set(1.0)

ksp.setUp()

ksp.solve(b, x)
print("Solution:", x.array)

debug = []
