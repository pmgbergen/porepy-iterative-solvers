from dataclasses import dataclass
from typing import Optional
from warnings import warn

import porepy as pp
import scipy.sparse as sps
from petsc4py import PETSc

from pp_solvers.block_matrix import BlockMatrixStorage
from pp_solvers.dof_manager import DofManager
from pp_solvers.petsc_utils import clear_petsc_options, csr_to_petsc
from pp_solvers.petsc_solvers import (
    LinearSolverWithTransformations,
    PetscKrylovSolver,
    insert_petsc_options,
)
from .preconditioners import CompositePreconditioner, SinglePhysicsPreconditioner


class MultiPhysicsPreconditioner:
    """Translate a general scheme to a specific PETSc preconditioner, specified as a
    dictionary (really a fully specified petsc options).
    """

    def __init__(
        self,
        components: list[SinglePhysicsPreconditioner],
        dof_manager: DofManager,
        model: pp.PorePyModel,
    ):
        """
        Args:
            groups: List of groups of equations and variables.
            schemes: List of schemes for each group.
        """
        self._single_physics_precond = components
        self._dof_manager = dof_manager
        self._nd = model.nd
        self._model = model

    def configure(
        self,
        bmat: BlockMatrixStorage,
        pc: PETSc.PC,  # PC comes from ksp or similar
        user_options: dict | None = None,
        precond_list: list[SinglePhysicsPreconditioner] | None = None,
        prefix=""
    ) -> dict:  # TODO: Return None?
        """
        Populate the PETSc preconditioner based on the groups and schemes. This entails
        making a bridge from the general settings defined in a scheme to the PETSc
        options needed to apply the scheme to a contrete linear system.

        Args:
            model: The model instance specifying the problem to be solved.
        """
        user_options = user_options if user_options is not None else {}

        if precond_list is None:
            precond_list = self._single_physics_precond

        options = {}
        dof_manager = self._dof_manager

        for counter, single_physics_precond in enumerate(precond_list):
            # Define a scheme for the group
            has_complement = counter < len(precond_list) - 1

            # Generate the actual petsc proconditioner.
            loc_options = single_physics_precond.configure(
                model=self._model,
                dof_manager=self._dof_manager,
                has_complement=has_complement,
                opts=user_options,
            )
            tagged_options = {f"{prefix}{k}": v for k, v in loc_options.items()}

            if isinstance(single_physics_precond, CompositePreconditioner):
                # Set up the composite preconditioner so that we can get hold of the
                # sub-preconditioners and configure them as well.
                insert_petsc_options(tagged_options)
                pc.setFromOptions()
                pc.setUp()
                for i, sub_solver in enumerate(single_physics_precond.solvers):
                    if isinstance(sub_solver, list):
                        pc.addCompositePCType("fieldsplit")
                        sub_pc = pc.getCompositePC(i)
                        sub_pc.setOperators(*pc.getOperators())
                        loc_options = self.configure(
                            bmat,
                            sub_pc,
                            user_options,
                            sub_solver,
                            prefix=f"{prefix}sub_{i}_",
                        )
                    else:
                        loc_options = sub_solver.configure(
                            model=self._model,
                            dof_manager=self._dof_manager,
                            has_complement=has_complement,
                            opts=user_options,
                        )
                        pc.addCompositePCType(loc_options["pc_type"])
                        sub_pc = pc.getCompositePC(i)
                        sub_pc.setOperators(*pc.getOperators())
                    # Implementation note: This is something of a break with how petsc
                    # options are set in the rest of the package: Instead of defining
                    # the option through PETSc.Options(), we use the Python API
                    # directly. This may be possible to avoid, but turned out to solve
                    # an issue with setting up CompositePC, so it will have to do for
                    # now.
                    # Set the matrix for the sub-preconditioner. This seems to be
                    # necessary for composite preconditioners.
                    if loc_options.get("pc_type") == "python":
                        # EK cannot wrap his head around what this would mean, so we
                        # rule it out for now.
                        assert not has_complement
                        python_pc = sub_solver.python_preconditioner(bmat, dof_manager)
                        python_pc.petsc_pc.setOptionsPrefix(f"{prefix}python_")
                        sub_pc.setType("python")
                        sub_pc.setPythonContext(python_pc)

                    tagged_loc_options = {
                        f"{prefix}sub_{i}_{k}": v for k, v in loc_options.items()
                    }

                    insert_petsc_options(tagged_loc_options)
                    sub_pc.setFromOptions()
                    sub_pc.setUp()
                    tagged_options |= tagged_loc_options

            # Get the tag for this group, and prepend it to the options.

            options |= tagged_options
            if not has_complement:
                # If this is the last preconditioner in the list, we can set it up and
                # return the options database.
                insert_petsc_options(options)
                pc.setFromOptions()
                pc.setUp()

                return options
            else:
                # There are more preconditioners to process, using a fieldsplit style
                # preconditioner. We need to parse the fieldsplit options and set up the
                # preconditioners of the next group.
                pc, prefix = self._parse_fieldsplit_pc(
                    precond_list[counter:],
                    bmat,
                    pc,
                    prefix,
                    tagged_options=tagged_options,
                )

        raise ValueError("Should have reached an empty complement")

    def _parse_fieldsplit_pc(
        self,
        precond_list,
        bmat: BlockMatrixStorage,
        pc: PETSc.PC,
        prefix: str,
        tagged_options: dict | None = None,
    ):
        dof_manager = self._dof_manager

        elim_precond = precond_list[0]

        elim_group = dof_manager.blocks_of_solver(elim_precond)
        keep_group = []
        for i in range(1, len(precond_list)):
            keep_group += dof_manager.blocks_of_solver(precond_list[i])

        empty_bmat = bmat.empty_container()[elim_group + keep_group]

        block_size = 1 if elim_precond.unit_block_size else self._nd

        # Get the IS for the group, but only if complement is not None.
        is_elim, is_keep = self._dof_manager.petsc_is(
            elim_precond, precond_list[1:], empty_bmat
        )
        is_elim.setBlockSize(block_size)
        elim_tag = elim_precond.tag
        keep_tag = elim_precond.complement_tag

        insert_petsc_options(tagged_options)
        pc.setFromOptions()
        pc.setFieldSplitIS((elim_tag, is_elim), (keep_tag, is_keep))

        # Invoke the inverter, if any. This is where the fixed-stress approximation
        # for hydromechanical problems is applied. Note to self: Need to send in
        # all remaining groups to the inverter to make sure the returned matrix is
        # correct.
        inverter = elim_precond.inverter(self._model, dof_manager, keep_group)
        if inverter is not None:
            S = pc.getOperators()[1].createSubMatrix(is_keep, is_keep)
            petsc_stab = inverter(bmat)
            S.axpy(1, petsc_stab)
            pc.setFieldSplitSchurPreType(PETSc.PC.FieldSplitSchurPreType.USER, S)

        pc.setUp()

        elim_ksp = pc.getFieldSplitSubKSP()[0]
        elim_pc = elim_ksp.getPC()

        keep_ksp = pc.getFieldSplitSubKSP()[1]
        keep_pc = keep_ksp.getPC()

        if len(keep_pc.getOptionsPrefix()) > 126:
            # PETSc has a limit on the prefix length, which seems to be 127
            # characters. If the prefix is too long, we raise a warning.
            msg = "The prefix for the PETSc preconditioner is too long. "
            msg += "Check the configuration of the preconditioner."
            warn(msg)

        if elim_precond.ksp_keep_use_pmat:
            _, pmat = keep_ksp.getOperators()
            # PETSc encourages using the same matrix for both arguments in the user
            # manual. We explicitly declare using the same operator for KSP and PC.
            keep_ksp.setOperators(pmat, pmat)

        if (near_null_space := elim_precond.near_null_space(self._model)) is not None:
            null_space_vectors = []
            for b in near_null_space:
                null_space_vec_petsc = PETSc.Vec().create()  # possibly mem leak
                null_space_vec_petsc.setSizes(b.shape[0], block_size)
                null_space_vec_petsc.setUp()
                null_space_vec_petsc.setArray(b)
                null_space_vectors.append(null_space_vec_petsc)
            # possibly mem leak
            null_space_petsc = PETSc.NullSpace().create(True, null_space_vectors)
            elim_pc.getOperators()[1].setNearNullSpace(null_space_petsc)

        # Call on self.complement to configure the PETSc PC object for the complement,
        # and update (override) the options with the options returned by the complement.
        # Note that, due to the tagging system, this may override some options that were
        # set above.

        pc = keep_pc
        prefix = f"{prefix}fieldsplit_{keep_tag}_"
        return pc, prefix


@dataclass
class PetscKSPScheme:
    """Scheme for a KSP solver for a multiphysics problem."""

    preconditioner: Optional = None
    """The preconditioner to be used."""

    petsc_options: Optional[dict] = None
    """Additional options to be passed to PETSc."""

    def get_groups(self) -> list[int]:
        """Return the groups of the preconditioner."""
        return self.preconditioner.get_groups()

    def make_solver(self, mat_orig: BlockMatrixStorage, options: dict | None = None):
        # Construct a PETSc matrix from the scipy matrix.
        # TODO: Can we at this point delete the scipy matrix to save memory?
        petsc_mat = csr_to_petsc(mat_orig.mat)

        # Clear the PETSc options from a previous solve.
        clear_petsc_options()

        # Hard coded options for the KSP solver. TODO: Figure out how this can be
        # configured from the outside.
        default_options = {
            # "ksp_monitor": None,
            "ksp_type": "gmres",
            "ksp_pc_side": "right",
            "ksp_rtol": 1e-12,
            "ksp_max_it": 120,
            "ksp_gmres_cgs_refinement_type": "refine_ifneeded",
            "ksp_gmres_classicalgramschmidt": True,  # Not givens rotations??
        }

        options = default_options | (self.petsc_options or {}) | (options or {})

        # Insert the above options into the PETSc options singleton.
        insert_petsc_options(options)

        # Create the PETSc KSP object, set matrix and preconditioner.
        petsc_ksp = PETSc.KSP().create()
        petsc_ksp.setOperators(petsc_mat)
        petsc_ksp.setFromOptions()
        petsc_pc = petsc_ksp.getPC()
        if self.preconditioner is not None:
            options |= self.preconditioner.configure(
                bmat=mat_orig,
                pc=petsc_pc,
                user_options=options,
            )
        petsc_ksp.setUp()
        self.options = options
        return PetscKrylovSolver(petsc_ksp)


@dataclass
class LinearTransformedScheme:

    inner: PetscKSPScheme
    """The actual solver, to be applied after the transformations."""

    left_transformations: Optional[list] = None
    right_transformations: Optional[list] = None

    def get_groups(self) -> list[int]:
        return self.inner.get_groups()

    def make_solver(
        self, mat_orig: BlockMatrixStorage, options: dict | None = None
    ) -> PetscKrylovSolver | LinearSolverWithTransformations:
        bmat = mat_orig[:]

        if self.left_transformations is None or len(self.left_transformations) == 0:
            Qleft = None
        else:
            # The steps should be roughly the same as for the right transfor (below).
            Qleft = self.left_transformations[0](bmat)
            for tmp in self.left_transformations[1:]:
                tmp = tmp(bmat)
                Qleft.mat @= tmp.mat

        if self.right_transformations is None or len(self.right_transformations) == 0:
            Qright = None
        else:
            Qright = self.right_transformations[0](bmat)
            for tmp in self.right_transformations[1:]:
                tmp = tmp(bmat)
                Qright.mat @= tmp.mat

        bmat_Q = bmat
        if Qleft is not None:
            bmat_Q.mat = Qleft.mat @ bmat_Q.mat
        if Qright is not None:
            bmat_Q.mat = bmat_Q.mat @ Qright.mat

        if self.inner is None:
            raise ValueError("No inner solver provided.")

        # Set up the inner solver.
        solver: PetscKrylovSolver = self.inner.make_solver(bmat_Q, options=options)
        self.options = self.inner.options

        if Qleft is not None or Qright is not None:
            solver = LinearSolverWithTransformations(
                inner=solver, Qright=Qright, Qleft=Qleft
            )

        return solver
