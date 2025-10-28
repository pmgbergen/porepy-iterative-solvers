FROM porepy/dev:latest

# HOME should be inherited from the porepy image.
ENV PETSC_SRC_DIR=${HOME}/petsc \
    PETSC_ARCH=arch-linux-c-opt \
    PETSC_INSTALL_DIR=${HOME}/petsc_dist

# Installing requirements needed to compile PETSc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git build-essential autoconf libtool flex cmake gfortran && \
    rm -rf /var/lib/apt/lists/* && \
    python -m pip install --no-cache-dir cython setuptools

# Downloading PETSc repository. Using --single-branch and --depth 1 to download only
# the latest commit, not the full git history.
RUN git clone --branch release --single-branch --depth 1 https://gitlab.com/petsc/petsc.git $PETSC_SRC_DIR
WORKDIR $PETSC_SRC_DIR

# Installing PETSc.
# --download-XXX-configure-arguments allows to pass custom configure arguments to a
# library XXX. We use it for PNETCDF because the default script fails to inform it about
# MPI location. This is probably a bug and should be revisited in 2026.
RUN echo "Configuring PETSc in $PETSC_SRC_DIR" && \
    ./configure \
        --COPTFLAGS=-O3 -march=native -mtune=native \
        --CXXOPTFLAGS=-O3 -march=native -mtune=native \
        --FOPTFLAGS=-O3 -march=native -mtune=native \
        --prefix=${PETSC_INSTALL_DIR} \
        --with-c2html=0 \
        --with-debugging=0 \
        --with-make-np=$(nproc) \
        --with-shared-libraries=1 \
        --with-zlib=1 \
        –-with-petsc4py=1 \
        --download-bison \
        --download-fblaslapack \
        --download-fftw \
        --download-hdf5 \
        --download-hwloc \
        --download-hypre \
        --download-metis \
        --download-mumps \
        --download-mpich \
        --download-mpi4py \
        --download-netcdf \
        --download-pnetcdf \
        --download-pnetcdf-configure-arguments=MPICC="${PETSC_SRC_DIR}/${PETSC_ARCH}/bin/mpicc" MPICXX="${PETSC_SRC_DIR}/${PETSC_ARCH}/bin/mpicxx" MPIF90="${PETSC_SRC_DIR}/${PETSC_ARCH}/bin/mpif90" \
        --download-ptscotch \
        --download-scalapack \
        --download-spai \
        --download-suitesparse \
        --download-superlu_dist \
        --download-zlib && \
    echo "Building PETSc" && \
    make PETSC_DIR=$PETSC_SRC_DIR PETSC_ARCH=$PETSC_ARCH all

# This command copies action 
RUN make -j$(nproc) test && \
    make install && \
    make -j$(nproc) PETSC_DIR=${PETSC_INSTALL_DIR} PETSC_ARCH="" check

# Changing workdir to HOME.
WORKDIR ${HOME}

# Add petsc4py and mpi4py to PYTHONPATH
ENV PYTHONPATH=${PYTHONPATH}:${PETSC_INSTALL_DIR}/lib
