FROM porepy/dev:latest

ENV PETSC_DIR=${HOME}/petsc
ENV PETSC_ARCH=arch-linux-c-opt

RUN git clone -b release https://gitlab.com/petsc/petsc.git $PETSC_DIR; \
    cd ${PETSC_DIR}
    

RUN echo "Configuring PETSc in $PETSC_DIR"; \
    ./configure \
        --COPTFLAGS=-O3 -march=native -mtune=native \
        --CXXOPTFLAGS=-O3 -march=native -mtune=native \
        --FOPTFLAGS=-O3 -march=native -mtune=native \
        --with-c2html=0 \
        --with-debugging=0 \
        --with-make-np=12 \
        --with-shared-libraries=1 \
        --with-zlib \
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
        --download-ptscotch \
        --download-scalapack \
        --download-spai \
        --download-suitesparse \
        --download-superlu_dist \
        --download-zlib ;

RUN echo "Building PETSc"; \
    make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH all; \
    echo "Running PETSc tests"; \
    make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH check

# Adding PETSc executables to path to make downloaded packages (e.g. mpiexec) available.
ENV PATH=${PATH}:${PETSC_DIR}/${PETSC_ARCH}/bin

RUN echo "Installing mpi4py and petsc4py"; \
    python -m pip install src/binding/petsc4py