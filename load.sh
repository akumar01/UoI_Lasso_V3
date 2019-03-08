module load eigen3
module load boost
module load gsl
module list
module unload cray-mpich
module load pe_archive
module load cray-mpich/7.4.4

module load cray-hdf5-parallel/1.8.14
module list
export LD_LIBRARY_PATH=$HDF5_ROOT/lib:$LD_LIBRARY_PATH
