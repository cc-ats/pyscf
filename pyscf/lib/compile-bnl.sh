#!/bin/bash
# this file must be copied in pyscf/lib
# Python path binary if necessary (Anaconda, intelPython)

module purge
module load cmake
module load intel
module load anaconda3

which cmake
which python

export CC=icc
export FC=ifort
export CXX=icpc

export PYTHONPATH=/sdcc/u/yihan/rt-tddft/pyscf:$PYTHONPATH
export LD_LIBRARY_PATH=$MKLROOT/lib:$LD_LIBRARY_PATH

# mkdir -p build
# cd build
# rm -rf *
# cmake -DBLA_VENDOR=Intel10_64lp_seq ..
# make -j 20 VERBOSE=1

export OMP_NUM_THREADS=20
export TMPDIR=/tmp

python -c "import pyscf; print('pyscf path is ', pyscf.__path__[0])"
