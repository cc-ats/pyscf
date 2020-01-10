#!/bin/bash

# this file must be copied in pyscf/lib
# Python path binary if necessary (Anaconda, intelPython)
export PATH=/home/yangjunjie/anaconda3/bin:$PATH
export PYTHONPATH=/work/yangjunjie/pyscf:$PYTHONPATH

which python
which cmake
module purge
module load imkl/2018.1.163-iompi-2018a

export CC=gcc
export FC=gfortran
export CXX=g++

export LD_LIBRARY_PATH=$MKLROOT/lib:$LD_LIBRARY_PATH
python -c "import pyscf; print('pyscf path is ', pyscf.__path__[0])"

# mkdir -p build
# cd build
# rm -rf *
# cmake -DBLA_VENDOR=Intel10_64lp_seq ..
# make VERBOSE=1
