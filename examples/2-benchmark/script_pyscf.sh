#!/bin/bash -v
#
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=40000
#SBATCH --output=./log/%J_stdout.log
#SBATCH --error=./log/%J_stderr.log
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --job-name=Benchmark
#SBATCH --mail-user=yangjunjie0320@ou.edu

source /home/yangjunjie/pyscf/pyscf/lib/compile-oscer.sh
# MAIN

export OMP_NUM_THREADS=1
python bz.py > bz_ntasks_1.log

