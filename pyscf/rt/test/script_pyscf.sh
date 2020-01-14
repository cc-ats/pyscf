#!/bin/bash
#
#SBATCH --partition=debug
#SBATCH --ntasks=20
#SBATCH --tasks-per-node=20
#SBATCH --mem=40000
#SBATCH --output=./log/%J_stdout.log
#SBATCH --error=./log/%J_stderr.log
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --job-name=RT
#SBATCH --mail-user=yangjunjie0320@ou.edu

source /home/yangjunjie/pyscf/pyscf/lib/compile-oscer.sh

# MAIN

python test_spectrum.py > test_spectrum.log
