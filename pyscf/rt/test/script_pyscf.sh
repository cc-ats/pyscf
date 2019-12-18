#!/bin/bash -v
#
#SBATCH --partition=normal
#SBATCH --ntasks=20
#SBATCH --tasks-per-node=20
#SBATCH --mem=40000
#SBATCH --output=%J_stdout.log
#SBATCH --error=%J_stderr.log
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --job-name=TDSCF
#SBATCH --mail-user=yangjunjie0320@ou.edu

date;
time;

source /home/yangjunjie/work/pyscf/pyscf/lib/compile-oscer.sh;

# MAIN
python test_abos_spectrum.py > test_abos_spectrum.log

