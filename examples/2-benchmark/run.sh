#!/bin/bash
#
#SBATCH --partition=normal
#SBATCH --ntasks=20
#SBATCH --tasks-per-node=20
#SBATCH --mem=40000
#SBATCH --output=./%J.log
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --job-name=RT
#SBATCH --mail-user=yangjunjie0320@ou.edu

source /home/yangjunjie/pyscf/pyscf/lib/compile-oscer.sh

# MAIN

# python test_hf.py > test_hf.log
python bz.py
