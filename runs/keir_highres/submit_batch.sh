#!/bin/bash -l

#SBATCH --partition=nvdimm
#SBATCH --nodes=2
#SBATCH --ntasks=128
#SBATCH --tasks-per-node=64
#SBATCH --time=24:00:00     # 1 day
#SBATCH --mail-user=mho101@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name="hydro_lym"
#SBATCH -A AST21005

# Print current date
date

# Load samtools
# module load samtools
module load ooops gsl

# Job
# ROOT=/rhome/mho101/MP-Gadget
#mpirun ./MP-GenIC paramfile.genic
mpirun ./MP-Gadget paramfile.gadget 1
# mpirun -np 4 $ROOT/gadget/MP-Gadget paramfile.gadget || exit 1
# python ./spectra.py

# Print name of node
hostname
