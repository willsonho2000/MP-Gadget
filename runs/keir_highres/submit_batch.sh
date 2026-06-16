#!/bin/bash -l

#SBATCH --partition=normal
#SBATCH --nodes=16
#SBATCH --tasks-per-node=2
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

export OMP_NUM_THREADS=28

# Job
# ROOT=/rhome/mho101/MP-Gadget
#ibrun ./MP-GenIC paramfile.genic
ibrun ./MP-Gadget paramfile.gadget

# Print name of node
hostname
