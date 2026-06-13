#!/bin/bash -l

#SBATCH --partition=normal
#SBATCH --job-name=lym-alpha
#SBATCH --time=12:00:00
#SBATCH --nodes=40 
#SBATCH --ntasks-per-node=2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mho101@ucr.edu
#SBATCH -A AST21005

export OMP_NUM_THREADS=28

module load ooops gsl

#ibrun MP-GenIC paramfile.genic
#ibrun MP-Gadget paramfile.gadget 1
#python spectra.py
