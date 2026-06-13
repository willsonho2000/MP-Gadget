#!/bin/bash -l

#SBATCH --partition=normal
#SBATCH --job-name=lym-alpha
#SBATCH --time=0:15:00
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mho101@ucr.edu
#SBATCH -A AST21005

export OMP_NUM_THREADS=2

#module rm python3
module load ooops gsl

#alias python3="/home1/11547/willsonho2000/miniconda3/bin/python"

#ibrun MP-GenIC paramfile.genic
#ibrun MP-Gadget paramfile.gadget 1
which python3
python3 spectra.py

echo "This line executes."
exit 0
