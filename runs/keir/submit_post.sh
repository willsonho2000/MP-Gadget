#!/bin/bash -l

#SBATCH --partition=small
#SBATCH --job-name=lym-alpha
#SBATCH --time=8:00:00
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mho101@ucr.edu
#SBATCH -A AST21005


#module rm python3
module load ooops gsl
export OMP_NUM_THREADS=56

#alias python3="/home1/11547/willsonho2000/miniconda3/bin/python"

#ibrun MP-GenIC paramfile.genic
#ibrun MP-Gadget paramfile.gadget 1
python3=/home1/11547/willsonho2000/miniconda3/bin/python
ibrun python3 spectra.py
#python3 spectra_bk.py

exit 0
