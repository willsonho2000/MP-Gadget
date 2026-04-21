#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=20gb
#SBATCH --time=24:00:00     # 1 day
#SBATCH --mail-user=mho101@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name="hydro_lym"
#SBATCH -p intel # You can use any of the following; epyc, intel, batch, highmem, gpu

# Print current date
date

# Load samtools
# module load samtools

# Job
# ROOT=/rhome/mho101/MP-Gadget
# mpirun -np 16 ./MP-GenIC paramfile.genic
mpirun ./MP-Gadget paramfile.gadget 1
# mpirun -np 4 $ROOT/gadget/MP-Gadget paramfile.gadget || exit 1
# python ./spectra.py

# Print name of node
hostname