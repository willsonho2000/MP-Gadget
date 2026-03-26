#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20gb
#SBATCH --time=02:00:00     # 1 day and 15 minutes
#SBATCH --mail-user=mho101@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name="lym_test"
#SBATCH -p short # You can use any of the following; epyc, intel, batch, highmem, gpu

# Print current date
date

# Load samtools
# module load samtools

# Job
# ROOT=/rhome/mho101/MP-Gadget
# mpirun ./MP-GenIC paramfile.genic
# mpirun ./MP-Gadget paramfile.gadget 1
# mpirun -np 4 $ROOT/gadget/MP-Gadget paramfile.gadget || exit 1
python ./spectra.py

# Print name of node
hostname