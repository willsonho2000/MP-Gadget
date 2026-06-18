# from fake_spectra.randspectra import RandSpectra
from fake_spectra.griddedspectra import GriddedSpectra
import numpy as np
# Only if you want the MPI feature; otherwise, MPI is None by default
from mpi4py import MPI

nspec_list = [100, 300]
for j in range(2):
    for i in range(3):
        gs = GriddedSpectra(i, "output", nspec=nspec_list[j], res=1.0, MPI=MPI, savefile="gridded_spectra_%03d.hdf5" % nspec_list[j])
        gs.get_tau("H",1,1215)
        #Lyman-beta
        gs.get_tau("H",1,1025)
        gs.get_col_density("H",1)
        gs.save_file()