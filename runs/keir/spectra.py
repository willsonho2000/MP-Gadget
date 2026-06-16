# from fake_spectra.randspectra import RandSpectra
from fake_spectra.griddedspectra import GriddedSpectra
import numpy as np
# Only if you want the MPI feature; otherwise, MPI is None by default
from mpi4py import MPI

for i in range(3):
    gs = GriddedSpectra(i, "output", nspec=200, res=1.0, MPI=MPI, savefile="gridded_spectra_200.hdf5")
    gs.get_tau("H",1,1215)
    #Lyman-beta
    gs.get_tau("H",1,1025)
    gs.get_col_density("H",1)
    gs.save_file()


