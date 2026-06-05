# from fake_spectra.randspectra import RandSpectra
from fake_spectra.griddedspectra import GriddedSpectra
import numpy as np
# Only if you want the MPI feature; otherwise, MPI is None by default
from mpi4py import MPI

# rr = RandSpectra(5, "MySim", MPI=MPI, thresh=0.)
for i in range(3):
    gs = GriddedSpectra(i, "output", nspec=10, MPI=MPI, res=1.0, savefile="gridded_spectra.hdf5", reload_file=True)
    gs.get_tau("H",1,1215)
    #Lyman-beta
    gs.get_tau("H",1,1025)
    gs.get_col_density("H",1)
    gs.get_flux_power_1D("H",1)     # get 1D flux power spectrum
    #fps.reshape(-1, 1)
    #print(fps)
    #Save spectra to file
    #inp.savetxt("fps_%03d.txt" % i, fps, fmt="%.3e")
    gs.save_file()


