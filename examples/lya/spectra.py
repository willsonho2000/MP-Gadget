# from fake_spectra.randspectra import RandSpectra
from fake_spectra.griddedspectra import GriddedSpectra
# Only if you want the MPI feature; otherwise, MPI is None by default
from mpi4py import MPI

# rr = RandSpectra(5, "MySim", MPI=MPI, thresh=0.)
for i in range(14):

    gs = GriddedSpectra(i, "output", nspec=10, MPI=MPI, nbins=1000)
    gs.get_tau("H",1,1215)
    #Lyman-beta
    gs.get_tau("H",1,1025)
    gs.get_col_density("H",1)
    #Save spectra to file
    gs.save_file()