# from fake_spectra.randspectra import RandSpectra
from fake_spectra.spectra import Spectra

from lyaemu import lyman_data
from lyaemu import flux_power
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
# Only if you want the MPI feature; otherwise, MPI is None by default
# from mpi4py import MPI

# Three redshifts
z = [4.2, 4.6, 5.0]
obser_data = lyman_data.BoeraData()
kf = obser_data.get_kf()    # km / s

# Extract the observed power spectrum for different redshifts
zout = flux_power.MySpectra(max_z = 5.0, redshifts=z, pixel_resolution_km_s=1.0).zout
data_power  = obser_data.pf.reshape(-1, kf.shape[0]) #km / s; n_z * n_k # n_z * n_k
sigma_sq    = obser_data.covar_diag.reshape(-1, kf.shape[0])

# Post analysis function to generate spectra with the given tau value and extract the flux power spectrum
def post_analysis(i, tau):
    # To just spectra
    gs = Spectra(i, "output", MPI=None, res=1.0, savefile="gridded_spectra.hdf5", cofm=None, axis=None)
    fps = gs.get_flux_power_1D("H",1,mean_flux_desired=np.exp(-tau))     # get 1D flux power spectrum
    return fps

# Do the chi2 analysis for the flux power spectrum
def chi_squared(i, sim):
    # Chi-squared results
    # Preparing data for the current redshift

    skf = sim[0]          # km / s
    sim_power = sim[1]    # P_F(k) for the simulated data at the current redshift

    # Rebinning simulated data to match the k bins of the observed data
    interp_func = interp1d(skf, sim_power, kind='linear', bounds_error=False, fill_value="extrapolate")
    sim_power_rebinned = interp_func(kf)

    return np.sum((sim_power_rebinned - data_power[i])**2 / sigma_sq[i])

# Measure how good is the given tau value for each redshift
def chi_2_pipeline(x, num_z):
    # Use the post_analysis function to generate spectra with the current tau value
    fps = post_analysis(2 - num_z, tau=x)

    # chi2 analysis
    return chi_squared(num_z, fps)

# Use the least squares method to find the best tau value that minimizes the chi2
tau_0 = [0.944, 1.2377, 1.592 ]
best_tau = np.zeros_like(tau_0)
best_chi2 = np.zeros_like(tau_0)

# Run the three optimization processes for each redshift
for i in range(3):
    print(chi_2_pipeline(tau_0[i], i))
    res = minimize_scalar(chi_2_pipeline, bounds=(0.5, 2.5), args=(i))
    best_tau[i] = res.x
    best_chi2[i] = res.fun

# Print out the final results
print(best_tau, best_chi2)