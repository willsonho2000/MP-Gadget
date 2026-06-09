# from fake_spectra.randspectra import RandSpectra
from fake_spectra.griddedspectra import GriddedSpectra
from fake_spectra import flux_power
from fake_spectra import lyman_data
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
# Only if you want the MPI feature; otherwise, MPI is None by default
# from mpi4py import MPI

# Three redshifts
z = [4.2, 4.6, 5.0]
obser_data = lyman_data.BoeraData()
kf = obser_data.get_kf()    # km / s

# Extract the power spectrum for different redshifts
zout = flux_power.MySpectra(max_z = 5.0, redshifts=z, pixel_resolution_km_s=1.0).zout
data_power  = obser_data.pf.reshape(-1, kf.shape[0]) #km / s; n_z * n_k # n_z * n_k
sigma_sq    = obser_data.covar_diag.reshape(-1, kf.shape[0])

def tau(A, B, z):
    return A*(1.0+z)**B

def post_analysis(num, A, B):
    for i in range(num):
        z_i = z[-1-i]
        gs = GriddedSpectra(i, "output", nspec=10, MPI=None, res=1.0, savefile="gridded_spectra.hdf5", reload_file=True)
        # gs.get_tau("H",1,1215)
        # #Lyman-beta
        # gs.get_tau("H",1,1025)
        # gs.get_col_density("H",1)
        mean_tau = tau(A, B, z_i)
        pfs = gs.get_flux_power_1D("H",1,mean_flux_desired=np.exp(mean_tau))     # get 1D flux power spectrum
        # Save spectra to file
        np.savetxt("SPECTRA_%03d/flux_power.txt" % i, pfs, fmt="%.3e")
        gs.save_file()

# Do the chi2 analysis for the flux power spectrum
def chi_squared():
    # Chi-squared results
    chi2 = np.zeros_like(z)

    for i, z in enumerate(zout):
        # Preparing data for the current redshift
        sim = np.loadtxt("SPECTRA_%03d/flux_power.txt" % (2 - i))

        skf = sim[0,:]          # km / s
        sim_power = sim[1,:]    # P_F(k) for the simulated data at the current redshift

        # Rebinning simulated data to match the k bins of the observed data
        interp_func = interp1d(skf, sim_power, kind='linear', bounds_error=False, fill_value="extrapolate")
        sim_power_rebinned = interp_func(kf)

        chi2[i] = np.sum((sim_power_rebinned - data_power[i])**2 / sigma_sq[i])
    
    return chi2

# Measure how good is the given tau value for each redshift
def chi_2_pipeline(A, B):
    # Use the post_analysis function to generate spectra with the current tau value
    post_analysis(3, A, B)

    # chi2 analysis
    chi2 = chi_squared()
    # print(chi2)
    # print(f"Chi-squared value: {np.sum(chi2):.3f} at z = {z:.1f}\n")
    
    return np.sum(chi2)

# Use the least squares method to find the best tau value that minimizes the chi2
res = minimize(chi_2_pipeline, x0=(2.3e-3, 3.65))

# Print the best tau value and the corresponding chi2
print(f"Best tau value: {res.fun:.3f}")

post_analysis(3, mean_tau=res.x)
print(f"Minimum chi-squared value: {chi_squared():.3f}")