"""
Optimize the effective optical depth (tau_eff) of the Lyman-alpha forest
at three redshifts (z = 4.2, 4.6, 5.0) by fitting simulated flux power
spectra to observational data from Boera et al.

For each redshift, a scalar optimizer varies tau_eff, generates a 1D flux
power spectrum from simulation snapshots via fake_spectra, and minimizes
the chi-squared residual against the observed P_F(k).
"""

from fake_spectra.spectra import Spectra

from lyaemu import lyman_data
from lyaemu import flux_power
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
# Only if you want the MPI feature; otherwise, MPI is None by default

# ── Observational data ───────────────────────────────────────────────
# Three target redshifts corresponding to Boera et al. measurements
z = [4.2, 4.6, 5.0]
obser_data = lyman_data.BoeraData()
# Wavenumber bins from the observed data in velocity units (km/s)
kf = obser_data.get_kf()

# Reshape the flat observed arrays into (n_redshifts, n_k_bins) matrices
zout = flux_power.MySpectra(max_z = 5.0, redshifts=z, pixel_resolution_km_s=1.0).zout
data_power  = obser_data.pf.reshape(-1, kf.shape[0])           # observed P_F(k), shape (n_z, n_k)
sigma_sq    = obser_data.covar_diag.reshape(-1, kf.shape[0])    # diagonal variance, shape (n_z, n_k)

# ── Helper functions ─────────────────────────────────────────────────

def post_analysis(i, tau, nspec):
    """
    Load simulation snapshot `i` and compute the 1D Lyman-alpha flux power
    spectrum, rescaling the mean flux to exp(-tau).

    Parameters
    ----------
    i : int
        Snapshot index (maps to a redshift via the simulation output ordering).
    tau : float
        Effective optical depth; the mean transmitted flux is set to exp(-tau).
    nspec : int
        Number of sightlines used when the spectra were extracted; determines
        the savefile name (e.g. gridded_spectra_300.hdf5).

    Returns
    -------
    tuple of (k, P_F(k))
        Wavenumber array and corresponding 1D flux power spectrum.
    """
    gs = Spectra(i, "output", MPI=None, res=1.0,
                 savefile="gridded_spectra_%03d.hdf5" % nspec,
                 cofm=None, axis=None)
    # mean_flux_desired = exp(-tau) rescales optical depths so the mean
    # transmitted flux matches the target; tau_thresh caps individual pixels
    # to avoid numerical issues from saturated absorption.
    fps = gs.get_flux_power_1D("H", 1,
                               mean_flux_desired=np.exp(-tau),
                               tau_thresh=1e6)
    return fps


def chi_squared(i, sim):
    """
    Compute the chi-squared statistic between a simulated flux power spectrum
    and the Boera et al. observed data at redshift index `i`.

    The simulated P_F(k) is interpolated onto the observed k-bins before
    comparison.
    """
    skf = sim[0]          # simulated wavenumbers (km/s)
    sim_power = sim[1]    # simulated P_F(k)

    # Interpolate the simulation onto the (coarser) observational k-bins
    interp_func = interp1d(skf, sim_power, kind='linear',
                           bounds_error=False, fill_value="extrapolate")
    sim_power_rebinned = interp_func(kf)

    # Simple diagonal chi-squared (no off-diagonal covariance)
    return np.sum((sim_power_rebinned - data_power[i])**2 / sigma_sq[i])


def chi_2_pipeline(x, num_snap, last_snap):
    """
    Objective function for the optimizer: given a trial tau_eff (`x`) and a
    redshift index (`num_z`), return the chi-squared.

    The snapshot index is reversed (last_snap - num_snap) because snapshot ordering runs
    from high to low redshift while the data arrays are ordered low to high.
    """
    fps = post_analysis(num_snap, tau=x, nspec=200)
    return chi_squared(last_snap - num_snap, fps)


# ── Optimization ─────────────────────────────────────────────────────
# Initial guesses for tau_eff at z = 4.2, 4.6, 5.0 (from literature values)
tau_0 = [0.944, 1.2377, 1.592]
best_tau = np.zeros_like(tau_0)
best_chi2 = np.zeros_like(tau_0)

# Optimize tau_eff independently at each redshift using bounded scalar minimization
# Last snapshot index (z = 4.2)
last_snap = 5
# i from the snapshot of z = 5.0 to 4.2 (total of 3)
for i in range(last_snap - 2, last_snap + 1):
    res = minimize_scalar(chi_2_pipeline, bounds=(0.5, 2.5), args=(i, last_snap))
    best_tau[last_snap - i] = res.x
    best_chi2[last_snap - i] = res.fun

# Report best-fit tau_eff and corresponding chi-squared at each redshift
print("\n Best Tau:")
print(np.array2string(best_tau, separator=","))
print("\n Best Chi-2:")
print(np.array2string(best_chi2, separator=","))