"""
Optimize the effective optical depth (tau_eff) of the Lyman-alpha forest
at three redshifts (z = 4.2, 4.6, 5.0) by fitting simulated flux power
spectra to observational data from Boera et al.

For each redshift, a scalar optimizer varies tau_eff, generates a 1D flux
power spectrum from simulation snapshots via fake_spectra, and minimizes
the chi-squared residual against the observed P_F(k).
"""

import glob
import os.path as path

from fake_spectra.spectra import Spectra

from lyaemu import lyman_data
from lyaemu import flux_power
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
# Only if you want the MPI feature; otherwise, MPI is None by default

# ── Observational data ───────────────────────────────────────────────
# Three target redshifts corresponding to Boera et al. measurements
z = [4.2, 4.6, 4.9]
obser_data = lyman_data.BoeraData()
# Wavenumber bins from the observed data in velocity units (km/s)
kf = obser_data.get_kf()

# Reshape the flat observed arrays into (n_redshifts, n_k_bins) matrices
zout = flux_power.MySpectra(max_z = 5.0, redshifts=z, pixel_resolution_km_s=1.0).zout
data_power  = obser_data.pf.reshape(-1, kf.shape[0])           # observed P_F(k), shape (n_z, n_k)
sigma_sq    = obser_data.covar_diag.reshape(-1, kf.shape[0])    # diagonal variance, shape (n_z, n_k)

# ── Helper functions ─────────────────────────────────────────────────

def find_spectra_file(i, base="output"):
    """
    Find the saved spectra .hdf5 file in the snapshot's SPECTRA_NNN folder
    and return its basename (e.g. lya_forest_spectra_grid_200.hdf5).
    """
    files = glob.glob(path.join(base, "SPECTRA_%03d" % i, "*.hdf5"))
    if len(files) != 1:
        raise IOError("Expected one spectra file for snapshot %d, found: %s" % (i, files))
    return path.basename(files[0])


def post_analysis(i, tau):
    """
    Load simulation snapshot `i` and compute the 1D Lyman-alpha flux power
    spectrum, rescaling the mean flux to exp(-tau).

    Parameters
    ----------
    i : int
        Snapshot index (maps to a redshift via the simulation output ordering).
    tau : float
        Effective optical depth; the mean transmitted flux is set to exp(-tau).

    Returns
    -------
    tuple of (k, P_F(k))
        Wavenumber array and corresponding 1D flux power spectrum.
    """
    gs = Spectra(i, "output", MPI=None, res=1.0,
                 savefile=find_spectra_file(i),
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


def find_snapshots(base="output"):
    """
    Find the available snapshot indices in `base` (need not be contiguous,
    e.g. [7, 8, 10]), checking each of the directory naming conventions
    fake_spectra looks for (PART_, snapdir_, SPECTRA_).
    """
    nums = {int(path.basename(d)[len(prefix):])
            for prefix in ("PART_", "snapdir_", "SPECTRA_")
            for d in glob.glob(path.join(base, prefix + "[0-9][0-9][0-9]"))}
    return sorted(nums)


def chi_2_pipeline(x, num_snap, num_z):
    """
    Objective function for the optimizer: given a trial tau_eff (`x`), a
    snapshot index (`num_snap`) and a redshift index (`num_z`), return
    the chi-squared.
    """
    fps = post_analysis(num_snap, tau=x)
    return chi_squared(num_z, fps)


# ── Optimization ─────────────────────────────────────────────────────
# Initial guesses for tau_eff at z = 4.2, 4.6, 5.0 (from literature values)
tau_0 = [0.944, 1.2377, 1.592]
best_tau = np.zeros_like(tau_0)
best_chi2 = np.zeros_like(tau_0)

# Optimize tau_eff independently at each redshift using bounded scalar minimization
# Auto-detect the available snapshots (e.g. [7, 8, 10]) and take the last
# len(z) of them. Snapshot ordering runs from high to low redshift while the
# data arrays are ordered low to high, so the last snapshot maps to z index 0.
snaps = find_snapshots("output")[-len(z):]
print("Using snapshots:", snaps)
for j, snap in enumerate(snaps):
    num_z = len(snaps) - 1 - j
    res = minimize_scalar(chi_2_pipeline, bounds=(0.5, 2.5), args=(snap, num_z))
    best_tau[num_z] = res.x
    best_chi2[num_z] = res.fun

# Report best-fit tau_eff and corresponding chi-squared at each redshift
print("\n Best Tau:")
print(np.array2string(best_tau, separator=","))
print("\n Best Chi-2:")
print(np.array2string(best_chi2, separator=","))