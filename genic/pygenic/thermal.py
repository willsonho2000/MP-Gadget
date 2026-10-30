"""Thermal velocities drawn from a Fermi-Dirac distribution. Port of libgenic/thermal.c."""

import math

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline

from .physconst import LIGHTCGS

MAX_FERMI_DIRAC = 17.0
LENGTH_FERMI_DIRAC_TABLE = 2000


def NU_V0(Time, kBTNubyMNu, UnitVelocity_in_cm_per_s):
    """Velocity scale for neutrinos: v/c = x kT / (m a).

    NOTE: this m is the mass of a SINGLE neutrino species, not the sum of neutrinos."""
    return kBTNubyMNu / Time * (LIGHTCGS / UnitVelocity_in_cm_per_s)


def WDM_V0(Time, WDM_therm_mass, Omega_CDM, HubbleParam, UnitVelocity_in_cm_per_s):
    """Amplitude of the random velocity for WDM (fiducial values from Bode, Ostriker & Turok 2001)."""
    v0 = (0.012 / Time * (Omega_CDM / 0.3) ** (1.0 / 3) * (HubbleParam / 0.65) ** (2.0 / 3)
          * (1.0 / WDM_therm_mass) ** (4.0 / 3))
    return v0 * 1.0e5 / UnitVelocity_in_cm_per_s


def _fermi_dirac_kernel(x):
    return x * x / (math.exp(x) + 1)


class ThermalVel:
    """Inverse cumulative distribution of Fermi-Dirac speeds, as init_thermalvel()."""

    def __init__(self, v_amp, max_fd, min_fd=0.0):
        if max_fd <= min_fd:
            raise ValueError("Thermal vel called with negative interval: %g <= %g" % (max_fd, min_fd))
        max_fd = min(max_fd, MAX_FERMI_DIRAC)
        self.m_vamp = v_amp
        vel = min_fd + (max_fd - min_fd) * np.arange(LENGTH_FERMI_DIRAC_TABLE) / (LENGTH_FERMI_DIRAC_TABLE - 1.0)
        # Cumulative integral, done piecewise between table entries.
        pieces = [quad(_fermi_dirac_kernel, vel[i], vel[i + 1], epsabs=0, epsrel=1e-6)[0]
                  for i in range(LENGTH_FERMI_DIRAC_TABLE - 1)]
        cumprob = np.concatenate([[0.], np.cumsum(pieces)])
        total_fd = quad(_fermi_dirac_kernel, 0, MAX_FERMI_DIRAC, epsabs=0, epsrel=1e-6)[0]
        assert total_fd > 1.8
        # Fraction of the total F-D probability (3 zeta(3) / 2) that is below max_fd.
        self.total_frac = cumprob[-1] / total_fd
        cumprob /= cumprob[-1]
        self.fermi_dirac_vel = vel
        self.fermi_dirac_cumprob = cumprob
        self._spline = CubicSpline(cumprob, vel, bc_type="natural")

    def add_thermal_speeds(self, rng, vel):
        """Add an isotropic random thermal velocity to each row of vel (shape (N, 3)), in place."""
        n = vel.shape[0]
        p = rng.uniform(size=n)
        v = self.m_vamp * self._spline(p)
        # Random direction
        phi = 2 * np.pi * rng.uniform(size=n)
        theta = np.arccos(2 * rng.uniform(size=n) - 1)
        vel[:, 0] += v * np.sin(theta) * np.cos(phi)
        vel[:, 1] += v * np.sin(theta) * np.sin(phi)
        vel[:, 2] += v * np.cos(theta)
        return vel
