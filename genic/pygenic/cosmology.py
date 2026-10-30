"""Background cosmology: Hubble function, growth factor and neutrino density.

Port of libgadget/cosmology.c and libgadget/omega_nu_single.c (without hybrid neutrinos,
which GenIC never enables)."""

import math

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import CubicSpline

from .physconst import (BOLEVK, GRAVITY, HBAR, HUBBLE, LIGHTCGS, NUSPECIES,
                        STEFAN_BOLTZMANN, TNUCMB)

# Size of matter density tables
NRHOTAB = 200
# Floating point accuracy for comparing neutrino masses
FLOAT_ACC = 1e-6
# Value of kT/aM_nu on which to switch from the analytic expansion to the numerical integration
NU_SW = 100
ZETA3 = 1.202056903159594


def get_rho_nu_conversion():
    """Conversion factor from (eV/c)^4 to g/cm^3 for a single neutrino species."""
    # The factor of two is for antineutrinos
    convert = 4 * math.pi * 2
    chbar = 1. / (2 * math.pi * LIGHTCGS * HBAR)
    convert *= chbar ** 3
    # 1 eV/c^2 in g
    convert *= 1.60217646e-12 / LIGHTCGS / LIGHTCGS
    return convert


def _rho_nu_int(q, amnu, kT):
    epsilon = math.sqrt(q * q + amnu * amnu)
    return q * q * epsilon / (math.exp(q / kT) + 1)


class RhoNuSingle:
    """Tabulated density of a single neutrino species, as in rho_nu_init()."""

    def __init__(self, a0, mnu, kBtnu):
        self.mnu = mnu
        self.spline = None
        self.loga0 = None
        # Massless neutrinos are always relativistic: no table needed.
        if mnu < 1e-6 * kBtnu:
            return
        # Tabulate to 1e-3, unless earlier requested. Need early times for growth function.
        a0 = min(a0, 1e-3)
        # Do not need a table when relativistic
        if a0 * mnu < 1e-6 * kBtnu:
            a0 = 1e-6 * kBtnu / mnu
        logA0 = math.log(a0) - math.log(1.2)
        logaf = math.log(NU_SW * kBtnu / mnu) + math.log(1.2)
        if logaf < logA0:
            return
        loga = logA0 + np.arange(NRHOTAB) * (logaf - logA0) / (NRHOTAB - 1)
        rhonu = np.empty(NRHOTAB)
        conv = get_rho_nu_conversion()
        for i, la in enumerate(loga):
            val, _ = quad(_rho_nu_int, 0, 500 * kBtnu, args=(mnu * math.exp(la), kBtnu),
                          epsabs=0, epsrel=1e-9, limit=200)
            rhonu[i] = val / math.exp(la) ** 4 * conv
        self.loga0 = loga[0]
        self.spline = CubicSpline(loga, rhonu, bc_type="natural")

    def rho_nu(self, a, kT):
        """Physical density of this species at scale factor a, in g/cm^3."""
        amnu = a * self.mnu
        if amnu > 0:
            kTamnu2 = kT * kT / amnu / amnu
        else:
            kTamnu2 = np.inf
        if NU_SW * NU_SW * kTamnu2 < 1:
            # Heavily non-relativistic: series expansion with Riemann zetas 3, 5, 7
            return amnu * kT ** 3 / a ** 4 * (
                1.5 * ZETA3 + kTamnu2 * 45. / 4. * 1.0369277551433704
                + 2835. / 32. * kTamnu2 ** 2 * 1.0083492773819229
                + 80325 / 32. * kTamnu2 ** 3 * 1.0020083928260826) * get_rho_nu_conversion()
        if amnu < 1e-6 * kT or self.spline is None or math.log(a) < self.loga0:
            # Heavily relativistic
            return 7 * (math.pi * kT / a) ** 4 / 120. * get_rho_nu_conversion()
        return float(self.spline(math.log(a)))


class OmegaNu:
    """Neutrino background density, as in init_omega_nu()."""

    def __init__(self, MNu, a0, HubbleParam, tcmb0):
        self.tcmb0 = tcmb0
        self.kBtnu = BOLEVK * TNUCMB * tcmb0
        self.rhocrit = (3 * (HUBBLE * HubbleParam) ** 2) / (8 * math.pi * GRAVITY)
        # Work out which neutrinos are degenerate with each other
        self.nu_degeneracies = [0] * NUSPECIES
        for mi in range(NUSPECIES):
            for mmi in range(mi):
                if abs(MNu[mi] - MNu[mmi]) < FLOAT_ACC:
                    self.nu_degeneracies[mmi] += 1
                    break
            else:
                self.nu_degeneracies[mi] = 1
        self.tabs = [RhoNuSingle(a0, MNu[mi], self.kBtnu) if self.nu_degeneracies[mi] else None
                     for mi in range(NUSPECIES)]

    def get_omega_nu(self, a):
        """Total matter density in neutrinos at scale factor a, in units of the critical density today."""
        rhonu = 0
        for mi in range(NUSPECIES):
            if self.nu_degeneracies[mi] > 0:
                rhonu += self.nu_degeneracies[mi] * self.tabs[mi].rho_nu(a, self.kBtnu)
        return rhonu / self.rhocrit

    def omega_nu_single(self, a, i):
        """Matter density in neutrino species i."""
        if self.nu_degeneracies[i] == 0:
            for j in range(i, -1, -1):
                if self.nu_degeneracies[j]:
                    i = j
                    break
        return self.tabs[i].rho_nu(a, self.kBtnu) / self.rhocrit


class Cosmology:
    """Port of the Cosmology struct and its functions."""

    def __init__(self, cp, TimeBegin, UnitLength_in_cm, UnitMass_in_g, UnitVelocity_in_cm_per_s):
        for key, value in vars(cp).items():
            setattr(self, key, value)
        unit_time_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s
        self.Hubble = HUBBLE * unit_time_in_s
        self.UnitTime_in_s = unit_time_in_s
        self.GravInternal = GRAVITY / UnitLength_in_cm ** 3 * UnitMass_in_g * unit_time_in_s ** 2
        self.RhoCrit = 3.0 * self.Hubble ** 2 / (8.0 * math.pi * self.GravInternal)
        # Omega_g = 4 sigma_B T_CMB^4 8 pi G / (3 c^3 H^2)
        self.OmegaG = (4 * STEFAN_BOLTZMANN * self.CMBTemperature ** 4 * (8 * math.pi * GRAVITY)
                       / (3 * LIGHTCGS ** 3 * HUBBLE ** 2) / self.HubbleParam ** 2)
        self.ONu = OmegaNu(self.MNu, TimeBegin, self.HubbleParam, self.CMBTemperature)
        self.OmegaCDM = self.Omega0 - self.OmegaBaryon
        # Massive neutrinos are included in Omega0, so remove them from OmegaCDM.
        if sum(self.MNu) > 0:
            self.OmegaCDM -= self.ONu.get_omega_nu(1)
        self.OmegaK = 1.0 - self.Omega0 - self.OmegaLambda - self.Omega_fld
        if self.use_class_radiation_convention:
            self.OmegaK = (1.0 - self.OmegaCDM - self.OmegaBaryon - self.OmegaLambda - self.Omega_fld
                           - self.Omega_ur - self.OmegaG - self.ONu.get_omega_nu(1))

    def get_omega_nu(self, a):
        return self.ONu.get_omega_nu(a)

    def omega_fld(self, a):
        """Dark energy fluid density with w(a) = w0 + (1-a) wa."""
        if self.Omega_fld == 0.:
            return 0
        return (self.Omega_fld * a ** (-3 * (1 + self.w0_fld + self.wa_fld))
                * math.exp(-3 * self.wa_fld * (1 - a)))

    def hubble_function(self, a):
        """Hubble function at scale factor a, in internal units (same as self.Hubble)."""
        h = self.OmegaLambda
        h += self.omega_fld(a)
        h += self.OmegaK / (a * a)
        h += (self.OmegaCDM + self.OmegaBaryon) / a ** 3
        if self.RadiationOn:
            h += self.OmegaG / a ** 4
            h += self.get_omega_nu(a)
        else:
            h += self.get_omega_nu(1)
        h += self.Omega_ur / a ** 4
        return self.Hubble * math.sqrt(h)

    def _growth(self, a):
        """Solve the linear growth ODE (Peacock 1999). Returns (D, dD/da), arbitrarily normalised.

        D'' + a'/a D' - 1.5 (a'/a)^2 D = 0, with F = a^3 H dD/da, dF/da = 1.5 a H D.
        Neutrinos are not included in the gravitating part as they free-stream."""
        omegam = self.OmegaCDM + self.OmegaBaryon

        def ode(a, yy):
            hub = self.hubble_function(a) / self.Hubble
            return [yy[1] / a ** 3 / hub, yy[0] * 1.5 * a * omegam / a ** 3 / hub]

        # Start early to avoid lambda.
        curtime = 1e-5
        if a < curtime:
            curtime = a / 10
        yinit = [1.5 * omegam / curtime ** 2,
                 curtime ** 3 * self.hubble_function(curtime) / self.Hubble * 1.5 * omegam / curtime ** 3]
        if self.RadiationOn:
            yinit[0] += self.OmegaG / curtime ** 4 + self.get_omega_nu(curtime)
        sol = solve_ivp(ode, (curtime, a), yinit, method="RK45", rtol=1e-8, atol=1e-8)
        if not sol.success:
            raise RuntimeError("Growth function integration failed: %s" % sol.message)
        D, F = sol.y[0, -1], sol.y[1, -1]
        dDda = F / a ** 3 / (self.hubble_function(a) / self.Hubble)
        return D, dDda

    def GrowthFactor(self, astart, aend):
        """Ratio of linear growth factors D(astart) / D(aend)."""
        return self._growth(astart)[0] / self._growth(aend)[0]

    def F_Omega(self, a):
        """Zel'dovich velocity prefactor f = dlnD / dlna."""
        D, dDda = self._growth(a)
        return a / D * dDda
