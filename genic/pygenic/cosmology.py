"""Background cosmology: Hubble function and growth factor.

Port of libgadget/cosmology.c, without neutrinos beyond the massless radiation background."""

import math

from scipy.integrate import solve_ivp

from .physconst import GRAVITY, HUBBLE, LIGHTCGS, STEFAN_BOLTZMANN, TNUCMB


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
        # Radiation density today in three massless neutrino species, which MP-Gadget always
        # includes in its background: 3 x 7/8 x (T_nu / T_cmb)^4 x Omega_g.
        self.OmegaNuRad = 3 * 7. / 8. * TNUCMB ** 4 * self.OmegaG
        self.OmegaCDM = self.Omega0 - self.OmegaBaryon
        self.OmegaK = 1.0 - self.Omega0 - self.OmegaLambda - self.Omega_fld
        if self.use_class_radiation_convention:
            self.OmegaK = (1.0 - self.OmegaCDM - self.OmegaBaryon - self.OmegaLambda - self.Omega_fld
                           - self.OmegaG - self.OmegaNuRad)

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
            h += (self.OmegaG + self.OmegaNuRad) / a ** 4
        else:
            # As in the C code, the neutrino background still contributes its value today.
            h += self.OmegaNuRad
        return self.Hubble * math.sqrt(h)

    def _growth(self, a):
        """Solve the linear growth ODE (Peacock 1999). Returns (D, dD/da), arbitrarily normalised.

        D'' + a'/a D' - 1.5 (a'/a)^2 D = 0, with F = a^3 H dD/da, dF/da = 1.5 a H D."""
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
            yinit[0] += (self.OmegaG + self.OmegaNuRad) / curtime ** 4
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
