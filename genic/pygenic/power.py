"""Input power spectrum and transfer functions. Port of libgenic/power.c.

All functions take k in internal (inverse length) units and are vectorised over k."""

import math
import os

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline

from .physconst import CM_PER_MPC, LIGHTCGS, message

# Transfer function types, matching enum TransferType in libgenic/power.h
DELTA_BAR = 0
DELTA_CDM = 1
DELTA_NU = 2
DELTA_CB = 3
VEL_BAR = 4
VEL_CDM = 5
VEL_NU = 6
VEL_CB = 7
VEL_TOT = 8
# Always unity, so there is no column for it
DELTA_TOT = 9
MAXCOLS = 9

TNAMES = ["DELTA_BAR", "DELTA_CDM", "DELTA_NU", "DELTA_CB",
          "VEL_BAR", "VEL_CDM", "VEL_NU", "VEL_CB", "VEL_TOT"]

NUGGET = 1e-30


def _read_rows(inputfile):
    """Read the non-comment rows of a whitespace-separated table.

    Returns a list of lists of strings, and the number of columns in the first row."""
    if not os.path.exists(inputfile):
        raise IOError("can't read input spectrum in file '%s'" % inputfile)
    rows = []
    with open(inputfile) as fh:
        for line in fh:
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            rows.append(fields)
    if len(rows) < 2:
        raise ValueError("Input spectrum too short")
    ncolumns = len(rows[0])
    message("Detected %d columns in file '%s'." % (ncolumns, inputfile))
    return rows, ncolumns


class PowerSpectrum:
    """The input linear power spectrum, normalised to the initial redshift.

    Port of init_powerspectrum(), DeltaSpec() and dlogGrowth()."""

    def __init__(self, InitTime, UnitLength_in_cm, cosmo, ppar):
        self.WhichSpectrum = ppar.WhichSpectrum
        self.PrimordialIndex = ppar.PrimordialIndex
        self.UnitLength_in_cm = UnitLength_in_cm
        self.scale = CM_PER_MPC / UnitLength_in_cm
        self.CP = cosmo
        self.power_logk = np.array([])
        self.power_logD = np.array([])
        self.transfer_logk = np.array([])
        self.transfer = np.zeros((MAXCOLS, 0))
        self._power_spline = None
        self._transfer_splines = None

        if ppar.WhichSpectrum == 2:
            self._read_power_table(ppar.FileWithInputSpectrum)
            if ppar.DifferentTransferFunctions or ppar.ScaleDepVelocity:
                self._init_transfer_table(InitTime, ppar.FileWithTransferFunction)

        self.Norm = 1.0
        if ppar.InputPowerRedshift >= 0 or ppar.Sigma8 > 0:
            R8 = 8 * self.scale  # 8 Mpc/h
            if ppar.Sigma8 > 0:
                res = self.TopHatSigma2(R8)
                if np.isfinite(res) and res > 0:
                    self.Norm = ppar.Sigma8 / math.sqrt(res)
                else:
                    raise ValueError("Could not normalize P(k) to Sigma8=%g! Measured Sigma8^2 is %g"
                                     % (ppar.Sigma8, res))
            Dplus = cosmo.GrowthFactor(InitTime, 1 / (1 + ppar.InputPowerRedshift))
            if ppar.InputPowerRedshift >= 0:
                self.Norm *= Dplus
                message("Growth factor from z=%g (InputPowerRedshift) to z=%g (Init): %g"
                        % (ppar.InputPowerRedshift, 1. / InitTime - 1, Dplus))
            message("Normalization adjusted to  Sigma8=%g (at z=0)  (Normfac=%g)."
                    % (math.sqrt(self.TopHatSigma2(R8)) / Dplus, self.Norm))

    # ---------------------------------------------------------------- tables
    def _read_power_table(self, inputfile):
        """Read a matter power spectrum (k [h/Mpc], P [Mpc/h]^3), as parse_power()."""
        rows, _ = _read_rows(inputfile)
        k = np.array([float(r[0]) for r in rows])
        p = np.array([float(r[1]) for r in rows])
        # If any k is negative, the file is (probably) in log10 units already.
        # The C code makes this decision on the first negative k it sees; we do the same.
        inlog = np.zeros(len(k), dtype=bool)
        neg = np.nonzero(k < 0)[0]
        if len(neg):
            message("some input k is negative, guessing the file is in log10 units")
            inlog[neg[0]:] = True
        logk = np.where(inlog, k, np.log10(np.where(inlog, 1, k)))
        logp = np.where(inlog, p, np.log10(np.where(inlog, 1, p) + NUGGET))
        self.power_logk = logk
        # Store log10 delta, the square root of power
        self.power_logD = logp / 2
        self._power_spline = CubicSpline(self.power_logk, self.power_logD, bc_type="natural")

    def _parse_transfer(self, rows, numcol, InitTime):
        """Read a CLASS transfer function file, as parse_transfer().

        Columns (1-indexed, entries in parentheses only present for a dark energy fluid):
        1:k (h/Mpc) 2:d_g 3:d_b 4:d_cdm (5:d_fld) 5:d_ur 6..:d_ncdm[i] d_tot phi psi h h_prime
        eta eta_prime t_g t_b t_ur (t_fld) t_ncdm[i] t_tot"""
        ncols = numcol - 1
        # Detect dark energy fluid perturbations.
        defld = 1 if numcol > 22 else 0
        nnu = (ncols - 15 - defld * 2) // 2
        data = np.array([[float(x) for x in r[:numcol]] for r in rows])
        if data.shape[1] < numcol:
            raise ValueError("Incomplete line in transfer function. Did you remember to set "
                             "extra metric transfer functions=y?")
        logk = np.log10(data[:, 0])
        transfers = data[:, 1:]
        out = np.zeros((MAXCOLS, len(logk)))
        # We use the formulae for velocities from fastpm in synchronous gauge:
        #  CDM = - h_prime / 2 / d_cdm
        #  bar = - (h_prime / 2  + t_b) / d_b
        #  nu = - (h_prime / 2 + t_ncdm) / d_ncdm
        # (Ma & Bertschinger 1995 eq. 42, 66), with a normalisation of (1+z)/hubble applied later.
        out[DELTA_BAR] = -1 * transfers[:, 1]
        out[DELTA_CDM] = -1 * transfers[:, 2]
        # Neutrino rows are the omega_nu weighted average over the massive species.
        # NOTE: the C code assigns (=) instead of accumulating (+=) inside this loop, and reads
        # t_ncdm one column early (t_ur for the first species). We use the intended weighted sum.
        onu = self.CP.get_omega_nu(InitTime)
        onu_single = [self.CP.ONu.omega_nu_single(InitTime, j) for j in range(nnu)]
        for j in range(nnu):
            out[DELTA_NU] += -1 * transfers[:, 4 + j + defld] * onu_single[j]
            out[VEL_NU] += transfers[:, 14 + nnu + defld * 2 + j] * onu_single[j]
        out[DELTA_NU] /= onu
        out[VEL_NU] /= onu
        out[VEL_BAR] = transfers[:, 12 + nnu + defld]
        out[VEL_CDM] = transfers[:, 8 + nnu + defld] * 0.5
        return logk, out

    def _init_transfer_table(self, InitTime, inputfile):
        """Read and normalise the transfer functions, as init_transfer_table()."""
        CP = self.CP
        if len(inputfile) == 0:
            raise ValueError("Could not read transfer table at: '%s'" % inputfile)
        rows, numcol = _read_rows(inputfile)
        logk, t = self._parse_transfer(rows, numcol, InitTime)
        nnu = sum(m > 0 for m in CP.MNu)
        # Normalise the velocity transfer functions: divide by a * hubble,
        # where hubble is the hubble function in Mpc^-1, so H0/c
        fac = InitTime * CP.hubble_function(InitTime) / CP.Hubble * 100 * CP.HubbleParam / (LIGHTCGS / 1e5)
        onu = CP.get_omega_nu(InitTime) * InitTime ** 3
        # Now VEL_CDM is t_cdm
        t[VEL_CDM] /= fac
        t[VEL_BAR] /= fac
        t[VEL_NU] /= fac
        # Now change to t bar and tnu
        t[VEL_BAR] += t[VEL_CDM]
        t[VEL_NU] += t[VEL_CDM]
        # Set up the CDM + baryon rows
        t[DELTA_CB] = CP.OmegaBaryon * t[DELTA_BAR] + CP.OmegaCDM * t[DELTA_CDM]
        t[VEL_CB] = CP.OmegaBaryon * t[VEL_BAR] + CP.OmegaCDM * t[VEL_CDM]
        # Total growth and delta: start as CDM + baryon and then add nu if needed.
        t[VEL_TOT] = t[VEL_CB].copy()
        T_tot = t[DELTA_CB].copy()
        Omega0a3 = CP.OmegaBaryon + CP.OmegaCDM
        t[DELTA_CB] /= Omega0a3
        t[VEL_CB] /= Omega0a3
        if nnu > 0:
            t[VEL_TOT] += onu * t[VEL_NU]
            T_tot += onu * t[DELTA_NU]
            Omega0a3 += onu
        t[VEL_TOT] /= Omega0a3
        T_tot /= Omega0a3
        # Normalize growth_i and delta_i by delta_tot
        t /= T_tot
        self.transfer_logk = logk
        self.transfer = t
        self._transfer_splines = [CubicSpline(logk, t[i], bc_type="natural") for i in range(MAXCOLS)]

        sel = logk > self.power_logk[0]
        meangrowth = t[VEL_BAR:VEL_TOT + 1, sel].mean(axis=1) if np.any(sel) else np.zeros(5)
        message("Scale-dependent growth calculated. Mean = %s" % " ".join("%g" % m for m in meangrowth))
        message("Power spectrum rows: %d, Transfer: %d (%g -> %g)"
                % (len(self.power_logk), len(logk), t[DELTA_BAR][0], t[DELTA_BAR][-1]))

    # ------------------------------------------------------------ evaluation
    def _get_tabulated(self, k, Type):
        """Interpolate the power table, times the transfer ratio T_type / T_tot."""
        k = np.asarray(k, dtype=float)
        logk = np.log10(k * self.scale)
        logk0 = self.power_logk[0]
        maxlogk = self.power_logk[-1]
        if np.any(logk < logk0 - 0.1) or np.any(logk > maxlogk + 0.25):
            raise ValueError("Requested k = %g - %g h/Mpc for type %d but power table is from %g to %g h/Mpc"
                             % (10 ** np.min(logk), 10 ** np.max(logk), Type, 10 ** logk0, 10 ** maxlogk))
        intlogk = np.clip(logk, logk0, maxlogk)
        logD = self._power_spline(intlogk)
        # Past the end of the table, assume the power scales like k^-3 log (k)
        # and the transfer function is constant (as the C code does).
        over = logk > maxlogk
        if np.any(over):
            logD = np.where(over, logD - 3 * (logk - intlogk) + np.log(np.where(over, logk / intlogk, 1)), logD)
        trans = 1
        if self._transfer_splines is not None and DELTA_BAR <= Type < DELTA_TOT:
            trans = self._transfer_splines[Type](intlogk)
        # Convert delta from (Mpc/h)^3/2 to internal units
        logD = logD + 1.5 * np.log10(self.scale)
        delta = (10.0 ** logD - NUGGET) * trans
        if not np.all(np.isfinite(delta)):
            raise ValueError("infinite delta or growth for Type = %d" % Type)
        return delta

    def _tk_eh(self, k):
        """Eisenstein & Hu transfer function (from Martin White)."""
        CP = self.CP
        hubble = CP.HubbleParam
        omegam = CP.Omega0
        ombh2 = CP.OmegaBaryon * hubble ** 2
        if CP.OmegaBaryon == 0:
            ombh2 = 0.044 * hubble ** 2
        k = np.asarray(k) * self.scale  # convert to h/Mpc
        theta = 2.728 / 2.7
        ommh2 = omegam * hubble ** 2
        s = 44.5 * np.log(9.83 / ommh2) / np.sqrt(1. + 10. * np.exp(0.75 * np.log(ombh2))) * hubble
        a = (1. - 0.328 * np.log(431. * ommh2) * ombh2 / ommh2
             + 0.380 * np.log(22.3 * ommh2) * (ombh2 / ommh2) ** 2)
        gamma = a + (1. - a) / (1. + np.exp(4 * np.log(0.43 * k * s)))
        gamma *= omegam * hubble
        q = k * theta ** 2 / gamma
        L0 = np.log(2. * np.e + 1.8 * q)
        C0 = 14.2 + 731. / (1. + 62.5 * q)
        return L0 / (L0 + C0 * q * q)

    def Delta_EH(self, k):
        k = np.asarray(k, dtype=float)
        return np.sqrt(k * self._tk_eh(k) ** 2 * k ** (self.PrimordialIndex - 1.0))

    def DeltaSpec(self, k, Type):
        """delta (square root of the power spectrum) for transfer type Type at the initial redshift."""
        if self.WhichSpectrum == 2:
            if VEL_BAR <= Type <= VEL_TOT:
                raise ValueError("Velocity Type %d passed to Delta_Tabulated" % Type)
            power = self._get_tabulated(k, Type)
        else:
            power = self.Delta_EH(k)
        return power * self.Norm

    def dlogGrowth(self, k, Type):
        """delta times the scale-dependent growth rate f(k) = dlnD/dlna.

        Like the C code, this is not multiplied by Norm."""
        if Type < DELTA_BAR or Type > DELTA_CB:
            Type = VEL_TOT
        else:
            Type = VEL_BAR + (Type - DELTA_BAR)
        return self._get_tabulated(k, Type)

    def TopHatSigma2(self, R):
        """sigma^2 of the (current) power spectrum in a top hat of radius R."""
        def integrand(k):
            kr = R * k
            kr2 = kr * kr
            if kr < 1e-3:
                w = 1. / 3. - kr2 / 30. + kr2 * kr2 / 840.
            else:
                w = 3 * (math.sin(kr) / kr - math.cos(kr)) / kr2
            return 4 * math.pi / (2 * math.pi) ** 3 * k * k * w * w * float(self.DeltaSpec(k, DELTA_TOT)) ** 2

        # Integral is oscillatory but almost zero at kr = (n + 1/2) pi for odd n.
        maxk = math.pi * (20 + 0.5) / R
        if len(self.power_logk):
            maxtabk = 10 ** (0.24 + self.power_logk[-1])
            if maxk * self.scale > maxtabk:
                raise ValueError("Trying to do sigma8 integral for rescaling, but need k = %g "
                                 "and largest k in power table is %g"
                                 % (maxk * self.scale, 10 ** self.power_logk[-1]))
        result, _ = quad(integrand, 0, maxk, epsabs=0, epsrel=1e-4, limit=1000)
        return result
