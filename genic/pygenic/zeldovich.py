"""Gaussian random field and Zel'dovich displacements. Port of libgenic/zeldovich.c."""

import numpy as np

from . import mesh
from .physconst import message


class ParticleGrid:
    """Positions and IDs of a regular Ngrid^3 lattice (the IDGenerator of the C code).

    Particles are ordered with x outermost and z innermost, so that particle i has
    ID = i + 1 (plus a per-type offset) for any number of processors."""

    def __init__(self, Ngrid, BoxSize):
        self.Ngrid = Ngrid
        self.BoxSize = BoxSize
        self.NumPart = Ngrid ** 3

    def positions(self, shift=0.):
        idx = np.arange(self.Ngrid) * (self.BoxSize / self.Ngrid)
        pos = np.empty((self.Ngrid,) * 3 + (3,))
        pos[..., 0] = idx[:, None, None]
        pos[..., 1] = idx[None, :, None]
        pos[..., 2] = idx[None, None, :]
        return pos.reshape(-1, 3) + shift

    def ids(self, first_id=0):
        return np.arange(1, self.NumPart + 1, dtype=np.uint64) + np.uint64(first_id)


class ICParticles:
    """Structure-of-arrays version of struct ic_part_data."""

    def __init__(self, pos, mass):
        n = len(pos)
        self.Pos = np.array(pos, dtype=np.float64)
        self.PrePos = self.Pos.copy()
        self.Vel = np.zeros((n, 3), dtype=np.float32)
        self.Disp = np.zeros((n, 3), dtype=np.float32)
        self.Density = np.zeros(n, dtype=np.float32)
        self.Mass = np.full(n, mass, dtype=np.float64)

    def __len__(self):
        return len(self.Pos)

    @classmethod
    def concatenate(cls, parts):
        out = cls.__new__(cls)
        for attr in ("Pos", "PrePos", "Vel", "Disp", "Density", "Mass"):
            setattr(out, attr, np.concatenate([getattr(p, attr) for p in parts]))
        return out

    def split(self, n):
        """Split into two particle sets, of the first n and remaining particles."""
        first, second = (ICParticles.__new__(ICParticles) for _ in range(2))
        for attr in ("Pos", "PrePos", "Vel", "Disp", "Density", "Mass"):
            value = getattr(self, attr)
            setattr(first, attr, value[:n])
            setattr(second, attr, value[n:])
        return first, second


def setup_grid(idgen, shift, mass):
    """Unperturbed particles on a regular lattice offset by shift."""
    return ICParticles(idgen.positions(shift), mass)


def gaussian_fill(nmesh, seed, unitary_amplitude=True, invert_phase=False):
    """A complex Gaussian white noise field in Fourier space, with <|delta_k|^2> = 1.

    The field is the FFT of real-space white noise, so it is automatically Hermitian.
    With unitary_amplitude every mode has |delta_k| = 1 ('fixed' simulations); with
    invert_phase the sign is flipped ('paired' simulations).

    NOTE: the C code uses the N-GenIC GSL seed table, so the phases differ from MP-GenIC
    for the same seed. The statistics are identical."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((nmesh,) * 3)
    delta_k = mesh.rfftn(white) / np.sqrt(nmesh ** 3)
    del white
    if unitary_amplitude:
        amp = np.abs(delta_k)
        amp[amp == 0] = 1
        delta_k /= amp
    if invert_phase:
        delta_k *= -1
    delta_k[0, 0, 0] = 0
    return delta_k


class DisplacementField:
    """Computes Zel'dovich displacements and velocities for each particle species.

    The white noise field is generated once and shared by all species, as in the C code
    (which regenerates the same field from the same seed for each species)."""

    def __init__(self, nmesh, boxsize, seed, unitary_amplitude, invert_phase):
        self.nmesh = nmesh
        self.boxsize = boxsize
        self.delta_k = gaussian_fill(nmesh, seed, unitary_amplitude, invert_phase)
        kx, ky, kz = mesh.kgrid(nmesh)
        self.kint = (kx, ky, kz)
        k2 = kx ** 2 + ky ** 2 + kz ** 2
        k2[0, 0, 0] = 1
        self.k2 = k2
        # Physical wavenumber of each mode, in internal units
        self.kmag = np.sqrt(k2) * 2 * np.pi / boxsize

    def _to_particles(self, field_k, pos):
        """Inverse FFT (unnormalised, like the C code) and CIC readout to particles."""
        field_k[0, 0, 0] = 0
        real = mesh.irfftn(field_k, self.nmesh) * self.nmesh ** 3
        return mesh.cic_readout(real, pos, self.boxsize)

    def displacement_fields(self, icp, ptype, power, cosmo, config):
        """Displace particles and set their velocities, as displacement_fields() in C.

        The displacement is psi(k) = i k / k^2 delta(k), where delta(k) has the amplitude of
        the input spectrum for this particle type. The velocity is a H f psi, where f is either
        the scale-independent F_Omega or the tabulated scale-dependent growth rate.

        Note that velocities are NOT taken from the velocity transfer functions, because
        these are in synchronous (CLASS) or Newtonian (CAMB) gauge but N-body codes need
        N-body gauge, obtained by differentiating the synchronous gauge density.
        See arxiv:1505.04756."""
        nmesh = self.nmesh
        boxsize = self.boxsize
        pos = icp.Pos
        vol_norm = np.sqrt(boxsize ** 3)

        # Density, smoothed in k space by a gaussian kernel of 1 mesh grid (diagnostic only)
        amp = power.DeltaSpec(self.kmag, ptype) / vol_norm
        fac = np.exp(-self.k2 / nmesh ** 2) * amp
        icp.Density[:] = self._to_particles(self.delta_k * fac, pos)

        # Displacements: i k_axis / k^2 delta(k)
        kfac = boxsize / (2 * np.pi) / self.k2
        for d in range(3):
            icp.Disp[:, d] = self._to_particles(1j * self.kint[d] * kfac * amp * self.delta_k, pos)

        # Velocity pre-factors
        hubble_a = cosmo.hubble_function(config.TimeIC)
        vel_prefac = config.TimeIC * hubble_a
        if config.UsePeculiarVelocity:
            message("Producing Peculiar Velocity in the output.")
        else:
            # converts to Gadget velocity
            vel_prefac /= np.sqrt(config.TimeIC)

        fom = cosmo.F_Omega(config.TimeIC)
        if config.PowerP.ScaleDepVelocity:
            growth = power.dlogGrowth(self.kmag, ptype) / vol_norm
            vel = np.empty_like(icp.Disp)
            for d in range(3):
                vel[:, d] = self._to_particles(1j * self.kint[d] * kfac * growth * self.delta_k, pos)
        else:
            vel_prefac *= fom
            vel = icp.Disp

        icp.Vel[:] = vel * vel_prefac
        icp.Pos += icp.Disp
        icp.Pos %= boxsize

        maxdisp = float(np.max(icp.Disp)) if len(icp) else 0.
        maxvel = float(np.max(np.sum(icp.Vel.astype(np.float64) ** 2, axis=1))) if len(icp) else 0.
        message("Type = %d max disp = %g in units of cell sep %g"
                % (ptype, maxdisp, maxdisp / (boxsize / nmesh)))
        message("Max vel=%g km/s, vel_prefac= %g  hubble_a=%g fom=%g"
                % (np.sqrt(maxvel), vel_prefac, hubble_a, fom))
        return icp
