"""Glass particle loads. Port of libgenic/glass.c.

A glass is made by jittering a grid and evolving it under inverted (repulsive) gravity
with a damping term, which removes the grid's anisotropic structure."""

import numpy as np

from . import mesh
from .physconst import message
from .zeldovich import ICParticles


def _diff_kernel(w):
    """Fourier-space finite difference kernel (super lanczos, Hamming's Digital Filters ch. 6).

    The same as GADGET-2's 4-point gradient: c1 = 2 / 3, c2 = 1 / 12."""
    return 1 / 6.0 * (8 * np.sin(w) - np.sin(2 * w))


def glass_force(nmesh, boxsize, icp):
    """Compute the inverted-gravity force on each particle and store it in icp.Disp.

    pot(k) = delta(k) / k^2, force = -grad pot. In these units the oscillation
    period of a perturbation is 2 pi."""
    rho = mesh.cic_paint(icp.Pos, icp.Mass, boxsize, nmesh)
    totmass = np.sum(icp.Mass)
    # Overdensity in Fourier space (the mean mass per cell is totmass / nmesh^3)
    delta_k = mesh.rfftn(rho) * (nmesh ** 3 / totmass)
    del rho
    kx, ky, kz = mesh.kgrid(nmesh)
    k2 = (kx ** 2 + ky ** 2 + kz ** 2) * (2 * np.pi / boxsize) ** 2
    k2[0, 0, 0] = 1
    pot_k = delta_k / k2
    pot_k[0, 0, 0] = 0
    del delta_k, k2
    for d, kint in enumerate((kx, ky, kz)):
        fac = -1 * _diff_kernel(kint * (2 * np.pi / nmesh)) * (nmesh / boxsize)
        force = mesh.irfftn(1j * fac * pot_k, nmesh)
        icp.Disp[:, d] = mesh.cic_readout(force, icp.Pos, boxsize)


def _glass_stats(icp):
    n = len(icp)
    disp2 = np.sum(icp.Disp.astype(np.float64) ** 2)
    vel2 = np.sum(icp.Vel.astype(np.float64) ** 2)
    message("Force std = %g, vel std = %g" % (np.sqrt(disp2 / n), np.sqrt(vel2 / n)))


def glass_evolve(nmesh, boxsize, nsteps, icp):
    """Leap-frog (K D F K) evolution with inverted gravity and damping.

    With dt = pi / 2 there are 4 steps per oscillation, and 14 steps end close to a minimum.
    The setup was studied in
    https://github.com/rainwoodman/fastpm-python/blob/1be020b/Example/Glass.ipynb"""
    t_x = 0.
    glass_force(nmesh, boxsize, icp)
    dt = np.pi / 2
    hdt = 0.5 * dt
    for step in range(nsteps):
        # Kick, minding the damping term
        icp.Vel += (icp.Disp - icp.Vel) * hdt
        t_x += hdt
        # Drift
        icp.Pos += icp.Vel * dt
        icp.Pos %= boxsize
        glass_force(nmesh, boxsize, icp)
        t_f = t_x
        # Kick
        icp.Vel += (icp.Disp - icp.Vel) * hdt
        t_x += hdt
        message("Generating glass, step = %d, t_f= %g, t_x = %g" % (step, t_f / (2 * np.pi), t_x / (2 * np.pi)))
        _glass_stats(icp)
    icp.Vel[:] = 0
    icp.Disp[:] = 0
    return icp


def setup_glass(idgen, nmesh, seed, mass):
    """A glass with the same number of particles as idgen's grid."""
    rng = np.random.default_rng(seed)
    pos = idgen.positions(0)
    # A spread of 3 cells kills most of the grid anisotropy structure, while still being local.
    pos += idgen.BoxSize / idgen.Ngrid * 3 * (rng.uniform(size=pos.shape) - 0.5)
    pos %= idgen.BoxSize
    icp = ICParticles(pos, mass)
    return glass_evolve(nmesh, idgen.BoxSize, 14, icp)
