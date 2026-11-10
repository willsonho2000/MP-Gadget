"""Glass particle loads. Port of libgenic/glass.c.

A glass is made by jittering a grid and evolving it under inverted (repulsive) gravity
with a damping term, which removes the grid's anisotropic structure."""

import numpy as np

from . import mesh
from .physconst import message
from .zeldovich import grid_positions


def _diff_kernel(w):
    """Fourier-space finite difference kernel (super lanczos, Hamming's Digital Filters ch. 6).

    The same as GADGET-2's 4-point gradient: c1 = 2 / 3, c2 = 1 / 12."""
    return 1 / 6.0 * (8 * np.sin(w) - np.sin(2 * w))


def glass_force(nmesh, boxsize, pos, mass):
    """The inverted-gravity force on each particle.

    pot(k) = delta(k) / k^2, force = -grad pot. In these units the oscillation
    period of a perturbation is 2 pi."""
    rho = mesh.cic_paint(pos, mass, boxsize, nmesh)
    # Overdensity in Fourier space (the mean mass per cell is the total mass / nmesh^3)
    delta_k = mesh.rfftn(rho) * (nmesh ** 3 / np.sum(mass))
    kx, ky, kz = mesh.kgrid(nmesh)
    k2 = (kx ** 2 + ky ** 2 + kz ** 2) * (2 * np.pi / boxsize) ** 2
    k2[0, 0, 0] = 1
    pot_k = delta_k / k2
    pot_k[0, 0, 0] = 0
    # Single precision, as in the C code
    force = np.empty((len(pos), 3), dtype=np.float32)
    for d, kint in enumerate((kx, ky, kz)):
        fac = -1 * _diff_kernel(kint * (2 * np.pi / nmesh)) * (nmesh / boxsize)
        force[:, d] = mesh.cic_readout(mesh.irfftn(1j * fac * pot_k, nmesh), pos, boxsize)
    return force


def glass_evolve(nmesh, boxsize, pos, mass, nsteps=14):
    """Leap-frog (K D F K) evolution with inverted gravity and damping.

    pos are the starting positions and mass the mass of each particle.
    Returns the evolved positions.

    With dt = pi / 2 there are 4 steps per oscillation, and 14 steps end close to a minimum.
    The setup was studied in
    https://github.com/rainwoodman/fastpm-python/blob/1be020b/Example/Glass.ipynb"""
    pos = np.array(pos, dtype=np.float64)
    vel = np.zeros((len(pos), 3), dtype=np.float32)
    force = glass_force(nmesh, boxsize, pos, mass)
    dt = np.pi / 2
    hdt = 0.5 * dt
    for step in range(nsteps):
        # Kick, minding the damping term
        vel += (force - vel) * hdt
        # Drift
        pos += vel * dt
        pos %= boxsize
        force = glass_force(nmesh, boxsize, pos, mass)
        # Kick
        vel += (force - vel) * hdt
        message("Generating glass, step = %d, force std = %g, vel std = %g"
                % (step, np.sqrt(np.sum(force.astype(np.float64) ** 2) / len(pos)),
                   np.sqrt(np.sum(vel.astype(np.float64) ** 2) / len(pos))))
    return pos


def setup_glass(ngrid, boxsize, nmesh, seed, mass):
    """Positions of a glass of ngrid^3 particles, each with the given mass."""
    rng = np.random.default_rng(seed)
    pos = grid_positions(ngrid, boxsize)
    # A spread of 3 cells kills most of the grid anisotropy structure, while still being local.
    pos += boxsize / ngrid * 3 * (rng.uniform(size=pos.shape) - 0.5)
    pos %= boxsize
    return glass_evolve(nmesh, boxsize, pos, np.full(len(pos), mass))
