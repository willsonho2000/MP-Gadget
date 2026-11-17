"""Particle-mesh helpers: Fourier grids, CIC painting and CIC readout.

These replace the parts of libgadget/petapm.c that GenIC uses. The mesh is a single
in-memory numpy array; mesh point (i, j, k) sits at position (i, j, k) * BoxSize / Nmesh."""

import numpy as np
import scipy.fft

# Number of particles processed at once, to bound temporary memory.
CHUNK = 1 << 20


def rfftn(field):
    return scipy.fft.rfftn(field, workers=-1)


def irfftn(field_k, nmesh):
    return scipy.fft.irfftn(field_k, s=(nmesh,) * 3, workers=-1)


def kgrid(nmesh):
    """Integer wavenumbers along each axis, broadcastable to the rfftn output shape."""
    kx = np.fft.fftfreq(nmesh, d=1. / nmesh).reshape(-1, 1, 1)
    ky = np.fft.fftfreq(nmesh, d=1. / nmesh).reshape(1, -1, 1)
    kz = np.fft.rfftfreq(nmesh, d=1. / nmesh).reshape(1, 1, -1)
    return kx, ky, kz


def _cic_weights(pos, boxsize, nmesh):
    """Yield (flat mesh index, weight) for the 8 cells touched by each particle."""
    x = pos * (nmesh / boxsize)
    i0 = np.floor(x).astype(np.int64)
    f = x - i0
    for dx in (0, 1):
        wx = f[:, 0] if dx else 1 - f[:, 0]
        ix = (i0[:, 0] + dx) % nmesh
        for dy in (0, 1):
            wy = f[:, 1] if dy else 1 - f[:, 1]
            iy = (i0[:, 1] + dy) % nmesh
            for dz in (0, 1):
                wz = f[:, 2] if dz else 1 - f[:, 2]
                iz = (i0[:, 2] + dz) % nmesh
                yield (ix * nmesh + iy) * nmesh + iz, wx * wy * wz


def cic_paint(pos, mass, boxsize, nmesh):
    """Cloud-in-cell mass assignment of particles onto a periodic mesh."""
    mesh = np.zeros(nmesh ** 3)
    mass = np.broadcast_to(mass, (len(pos),))
    for start in range(0, len(pos), CHUNK):
        sl = slice(start, start + CHUNK)
        for idx, w in _cic_weights(pos[sl], boxsize, nmesh):
            mesh += np.bincount(idx, weights=w * mass[sl], minlength=nmesh ** 3)
    return mesh.reshape((nmesh,) * 3)


def cic_readout(mesh, pos, boxsize):
    """Cloud-in-cell interpolation of a periodic mesh to particle positions."""
    nmesh = mesh.shape[0]
    flat = mesh.reshape(-1)
    out = np.zeros(len(pos))
    for start in range(0, len(pos), CHUNK):
        sl = slice(start, start + CHUNK)
        for idx, w in _cic_weights(pos[sl], boxsize, nmesh):
            out[sl] += w * flat[idx]
    return out
