"""Physical constants in cgs units, copied from libgadget/physconst.h."""

GRAVITY = 6.672e-8
LIGHTCGS = 2.99792458e10
CM_PER_MPC = 3.085678e24
HUBBLE = 3.2407789e-18  # in h/sec
BOLEVK = 8.61734e-5  # Boltzmann constant in eV/K
STEFAN_BOLTZMANN = 5.670373e-5
HBAR = 6.582119e-16  # hbar in eV s
# Neutrino temperature in units of the CMB temperature
TNUCMB = (4 / 11.) ** (1 / 3.) * 1.00328
NUSPECIES = 3


def message(*args):
    """Print a progress message (the analogue of message(0, ...) in the C code)."""
    print(*args, flush=True)
