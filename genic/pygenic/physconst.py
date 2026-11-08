"""Physical constants in cgs units, copied from libgadget/physconst.h."""

GRAVITY = 6.672e-8
LIGHTCGS = 2.99792458e10
CM_PER_MPC = 3.085678e24
HUBBLE = 3.2407789e-18  # in h/sec
STEFAN_BOLTZMANN = 5.670373e-5
# Neutrino temperature in units of the CMB temperature
TNUCMB = (4 / 11.) ** (1 / 3.) * 1.00328


def message(*args):
    """Print a progress message (the analogue of message(0, ...) in the C code)."""
    print(*args, flush=True)
