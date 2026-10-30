"""A pure Python (numpy + scipy) port of MP-GenIC, the MP-Gadget initial condition generator.

The structure mirrors the C code:

  params.py     <- genic/params.c            (parameter file parsing)
  cosmology.py  <- libgadget/cosmology.c,
                   libgadget/omega_nu_single.c (background, growth, neutrino density)
  power.py      <- libgenic/power.c          (input power spectrum and transfer functions)
  thermal.py    <- libgenic/thermal.c        (Fermi-Dirac thermal velocities)
  mesh.py       <- libgadget/petapm.c        (CIC paint/readout and FFT helpers)
  zeldovich.py  <- libgenic/zeldovich.c      (Gaussian field and Zel'dovich displacements)
  glass.py      <- libgenic/glass.c          (glass particle load)
  save.py       <- libgenic/save.c           (BigFile output)
  main.py       <- genic/main.c              (driver)

Run with ``python -m pygenic paramfile.genic`` (with ``genic/`` on PYTHONPATH),
or call ``pygenic.generate("paramfile.genic")``.
"""

__all__ = ["generate"]

from .main import generate
