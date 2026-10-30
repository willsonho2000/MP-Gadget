# pygenic: a Python port of MP-GenIC

`pygenic` is a pure Python (numpy + scipy) version of MP-GenIC, the MP-Gadget
initial condition generator. It reads the same `paramfile.genic` files and writes
the same BigFile snapshot layout, so its output can be used directly by MP-Gadget.

It runs in a single process, without MPI or pfft, which makes it useful for small
boxes, tests, teaching and for experimenting with the algorithm. For production
runs with large particle numbers, use the C code.

## Usage

```bash
pip install numpy scipy bigfile
```

```bash
PYTHONPATH=genic python -m pygenic examples/dm-small/paramfile.genic
```

or from Python:

```python
import pygenic
path = pygenic.generate("paramfile.genic")
```

Relative paths in the parameter file (the power spectrum and transfer function files,
`OutputDir`) are resolved from the current directory, as in the C code.

## What it does

The steps are the same as in `genic/main.c`:

1. Read the parameter file (`params.py`) and set up the background cosmology:
   Hubble function, linear growth factor and growth rate, and the massive neutrino
   density (`cosmology.py`).
2. Read the input power spectrum (or use the Eisenstein & Hu fit) and the CLASS
   transfer functions, and normalise to `Sigma8` / the starting redshift (`power.py`).
3. Lay down unperturbed particles on a grid, or as a glass made by evolving a jittered
   grid under repulsive gravity (`zeldovich.py`, `glass.py`). CDM, gas and neutrino
   grids are offset from each other in proportion to their mass.
4. Draw a Gaussian white noise field in Fourier space, multiply by
   `i k / k^2 sqrt(P(k) / V)` for each species and FFT back to get the Zel'dovich
   displacement field, which is CIC-interpolated to the particles. Velocities are
   `a H(a) f psi`, with either the scale-independent `f = dlnD/dlna` or the scale-dependent
   growth rate from the transfer functions (`zeldovich.py`).
5. Add Fermi-Dirac thermal velocities to neutrino (and WDM) particles (`thermal.py`).
6. Write the header, the input power spectrum and transfer tables and the particles
   (`Position`, `Velocity`, `ID`, `ICDensity`, `PrePosition`) as a BigFile (`save.py`).

All parameters of the C code are accepted. `MaxMemSizePerNode`, `NumWriters` and
`ShowBacktrace` are ignored.

## Differences from the C code

* **Random phases.** The C code draws the Gaussian field with the N-GenIC GSL seed
  table, and the glass and thermal velocities with GSL generators. pygenic uses numpy's
  random number generator, so the same `Seed` gives a statistically equivalent but
  different realisation. Everything else that is deterministic agrees with the C code:
  the stored power and transfer tables, particle masses, IDs, unperturbed positions, the
  growth rate and velocity prefactors, and the measured power spectra.
* **Neutrino transfer functions.** When reading CLASS transfer functions with massive
  neutrinos, the C code (`parse_transfer` in `libgenic/power.c`) overwrites instead of
  accumulating the omega-weighted sum over neutrino species, and reads the neutrino
  velocity `t_ncdm` one column early (it reads `t_ur` for the first species). For three
  degenerate neutrinos this makes `DELTA_NU` a third of its true value. pygenic uses the
  intended weighted sum.
* **Header `Seed`.** The C code writes the 32-bit seed as an 8-byte integer, so the
  stored value has garbage in the upper bytes. pygenic stores the seed itself.

## Tests

```bash
PYTHONPATH=genic python -m pytest genic/pygenic/tests
```
