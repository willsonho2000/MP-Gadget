# pygenic: a Python port of MP-GenIC

`pygenic` is a pure Python (numpy + scipy) version of MP-GenIC, the MP-Gadget
initial condition generator. It reads the same `paramfile.genic` files and writes
the same BigFile snapshot layout, so its output can be used directly by MP-Gadget.
It makes CDM and gas only: there are no massive neutrinos (neutrino particles, neutrino
masses or neutrino transfer functions).

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
   Hubble function, linear growth factor and growth rate (`cosmology.py`). As in
   MP-Gadget, the radiation background includes three massless neutrino species.
2. Read the input power spectrum (or use the Eisenstein & Hu fit) and the CLASS
   transfer functions, and normalise to `Sigma8` / the starting redshift (`power.py`).
3. Lay down unperturbed particles on a grid, or as a glass made by evolving a jittered
   grid under repulsive gravity (`zeldovich.py`, `glass.py`). The CDM and gas
   grids are offset from each other in proportion to their mass.
4. Draw a Gaussian white noise field in Fourier space, multiply by
   `i k / k^2 sqrt(P(k) / V)` for each species and FFT back to get the Zel'dovich
   displacement field, which is CIC-interpolated to the particles. Velocities are
   `a H(a) f psi`, with either the scale-independent `f = dlnD/dlna` or the scale-dependent
   growth rate from the transfer functions (`zeldovich.py`).
5. Write the header, the input power spectrum and transfer tables and the particles
   (`Position`, `Velocity`, `ID`, `ICDensity`, `PrePosition`) as a BigFile (`save.py`).

Parameters that only matter to the C code (`MaxMemSizePerNode`, `NumWriters`,
`ShowBacktrace`, `PrimordialAmp`, `PrimordialRunning`) and the neutrino parameters
(`MNue`, `MNum`, `MNut`, `NgridNu`, `Max_nuvel`, `Omega_ur`) are not accepted: as for any
unknown parameter, pygenic stops with an error, so remove them from the parameter file.
The same goes for `MWDM_therm` (no WDM thermal velocities; a WDM power spectrum
still works), `InvertPhase` (no paired simulations) and `UsePeculiarVelocity`:
velocities are always written as peculiar velocities `v = a dx/dt`, and the header
says `UsePeculiarVelocity = 1` so that MP-Gadget reads them correctly.

## Baryon-CDM difference in the particle masses (`PerturbMass`)

With gas and `DifferentTransferFunctions = 1`, baryons and CDM start with different
density fields. `PerturbMass` chooses how that difference is set up:

* `PerturbMass = 0` (default): each species is displaced with its own transfer function.
  On two grids this couples the species spuriously, which is why the gas is a glass by
  default.
* `PerturbMass = 1`: the method of Hahn, Rampf & Uhlemann (2021, arXiv:2008.09124).
  Both species are on grids and get the same displacement and velocity, those of the
  total matter. The difference `delta_bc = delta_b - delta_cdm` at the unperturbed
  position `q` goes into the particle masses, with `f_b = OmegaBaryon / Omega0` and
  `f_c = 1 - f_b`:

  ```
  m_gas(q) = <m_gas> (1 + f_c delta_bc(q))
  m_cdm(q) = <m_cdm> (1 - f_b delta_bc(q))
  ```

Each particle type then has a `Mass` block. `MassTable` in the header stays the mean
mass, and the header has `PerturbMass = 1`.

Things to know:

* It needs `ProduceGas = 1`, `DifferentTransferFunctions = 1` and a tabulated spectrum,
  and it is only available for grids: pygenic stops if a glass is requested.
* **MP-Gadget does not read these masses yet.** Its IC reader sets every particle mass
  from `MassTable` (`petaio_read_internal` in `libgadget/petaio.c`), so the baryon-CDM
  difference is lost unless the reader is changed to use the `Mass` block.
* `delta_bc` comes from the transfer functions at the starting redshift. The paper takes
  it at a late reference time (a = 0.32) instead. There is no relative velocity
  between the species, as in the paper's default.
* `delta_bc` is limited to the modes the particle grid can represent before it is
  sampled at the particles.
* When measuring the power spectrum of these ICs, weight particles by `Mass`, and do
  not use cloud-in-cell on a mesh only twice as fine as the particle grid: it responds
  differently to masses and to displacements on a grid, which biases the gas-to-CDM
  ratio by about 2% at half the particle Nyquist frequency. A mesh four times as
  fine with a TSC kernel is accurate to about 0.1% there.

## Differences from the C code

* **Random phases.** The C code draws the Gaussian field with the N-GenIC GSL seed
  table, and the glass with a GSL generator. pygenic uses numpy's
  random number generator, so the same `Seed` gives a statistically equivalent but
  different realisation. Everything else that is deterministic agrees with the C code:
  the stored power and transfer tables, particle masses, IDs, unperturbed positions, the
  growth rate and velocity prefactors, and the measured power spectra.
* **No neutrinos.** pygenic has no massive neutrinos, so the header has no
  `FractionNuInParticles` or `OmegaUR`, and `ICTransfers` has no `DELTA_NU` or `VEL_NU`
  rows. The `ncdm` columns of a CLASS transfer file are skipped.
* **`PerturbMass`.** The C code has no such option: it always perturbs positions.
* **Header `Seed`.** The C code writes the 32-bit seed as an 8-byte integer, so the
  stored value has garbage in the upper bytes. pygenic stores the seed itself.

## Tests

```bash
PYTHONPATH=genic python -m pytest genic/pygenic/tests
```
