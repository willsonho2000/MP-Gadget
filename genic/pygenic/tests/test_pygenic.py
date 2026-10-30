"""Tests for the Python GenIC. Run with: PYTHONPATH=genic python -m pytest genic/pygenic/tests"""

import math
import os
from types import SimpleNamespace

import numpy as np
import pytest

from pygenic import mesh, params, thermal, zeldovich
from pygenic.cosmology import Cosmology
from pygenic.main import generate
from pygenic.power import DELTA_TOT, PowerSpectrum

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples")

BASE_PARAMS = """
OutputDir = {outdir}
FileBase = IC
Ngrid = {ngrid}
BoxSize = {box}
Omega0 = 0.288
OmegaLambda = 0.712
OmegaBaryon = 0.0472
HubbleParam = 0.7
ProduceGas = {gas}
Redshift = {redshift}
Seed = 42
FileWithInputSpectrum = {pk}
WhichSpectrum = {which}
DifferentTransferFunctions = 0
"""


def write_params(tmp_path, ngrid=16, box=64000., gas=0, which=1, pk="unused", redshift=49, extra=""):
    text = BASE_PARAMS.format(outdir=tmp_path / "out", ngrid=ngrid, box=box, gas=gas, which=which, pk=pk,
                              redshift=redshift) + extra
    fname = tmp_path / "paramfile.genic"
    fname.write_text(text)
    return str(fname)


def make_cosmo(**kw):
    cp = SimpleNamespace(Omega0=0.3, OmegaLambda=0.7, OmegaBaryon=0.05, HubbleParam=0.7, Omega_fld=0,
                         w0_fld=-1, wa_fld=0, Omega_ur=0, use_class_radiation_convention=0,
                         CMBTemperature=2.7255, RadiationOn=0, MNu=[0, 0, 0])
    for key, value in kw.items():
        setattr(cp, key, value)
    return Cosmology(cp, 0.01, 3.085678e21, 1.989e43, 1e5)


def test_parameter_parsing():
    text = """# comment line
    Ngrid = 32   # trailing comment
    BoxSize=1000
    OutputDir = out
    FileBase = IC
    Omega0 = 0.3
    OmegaBaryon = 0.04
    OmegaLambda = 0.7
    HubbleParam = 0.7
    ProduceGas = 1
    Redshift = 99
    Seed = 5
    FileWithInputSpectrum = pk.txt
    """
    ps = params.read_parameters(text)
    assert ps["Ngrid"] == 32
    assert ps["BoxSize"] == 1000.
    assert ps["NgridGas"] == -1
    assert ps["FileWithInputSpectrum"] == "pk.txt"
    with pytest.raises(ValueError):
        params.read_parameters(text + "NotAParameter = 1\n")
    with pytest.raises(ValueError):
        params.read_parameters("Ngrid = 32\n")


def test_einstein_de_sitter_growth():
    cosmo = make_cosmo(Omega0=1.0, OmegaLambda=0.0, OmegaBaryon=0.0)
    assert cosmo.F_Omega(0.5) == pytest.approx(1.0, rel=1e-4)
    assert cosmo.GrowthFactor(0.25, 0.5) == pytest.approx(0.5, rel=1e-4)
    # Massless neutrinos contribute Omega_nu(1) ~ 3e-5 even without radiation, as in the C code.
    assert cosmo.hubble_function(0.25) == pytest.approx(cosmo.Hubble * 8, rel=1e-5)


def test_lcdm_growth_rate():
    cosmo = make_cosmo()
    # f ~ Omega_m(a)^0.55 for LCDM
    omega_m = 0.3 / (0.3 + 0.7)
    assert cosmo.F_Omega(1.0) == pytest.approx(omega_m ** 0.55, rel=2e-2)


def test_massive_neutrino_density():
    cosmo = make_cosmo(MNu=[0.1, 0.1, 0.1], RadiationOn=1)
    # Today the neutrinos are non-relativistic: Omega_nu h^2 ~ sum m_nu / 93.14 eV
    assert cosmo.get_omega_nu(1) * 0.7 ** 2 == pytest.approx(0.3 / 93.14, rel=1e-2)
    assert cosmo.ONu.nu_degeneracies == [3, 0, 0]


def test_sigma8_normalisation():
    cosmo = make_cosmo()
    ppar = SimpleNamespace(WhichSpectrum=1, PrimordialIndex=0.97, Sigma8=0.8, InputPowerRedshift=0,
                           DifferentTransferFunctions=0, ScaleDepVelocity=0)
    power = PowerSpectrum(1.0, 3.085678e21, cosmo, ppar)
    assert math.sqrt(power.TopHatSigma2(8000.)) == pytest.approx(0.8, rel=1e-4)


def test_thermal_velocity_distribution():
    therm = thermal.ThermalVel(1.0, 17.0, 0)
    assert therm.total_frac == pytest.approx(1.0, abs=1e-5)
    vel = therm.add_thermal_speeds(np.random.default_rng(1), np.zeros((200000, 3)))
    # <v> for a relativistic Fermi-Dirac distribution is 7 pi^4 / (180 zeta(3)) ~ 3.15
    assert np.mean(np.linalg.norm(vel, axis=1)) == pytest.approx(7 * np.pi ** 4 / (180 * 1.2020569), rel=1e-2)


def test_cic_paint_readout():
    pos = np.random.default_rng(3).uniform(0, 10., size=(1000, 3))
    rho = mesh.cic_paint(pos, 2.0, 10., 8)
    assert np.sum(rho) == pytest.approx(2000.)
    field = np.arange(8.)[:, None, None] * np.ones((8, 8, 8))
    # A linear field is interpolated exactly away from the periodic boundary
    inside = pos[:, 0] < 10. * 7 / 8
    assert np.allclose(mesh.cic_readout(field, pos, 10.)[inside], pos[inside, 0] * 8 / 10.)


def test_gaussian_field():
    dk = zeldovich.gaussian_fill(16, 7, unitary_amplitude=True)
    assert np.allclose(np.abs(dk[1:, 1:, 1:]), 1)
    dk2 = zeldovich.gaussian_fill(16, 7, unitary_amplitude=True, invert_phase=True)
    assert np.allclose(dk, -dk2)


def measure_power(pos, box, nmesh):
    """Power spectrum of particles, deconvolved for CIC, binned in integer k."""
    rho = mesh.cic_paint(pos, 1.0, box, nmesh)
    dk = mesh.rfftn(rho / rho.mean() - 1) / nmesh ** 3
    kx, ky, kz = mesh.kgrid(nmesh)
    window = (np.sinc(kx / nmesh) * np.sinc(ky / nmesh) * np.sinc(kz / nmesh)) ** 2
    pk = np.abs(dk / window) ** 2 * box ** 3
    kint = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2).ravel()
    bins = np.rint(kint).astype(int)
    pk_binned = np.bincount(bins, pk.ravel()) / np.maximum(np.bincount(bins), 1)
    kmean = np.bincount(bins, kint) / np.maximum(np.bincount(bins), 1)
    return kmean * 2 * np.pi / box, pk_binned


def test_end_to_end_power_and_velocities(tmp_path):
    bigfile = pytest.importorskip("bigfile")
    pfile = write_params(tmp_path, ngrid=32, box=128000., extra="Sigma8 = 0.8\nPrimordialIndex = 0.97\n")
    outpath = generate(pfile, verbose=False)
    f = bigfile.File(outpath)
    header = f["Header"].attrs
    assert header["TotNumPart"][1] == 32 ** 3
    pos = f["1/Position"][:]
    prepos = f["1/PrePosition"][:]
    vel = f["1/Velocity"][:]
    ids = f["1/ID"][:]
    assert np.array_equal(ids, np.arange(1, 32 ** 3 + 1))
    box = header["BoxSize"][0]
    disp = (pos - prepos + box / 2) % box - box / 2

    config, cp = params.read_parameterfile(pfile, verbose=False)
    cosmo = Cosmology(cp, config.TimeIC, config.UnitLength_in_cm, config.UnitMass_in_g,
                      config.UnitVelocity_in_cm_per_s)
    # Zel'dovich velocity: v = a H(a) f(a) psi
    prefac = config.TimeIC * cosmo.hubble_function(config.TimeIC) * cosmo.F_Omega(config.TimeIC)
    assert np.allclose(vel, disp * prefac, rtol=1e-4, atol=1e-4 * np.abs(vel).max())

    # Measured power agrees with the input on large scales (unitary amplitudes: no cosmic variance)
    power = PowerSpectrum(config.TimeIC, config.UnitLength_in_cm, cosmo, config.PowerP)
    # (measured on a finer mesh than the particle grid to limit aliasing)
    k, pk = measure_power(pos, box, 64)
    sel = slice(1, 8)
    ratio = pk[sel] / power.DeltaSpec(k[sel], DELTA_TOT) ** 2
    assert np.allclose(ratio, 1, atol=0.07)


@pytest.mark.skipif(not os.path.exists(os.path.join(EXAMPLES, "neutrinos", "class_tk_nu_99.dat")),
                    reason="needs the example CLASS tables")
def test_gas_glass_and_neutrinos(tmp_path):
    bigfile = pytest.importorskip("bigfile")
    nudir = os.path.abspath(os.path.join(EXAMPLES, "neutrinos"))
    extra = ("DifferentTransferFunctions = 1\nMakeGlassGas = 1\nNgridNu = 8\nRadiationOn = 1\n"
             "MNue = 0.1\nMNum = 0.1\nMNut = 0.1\nFileWithTransferFunction = %s\n"
             % os.path.join(nudir, "class_tk_nu_99.dat"))
    # The transfer functions are at z = 99
    pfile = write_params(tmp_path, ngrid=16, box=256000., gas=1, which=2, redshift=99,
                         pk=os.path.join(nudir, "class_pk_nu_99.dat"), extra=extra)
    outpath = generate(pfile, verbose=False)
    f = bigfile.File(outpath)
    header = f["Header"].attrs
    assert list(header["TotNumPart"][:3]) == [16 ** 3, 16 ** 3, 8 ** 3]
    assert 0 < header["FractionNuInParticles"][0] < 1
    assert header["MassTable"][2] > 0
    # IDs are unique across types
    ids = np.concatenate([f["%d/ID" % t][:] for t in range(3)])
    assert len(np.unique(ids)) == len(ids)
    # Neutrinos have large thermal velocities compared to the CDM
    assert np.std(f["2/Velocity"][:]) > 10 * np.std(f["1/Velocity"][:])
    # Glass gas is not on a grid
    prepos = f["0/PrePosition"][:]
    cell = 256000. / 16
    assert np.std((prepos / cell) % 1) > 0.2
    assert f["ICTransfers"].attrs["Nentry"][0] > 0
    assert np.all(np.isfinite(f["ICTransfers/%s" % "DELTA_NU"][:]))


def test_wdm_paired_and_gadget_velocities(tmp_path):
    bigfile = pytest.importorskip("bigfile")
    base = "Sigma8 = 0.8\nUsePeculiarVelocity = 0\nPrePosGridCenter = 1\n"
    paths = []
    for i, extra in enumerate(["", "InvertPhase = 1\n", "MWDM_therm = 0.1\n"]):
        sub = tmp_path / str(i)
        sub.mkdir()
        paths.append(generate(write_params(sub, ngrid=8, extra=base + extra), verbose=False))
    files = [bigfile.File(p) for p in paths]
    box = files[0]["Header"].attrs["BoxSize"][0]
    # Pre-displacement positions are at the grid cell centres
    assert np.allclose(files[0]["1/PrePosition"][:][0], box / 8 / 2)

    def disp(f):
        return (f["1/Position"][:] - f["1/PrePosition"][:] + box / 2) % box - box / 2
    # The paired simulation has the opposite displacements
    assert np.allclose(disp(files[0]), -disp(files[1]), atol=1e-6 * box)
    # WDM thermal velocities are added on top of the same Zel'dovich velocities
    extra = files[2]["1/Velocity"][:] - files[0]["1/Velocity"][:]
    assert np.all(np.linalg.norm(extra, axis=1) > 0)
    assert np.allclose(disp(files[0]), disp(files[2]))
