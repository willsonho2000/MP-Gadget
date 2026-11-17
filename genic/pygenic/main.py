"""Driver for the Python GenIC. Port of genic/main.c."""

import argparse
import os
import time

import numpy as np

from . import glass, save, zeldovich
from .cosmology import Cosmology
from .params import read_parameterfile
from .physconst import CM_PER_MPC, message
from .power import DELTA_BAR, DELTA_CB, DELTA_CDM, DELTA_TOT, PowerSpectrum


def glass_seed_hash(seed):
    return (seed * 9999721) % (1 << 63)


def print_spec(config, cosmo, power):
    """Write the input power spectrum at the initial redshift, for checking."""
    fname = os.path.join(config.OutputDir, "inputspec_%s.txt" % config.InitCondFile)
    DDD = cosmo.GrowthFactor(config.TimeIC, 1.0)
    scale = CM_PER_MPC / config.UnitLength_in_cm
    # From 2x box size to the smallest mode in the simulation
    kstart = 2 * np.pi / (2 * config.BoxSize)
    kend = 2 * np.pi / (config.BoxSize / config.Ngrid)
    message("kstart=%g kend=%g (h/Mpc)" % (kstart * scale, kend * scale))
    nk = int(np.ceil(np.log(kend / kstart) / np.log(1.025)))
    k = kstart * 1.025 ** np.arange(nk)
    k = k[k < kend]
    po = power.DeltaSpec(k, DELTA_TOT) ** 2
    with open(fname, "w") as fd:
        fd.write("# %12g %12g\n" % (1 / config.TimeIC - 1, DDD))
        for kk, pp in zip(k, po):
            fd.write("%12g %12g\n" % (kk * scale, pp))


def generate(paramfile, verbose=True):
    """Generate initial conditions from a GenIC parameter file. Returns the output path."""
    start = time.time()
    config, cp = read_parameterfile(paramfile, verbose=verbose)
    cosmo = Cosmology(cp, config.TimeIC, config.UnitLength_in_cm, config.UnitMass_in_g,
                      config.UnitVelocity_in_cm_per_s)
    power = PowerSpectrum(config.TimeIC, config.UnitLength_in_cm, cosmo, config.PowerP)

    box, nmesh = config.BoxSize, config.Nmesh
    ncdm = config.Ngrid ** 3
    ngas = config.ProduceGas * config.NgridGas ** 3

    # Offset the species so particles do not sit on top of each other.
    meanspacing = box / max(config.Ngrid, config.NgridGas)
    shift_gas = -config.ProduceGas * 0.5 * (cosmo.Omega0 - cosmo.OmegaBaryon) / cosmo.Omega0 * meanspacing
    shift_dm = config.ProduceGas * 0.5 * cosmo.OmegaBaryon / cosmo.Omega0 * meanspacing
    if config.PrePosGridCenter:
        shift_dm += 0.5 * meanspacing
        shift_gas += 0.5 * meanspacing

    os.makedirs(config.OutputDir, exist_ok=True)
    outpath = os.path.join(config.OutputDir, config.InitCondFile)
    bf = save.create_file(outpath)

    mass = save.saveheader(bf, ncdm, ngas, cosmo, config)
    save.save_all_transfer_tables(bf, power)

    # Displace both species with the 'total' (CDM + baryon) transfer function,
    # unless they have different transfer functions and we are not perturbing the masses.
    DMType, GasType = DELTA_CB, DELTA_CB
    if config.ProduceGas and config.PowerP.DifferentTransferFunctions and not config.PerturbMass:
        DMType, GasType = DELTA_CDM, DELTA_BAR

    disp = zeldovich.DisplacementField(nmesh, box, config.Seed, config.UnitaryAmplitude)

    # Unperturbed positions of each species: a grid or a glass
    if config.MakeGlassCDM:
        cdm_pos = glass.setup_glass(config.Ngrid, box, nmesh, glass_seed_hash(config.Seed), mass[1])
    else:
        cdm_pos = zeldovich.grid_positions(config.Ngrid, box, shift_dm)
    if config.ProduceGas:
        if config.MakeGlassGas:
            gas_pos = glass.setup_glass(config.NgridGas, box, nmesh, glass_seed_hash(config.Seed + 1), mass[0])
        else:
            gas_pos = zeldovich.grid_positions(config.NgridGas, box, shift_gas)
        # Coherent glass evolution of both species, to avoid close pairs
        if config.MakeGlassGas or config.MakeGlassCDM:
            pos = glass.glass_evolve(nmesh, box, np.concatenate([cdm_pos, gas_pos]),
                                     np.repeat([mass[1], mass[0]], [ncdm, ngas]))
            cdm_pos, gas_pos = pos[:ncdm], pos[ncdm:]

    # Displace the particles and write them. Gas IDs follow the CDM IDs.
    # With PerturbMass both species follow the total matter, and their density difference
    # delta_bc = delta_b - delta_cdm at the unperturbed positions goes into the masses:
    # m_b = <m_b> (1 + f_c delta_bc) and m_c = <m_c> (1 - f_b delta_bc), with f_c = 1 - f_b.
    # (Hahn, Rampf & Uhlemann 2021, arXiv:2008.09124, eq. 15)
    f_b = cosmo.OmegaBaryon / cosmo.Omega0
    cdm = zeldovich.ICParticles(cdm_pos)
    if config.PerturbMass:
        cdm.Mass = mass[1] * (1 - f_b * disp.density_difference(cdm_pos, power, config.Ngrid))
    disp.displacement_fields(cdm, DMType, power, cosmo, config)
    save.write_particle_data(bf, 1, cdm, 0, config.SavePrePos, config.NumFiles)

    if config.ProduceGas:
        gas = zeldovich.ICParticles(gas_pos)
        if config.PerturbMass:
            gas.Mass = mass[0] * (1 + (1 - f_b) * disp.density_difference(gas_pos, power, config.NgridGas))
        disp.displacement_fields(gas, GasType, power, cosmo, config)
        save.write_particle_data(bf, 0, gas, ncdm, config.SavePrePos, config.NumFiles)

    bf.close()
    message("IC's generated in %.1f s." % (time.time() - start))
    message("Initial scale factor = %g" % config.TimeIC)
    print_spec(config, cosmo, power)
    return outpath


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pygenic",
                                     description="Generate MP-Gadget initial conditions (Python port of MP-GenIC).")
    parser.add_argument("paramfile", help="GenIC parameter file")
    parser.add_argument("-q", "--quiet", action="store_true", help="Do not print the parameter table")
    args = parser.parse_args(argv)
    generate(args.paramfile, verbose=not args.quiet)
    return 0
