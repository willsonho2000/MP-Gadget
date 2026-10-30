"""Driver for the Python GenIC. Port of genic/main.c."""

import argparse
import os
import time

import numpy as np

from . import glass, save, thermal, zeldovich
from .cosmology import Cosmology
from .params import read_parameterfile
from .physconst import CM_PER_MPC, message
from .power import DELTA_BAR, DELTA_CB, DELTA_CDM, DELTA_NU, DELTA_TOT, PowerSpectrum


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

    TotNumPart = config.Ngrid ** 3
    TotNumPartGas = config.ProduceGas * config.NgridGas ** 3

    # Offset the species so particles do not sit on top of each other.
    meanspacing = config.BoxSize / max(config.Ngrid, config.NgridGas)
    shift_gas = -config.ProduceGas * 0.5 * (cosmo.Omega0 - cosmo.OmegaBaryon) / cosmo.Omega0 * meanspacing
    shift_dm = config.ProduceGas * 0.5 * cosmo.OmegaBaryon / cosmo.Omega0 * meanspacing
    shift_nu = 0
    if not config.ProduceGas and config.NGridNu > 0:
        OmegaNu = cosmo.get_omega_nu(1)
        shift_nu = -0.5 * (cosmo.Omega0 - OmegaNu) / cosmo.Omega0 * meanspacing
        shift_dm = 0.5 * OmegaNu / cosmo.Omega0 * meanspacing
    if config.PrePosGridCenter:
        shift_dm += 0.5 * meanspacing
        shift_gas += 0.5 * meanspacing
        shift_nu += 0.5 * meanspacing

    os.makedirs(config.OutputDir, exist_ok=True)
    outpath = os.path.join(config.OutputDir, config.InitCondFile)
    bf = save.create_file(outpath)

    # Massive neutrinos
    TotNu = config.NGridNu ** 3
    total_nufrac = 0
    nu_therm = None
    if TotNu > 0:
        kBMNu = 3 * cosmo.ONu.kBtnu / sum(cosmo.MNu)
        v_th = thermal.NU_V0(config.TimeIC, kBMNu, config.UnitVelocity_in_cm_per_s)
        if not config.UsePeculiarVelocity:
            v_th /= np.sqrt(config.TimeIC)
        nu_therm = thermal.ThermalVel(v_th, config.Max_nuvel / v_th, 0)
        total_nufrac = nu_therm.total_frac
        message("F-D velocity scale: %g. Max particle vel: %g. Fraction of mass in particles: %g"
                % (v_th * np.sqrt(config.TimeIC), config.Max_nuvel * np.sqrt(config.TimeIC), total_nufrac))
    save.saveheader(bf, TotNumPart, TotNumPartGas, TotNu, total_nufrac, config.BoxSize, cosmo, config)
    save.save_all_transfer_tables(bf, power)

    # Use the 'total' (CDM + baryon) transfer function unless DifferentTransferFunctions are on.
    DMType, GasType, NuType = DELTA_CB, DELTA_CB, DELTA_NU
    if config.ProduceGas and config.PowerP.DifferentTransferFunctions:
        DMType, GasType = DELTA_CDM, DELTA_BAR

    # Neutrino mass can be neglected here since it only matters for the glass force.
    mass = save.compute_mass(TotNumPart, TotNumPartGas, 0, 0, config.BoxSize, cosmo)
    idgen_cdm = zeldovich.ParticleGrid(config.Ngrid, config.BoxSize)
    idgen_gas = zeldovich.ParticleGrid(config.NgridGas, config.BoxSize)

    disp = zeldovich.DisplacementField(config.Nmesh, config.BoxSize, config.Seed,
                                       config.UnitaryAmplitude, config.InvertPhase)

    # Unperturbed CDM particles
    if not config.MakeGlassCDM:
        cdm = zeldovich.setup_grid(idgen_cdm, shift_dm, mass[1])
    else:
        cdm = glass.setup_glass(idgen_cdm, config.Nmesh, glass_seed_hash(config.Seed), mass[1])

    gas = None
    if config.ProduceGas:
        if not config.MakeGlassGas:
            gas = zeldovich.setup_grid(idgen_gas, shift_gas, mass[0])
        else:
            gas = glass.setup_glass(idgen_gas, config.Nmesh, glass_seed_hash(config.Seed + 1), mass[0])
        # Coherent glass evolution of both species, to avoid close pairs
        if config.MakeGlassGas or config.MakeGlassCDM:
            both = zeldovich.ICParticles.concatenate([cdm, gas])
            glass.glass_evolve(config.Nmesh, config.BoxSize, 14, both)
            cdm, gas = both.split(len(cdm))

    # Pre-displacement positions
    for icp in (cdm, gas):
        if icp is not None:
            icp.PrePos[:] = icp.Pos

    if len(cdm) > 0:
        disp.displacement_fields(cdm, DMType, power, cosmo, config)
        # Add a thermal velocity to WDM particles
        if config.WDM_therm_mass > 0:
            v_th = thermal.WDM_V0(config.TimeIC, config.WDM_therm_mass,
                                  cosmo.Omega0 - cosmo.OmegaBaryon - cosmo.get_omega_nu(1),
                                  cosmo.HubbleParam, config.UnitVelocity_in_cm_per_s)
            if not config.UsePeculiarVelocity:
                v_th /= np.sqrt(config.TimeIC)
            wdm = thermal.ThermalVel(v_th, 10000 / v_th, 0)
            wdm.add_thermal_speeds(np.random.default_rng(config.Seed + 1), cdm.Vel)
        save.write_particle_data(bf, 1, cdm, idgen_cdm.ids(0), config.SavePrePos, config.NumFiles)
    del cdm

    if config.ProduceGas:
        disp.displacement_fields(gas, GasType, power, cosmo, config)
        save.write_particle_data(bf, 0, gas, idgen_gas.ids(TotNumPart), config.SavePrePos, config.NumFiles)
    del gas

    # Neutrino particles with random thermal velocities
    if config.NGridNu > 0:
        idgen_nu = zeldovich.ParticleGrid(config.NGridNu, config.BoxSize)
        nu = zeldovich.setup_grid(idgen_nu, shift_nu, mass[2])
        disp.displacement_fields(nu, NuType, power, cosmo, config)
        nu_therm.add_thermal_speeds(np.random.default_rng(config.Seed + 2), nu.Vel)
        save.write_particle_data(bf, 2, nu, idgen_nu.ids(TotNumPart + TotNumPartGas),
                                 config.SavePrePos, config.NumFiles)

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
