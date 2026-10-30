"""Write the initial conditions as a BigFile snapshot. Port of libgenic/save.c."""

import numpy as np

from .power import DELTA_BAR, MAXCOLS, TNAMES


def _bigfile():
    try:
        import bigfile
    except ImportError as err:  # pragma: no cover
        raise ImportError("The python 'bigfile' package is needed to write ICs: pip install bigfile") from err
    return bigfile


def create_file(path):
    return _bigfile().File(path, create=True)


def compute_mass(TotNumPartCDM, TotNumPartGas, TotNuPart, nufrac, BoxSize, cosmo):
    """Particle mass for each type, from the cosmology and the number of particles."""
    mass = np.zeros(6)
    OmegatoMass = cosmo.RhoCrit * BoxSize ** 3
    OmegaCDM = cosmo.Omega0
    if TotNumPartGas > 0:
        mass[0] = cosmo.OmegaBaryon * OmegatoMass / TotNumPartGas
        OmegaCDM -= cosmo.OmegaBaryon
    if sum(cosmo.MNu) > 0:
        OmegaNu = cosmo.get_omega_nu(1)
        OmegaCDM -= OmegaNu
        if TotNuPart > 0:
            mass[2] = nufrac * OmegaNu * OmegatoMass / TotNuPart
    mass[1] = OmegaCDM * OmegatoMass / TotNumPartCDM
    return mass


def saveheader(bf, TotNumPartCDM, TotNumPartGas, TotNuPart, nufrac, BoxSize, cosmo, config):
    mass = compute_mass(TotNumPartCDM, TotNumPartGas, TotNuPart, nufrac, BoxSize, cosmo)
    attrs = {
        "TotNumPart": np.array([TotNumPartGas, TotNumPartCDM, TotNuPart, 0, 0, 0], dtype="i8"),
        "MassTable": np.array(mass, dtype="f8"),
        "Time": np.float64(config.TimeIC),
        "Redshift": np.float64(1.0 / config.TimeIC - 1.),
        "BoxSize": np.float64(BoxSize),
        "UsePeculiarVelocity": np.int32(config.UsePeculiarVelocity),
        "Omega0": np.float64(cosmo.Omega0),
        "FractionNuInParticles": np.float64(nufrac),
        "OmegaBaryon": np.float64(cosmo.OmegaBaryon),
        "OmegaLambda": np.float64(cosmo.OmegaLambda),
        "OmegaFld": np.float64(cosmo.Omega_fld),
        "OmegaUR": np.float64(cosmo.Omega_ur),
        "OmegaK": np.float64(cosmo.OmegaK),
        "class_radiation_convention": np.int32(cosmo.use_class_radiation_convention),
        "W0_Fld": np.float64(cosmo.w0_fld),
        "WA_Fld": np.float64(cosmo.wa_fld),
        "UnitLength_in_cm": np.float64(config.UnitLength_in_cm),
        "UnitMass_in_g": np.float64(config.UnitMass_in_g),
        "UnitVelocity_in_cm_per_s": np.float64(config.UnitVelocity_in_cm_per_s),
        "HubbleParam": np.float64(cosmo.HubbleParam),
        "InvertPhase": np.int32(config.InvertPhase),
        "Seed": np.int64(config.Seed),
        "UnitaryAmplitude": np.int32(config.UnitaryAmplitude),
    }
    with bf.create("Header") as bheader:
        for key, value in attrs.items():
            bheader.attrs[key] = value
    return mass


def _save_transfer(bf, bname, logk, columns, colnames):
    with bf.create(bname) as block:
        block.attrs["Nentry"] = np.uint64(len(logk))
    bf.create_from_array("%s/logk" % bname, np.asarray(logk, dtype="f8"), Nfile=1)
    for name, col in zip(colnames, columns):
        bf.create_from_array("%s/%s" % (bname, name), np.asarray(col, dtype="f8"), Nfile=1)


def save_all_transfer_tables(bf, power):
    """Save the input power spectrum and the transfer function tables to the IC file."""
    _save_transfer(bf, "ICPower", power.power_logk, [power.power_logD], ["DELTA_MAT"])
    transfer = power.transfer
    if transfer.shape[1] == 0:
        transfer = np.zeros((MAXCOLS, 0))
    _save_transfer(bf, "ICTransfers", power.transfer_logk, transfer[DELTA_BAR:MAXCOLS], TNAMES)


def write_particle_data(bf, ptype, icp, ids, save_prepos, numfiles):
    """Write the particle arrays for one particle type."""
    def save(name, array, dtype):
        bf.create_from_array("%d/%s" % (ptype, name), np.ascontiguousarray(array, dtype=dtype), Nfile=numfiles)

    if save_prepos:
        save("PrePosition", icp.PrePos, "f8")
    save("ICDensity", icp.Density, "f4")
    save("Position", icp.Pos, "f8")
    save("Velocity", icp.Vel, "f4")
    save("ID", ids, "u8")
