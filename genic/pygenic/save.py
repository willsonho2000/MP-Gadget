"""Write the initial conditions as a BigFile snapshot. Port of libgenic/save.c."""

import numpy as np

from .power import TNAMES


def create_file(path):
    try:
        import bigfile
    except ImportError as err:  # pragma: no cover
        raise ImportError("The python 'bigfile' package is needed to write ICs: pip install bigfile") from err
    return bigfile.File(path, create=True)


def saveheader(bf, ncdm, ngas, cosmo, config):
    """Write the header for ncdm CDM and ngas gas particles. Returns the mass table."""
    # Particle mass of each type: without gas, the CDM particles carry the baryons too.
    mass = np.zeros(6)
    OmegatoMass = cosmo.RhoCrit * config.BoxSize ** 3
    OmegaCDM = cosmo.Omega0
    if ngas > 0:
        mass[0] = cosmo.OmegaBaryon * OmegatoMass / ngas
        OmegaCDM -= cosmo.OmegaBaryon
    mass[1] = OmegaCDM * OmegatoMass / ncdm
    attrs = {
        "TotNumPart": np.array([ngas, ncdm, 0, 0, 0, 0], dtype="i8"),
        "MassTable": mass,
        "Time": np.float64(config.TimeIC),
        "Redshift": np.float64(1.0 / config.TimeIC - 1.),
        "BoxSize": np.float64(config.BoxSize),
        # Velocities are always peculiar velocities. MP-Gadget reads this flag and
        # assumes the Gadget-2 convention (v / sqrt(a)) if it is missing.
        "UsePeculiarVelocity": np.int32(1),
        "Omega0": np.float64(cosmo.Omega0),
        "OmegaBaryon": np.float64(cosmo.OmegaBaryon),
        "OmegaLambda": np.float64(cosmo.OmegaLambda),
        "OmegaFld": np.float64(cosmo.Omega_fld),
        "OmegaK": np.float64(cosmo.OmegaK),
        "class_radiation_convention": np.int32(cosmo.use_class_radiation_convention),
        "W0_Fld": np.float64(cosmo.w0_fld),
        "WA_Fld": np.float64(cosmo.wa_fld),
        "UnitLength_in_cm": np.float64(config.UnitLength_in_cm),
        "UnitMass_in_g": np.float64(config.UnitMass_in_g),
        "UnitVelocity_in_cm_per_s": np.float64(config.UnitVelocity_in_cm_per_s),
        "HubbleParam": np.float64(cosmo.HubbleParam),
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
    _save_transfer(bf, "ICTransfers", power.transfer_logk, power.transfer, TNAMES)


def write_particle_data(bf, ptype, icp, first_id, save_prepos, numfiles):
    """Write the particle arrays for one particle type. IDs count up from first_id + 1."""
    def save(name, array, dtype):
        bf.create_from_array("%d/%s" % (ptype, name), np.ascontiguousarray(array, dtype=dtype), Nfile=numfiles)

    if save_prepos:
        save("PrePosition", icp.PrePos, "f8")
    save("ICDensity", icp.Density, "f4")
    save("Position", icp.Pos, "f8")
    save("Velocity", icp.Vel, "f4")
    save("ID", np.arange(first_id + 1, first_id + len(icp.Pos) + 1), "u8")
