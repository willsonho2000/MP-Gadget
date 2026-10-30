"""Parameter file parsing, following genic/params.c and libgadget/utils/paramset.c."""

from types import SimpleNamespace

from .physconst import CM_PER_MPC, HUBBLE, message

REQUIRED = True
OPTIONAL = False

# (name, type, required, default, help). Mirrors create_parameters() in genic/params.c.
SCHEMA = [
    ("FileWithInputSpectrum", str, REQUIRED, "", "File containing input power spectrum, from CLASS or CAMB."),
    ("OutputDir", str, REQUIRED, "", "Output directory in which to store the ICs"),
    ("FileBase", str, REQUIRED, "", "File name of the ICs."),
    ("Omega0", float, REQUIRED, 0.2814, "Total matter density, cdm + baryons + massive neutrinos at z=0."),
    ("OmegaBaryon", float, REQUIRED, 0.0464, "Omega Baryon."),
    ("OmegaLambda", float, REQUIRED, 0.7186, "Dark energy density at z=0"),
    ("HubbleParam", float, REQUIRED, 0.697, "Hubble parameter"),
    ("ProduceGas", int, REQUIRED, 0, "Should we create baryon particles?"),
    ("BoxSize", float, REQUIRED, 0, "Size of box in internal units."),
    ("Redshift", float, REQUIRED, 99, "Starting redshift"),
    ("Nmesh", int, OPTIONAL, 0, "Size of the FFT grid used to estimate displacements."),
    ("Ngrid", int, REQUIRED, 0, "Size of regular grid on which the undisplaced CDM particles are created."),
    ("NgridGas", int, OPTIONAL, -1, "Size of regular grid on which the undisplaced gas particles are created."),
    ("NgridNu", int, OPTIONAL, 0, "Number of neutrino particles created for hybrid neutrinos."),
    ("Seed", int, REQUIRED, 0, "Random number generator seed for the phases of the Gaussian random field."),
    ("MakeGlassGas", int, OPTIONAL, -1, "Generate Glass IC for gas instead of Grid IC."),
    ("MakeGlassCDM", int, OPTIONAL, 0, "Generate Glass IC for CDM instead of Grid IC."),
    ("UnitaryAmplitude", int, OPTIONAL, 1, "If 1, Fourier mode amplitudes are not scattered."),
    ("WhichSpectrum", int, OPTIONAL, 2, "Type of spectrum, 2 for file "),
    ("Omega_fld", float, OPTIONAL, 0, "Energy density of dark energy fluid."),
    ("w0_fld", float, OPTIONAL, -1., "Dark energy equation of state"),
    ("wa_fld", float, OPTIONAL, 0, "Dark energy evolution parameter"),
    ("Omega_ur", float, OPTIONAL, 0, "Extra radiation density, eg, a sterile neutrino"),
    ("CLASS_Radiation", int, OPTIONAL, 0, "If enabled, we enforce that sum(Omega_i) = 1."),
    ("MNue", float, OPTIONAL, 0, "First neutrino mass in eV."),
    ("MNum", float, OPTIONAL, 0, "Second neutrino mass in eV."),
    ("MNut", float, OPTIONAL, 0, "Third neutrino mass in eV."),
    ("MWDM_therm", float, OPTIONAL, 0, "WDM particle mass in keV for thermal velocities."),
    ("Max_nuvel", float, OPTIONAL, 5000, "Maximum neutrino velocity sampled from the F-D distribution."),
    ("DifferentTransferFunctions", int, OPTIONAL, 1, "Use species specific transfer functions for baryon and CDM."),
    ("ScaleDepVelocity", int, OPTIONAL, -1, "Use scale dependent velocity transfer functions."),
    ("FileWithTransferFunction", str, OPTIONAL, "", "File containing CLASS formatted transfer functions."),
    ("MaxMemSizePerNode", float, OPTIONAL, 0.6, "Ignored by the Python version."),
    ("CMBTemperature", float, OPTIONAL, 2.7255, "CMB temperature in K"),
    ("RadiationOn", float, OPTIONAL, 1, "Include radiation in the background."),
    ("UsePeculiarVelocity", int, OPTIONAL, 1, "If 0, v/sqrt(a) is saved in the ICs to match Gadget-2."),
    ("SavePrePos", int, OPTIONAL, 1, "Save the pre-displacement positions in the snapshot."),
    ("InvertPhase", int, OPTIONAL, 0, "Flip phase for paired simulation"),
    ("PrePosGridCenter", int, OPTIONAL, 0, "Set pre-displacement positions at the center of the grid"),
    ("ShowBacktrace", int, OPTIONAL, 1, "Ignored by the Python version."),
    ("PrimordialAmp", float, OPTIONAL, 2.215e-9, "Ignored."),
    ("Sigma8", float, OPTIONAL, -1, "Renormalise Sigma8 to this number if positive"),
    ("InputPowerRedshift", float, OPTIONAL, -1, "Redshift of the input power. Negative means the starting redshift."),
    ("PrimordialIndex", float, OPTIONAL, 0.971, "Tilting power, ignored for tabulated input."),
    ("PrimordialRunning", float, OPTIONAL, 0, "Ignored."),
    ("UnitVelocity_in_cm_per_s", float, OPTIONAL, 1e5, "Velocity unit in cm/sec."),
    ("UnitLength_in_cm", float, OPTIONAL, CM_PER_MPC / 1000, "Length unit in cm."),
    ("UnitMass_in_g", float, OPTIONAL, 1.989e43, "Mass unit in g."),
    ("NumPartPerFile", int, OPTIONAL, 1024 * 1024 * 128, "Number of particles per striped bigfile."),
    ("NumWriters", int, OPTIONAL, 0, "Ignored by the Python version."),
]

_BLANKS = " \t\r\n="
_COMMENTS = "%#"


def _convert(typ, value):
    if typ is int:
        # atoi-like: accept "1", "1.0" and "1e3"
        try:
            return int(value)
        except ValueError:
            return int(float(value))
    if typ is float:
        return float(value)
    return value


def parse_parameter_text(text):
    """Parse 'Name = value # comment' lines into a dict of strings, like param_emit()."""
    raw = {}
    for lineno, line in enumerate(text.split("\n")):
        stripped = line.lstrip(_BLANKS)
        if not stripped or stripped[0] in _COMMENTS:
            continue
        i = 0
        while i < len(stripped) and stripped[i] not in _BLANKS:
            i += 1
        name = stripped[:i]
        rest = stripped[i:].lstrip(_BLANKS)
        if not rest or rest[0] in _COMMENTS:
            raise ValueError("Line %d : `%s` is malformed." % (lineno, line))
        j = 0
        while j < len(rest) and rest[j] not in _COMMENTS:
            j += 1
        raw[name] = rest[:j].strip()
    return raw


def read_parameters(text):
    """Return a dict of all parameters with defaults filled in and types converted."""
    raw = parse_parameter_text(text)
    schema = {s[0]: s for s in SCHEMA}
    unknown = [name for name in raw if name not in schema]
    if unknown:
        raise ValueError("Parameter(s) %s unknown." % ", ".join("`%s`" % u for u in unknown))
    missing = [s[0] for s in SCHEMA if s[2] is REQUIRED and s[0] not in raw]
    if missing:
        raise ValueError("Parameter(s) %s required, but not set." % ", ".join("`%s`" % m for m in missing))
    ps = {}
    for name, typ, _, default, _ in SCHEMA:
        ps[name] = _convert(typ, raw[name]) if name in raw else default
    return ps


def read_parameterfile(fname, verbose=True):
    """Read a GenIC parameter file. Returns (config, cosmology_params).

    The returned namespaces mirror struct genic_config and the fields of Cosmology
    set in read_parameterfile() of genic/params.c."""
    with open(fname) as fh:
        ps = read_parameters(fh.read())

    if verbose:
        message("----------- Running with Parameters ----------")
        for name, _, _, _, helptext in SCHEMA:
            message("%-31s %-20s # %s" % (name, ps[name], helptext))
        message("----------------------------------------------")

    cp = SimpleNamespace(
        Omega0=ps["Omega0"],
        OmegaLambda=ps["OmegaLambda"],
        OmegaBaryon=ps["OmegaBaryon"],
        HubbleParam=ps["HubbleParam"],
        Omega_fld=ps["Omega_fld"],
        w0_fld=ps["w0_fld"],
        wa_fld=ps["wa_fld"],
        Omega_ur=ps["Omega_ur"],
        use_class_radiation_convention=ps["CLASS_Radiation"],
        CMBTemperature=ps["CMBTemperature"],
        RadiationOn=ps["RadiationOn"],
        MNu=[ps["MNue"], ps["MNum"], ps["MNut"]],
    )
    if cp.OmegaLambda > 0 and cp.Omega_fld > 0:
        raise ValueError("Cannot have OmegaLambda and Omega_fld (evolving dark energy) at the same time!")

    redshift = ps["Redshift"]
    c = SimpleNamespace()
    c.WDM_therm_mass = ps["MWDM_therm"]
    c.ProduceGas = ps["ProduceGas"]
    c.InvertPhase = ps["InvertPhase"]
    c.UnitVelocity_in_cm_per_s = ps["UnitVelocity_in_cm_per_s"]
    c.UnitLength_in_cm = ps["UnitLength_in_cm"]
    c.UnitMass_in_g = ps["UnitMass_in_g"]

    # Power spectrum parameters
    powerp = SimpleNamespace()
    powerp.InputPowerRedshift = ps["InputPowerRedshift"]
    if powerp.InputPowerRedshift < 0:
        powerp.InputPowerRedshift = redshift
    powerp.Sigma8 = ps["Sigma8"]
    # Always specify Sigma8 at z=0
    if powerp.Sigma8 > 0:
        powerp.InputPowerRedshift = 0
    powerp.FileWithInputSpectrum = ps["FileWithInputSpectrum"]
    powerp.FileWithTransferFunction = ps["FileWithTransferFunction"]
    powerp.DifferentTransferFunctions = ps["DifferentTransferFunctions"]
    powerp.ScaleDepVelocity = ps["ScaleDepVelocity"]
    # By default ScaleDepVelocity follows DifferentTransferFunctions.
    if powerp.ScaleDepVelocity < 0:
        powerp.ScaleDepVelocity = powerp.DifferentTransferFunctions
    powerp.WhichSpectrum = ps["WhichSpectrum"]
    powerp.PrimordialIndex = ps["PrimordialIndex"]
    powerp.PrimordialRunning = ps["PrimordialRunning"]
    c.PowerP = powerp

    # Simulation parameters
    c.UsePeculiarVelocity = ps["UsePeculiarVelocity"]
    c.SavePrePos = ps["SavePrePos"]
    c.PrePosGridCenter = ps["PrePosGridCenter"]
    c.BoxSize = ps["BoxSize"]
    c.Nmesh = ps["Nmesh"]
    c.Ngrid = ps["Ngrid"]
    c.NgridGas = ps["NgridGas"]
    if c.NgridGas < 0:
        c.NgridGas = c.Ngrid
    if not c.ProduceGas:
        c.NgridGas = 0
    # Enable 'hybrid' neutrinos
    c.NGridNu = ps["NgridNu"]
    # Convert physical km/s at z=0 in an unperturbed universe to
    # internal gadget (comoving) velocity units at starting redshift.
    # (This matches the C code, which is only correct for UnitVelocity_in_cm_per_s = 1e5.)
    c.Max_nuvel = ps["Max_nuvel"] * (1 + redshift) * (c.UnitVelocity_in_cm_per_s / 1e5)
    c.Seed = ps["Seed"]
    c.UnitaryAmplitude = ps["UnitaryAmplitude"]
    c.OutputDir = ps["OutputDir"]
    c.InitCondFile = ps["FileBase"]
    c.MakeGlassGas = ps["MakeGlassGas"]
    # Use a baryon glass by default if we have different transfer functions,
    # since that is the way we reproduce the linear growth. Otherwise use a grid by default.
    if c.MakeGlassGas < 0:
        c.MakeGlassGas = 1 if powerp.DifferentTransferFunctions else 0
    c.MakeGlassCDM = ps["MakeGlassCDM"]

    ngrid = max(c.Ngrid, c.NgridGas)
    numpartperfile = ps["NumPartPerFile"]
    c.NumFiles = (ngrid ** 3 + numpartperfile - 1) // numpartperfile

    mnu_sum = sum(cp.MNu)
    if powerp.DifferentTransferFunctions and powerp.InputPowerRedshift != redshift \
            and (c.ProduceGas or mnu_sum):
        message("WARNING: Using different transfer functions but also rescaling power "
                "to account for linear growth. NOT what you want!")
    if mnu_sum > 0 or powerp.DifferentTransferFunctions or powerp.ScaleDepVelocity:
        if len(powerp.FileWithTransferFunction) == 0:
            raise ValueError("For massive neutrinos, different transfer functions, or scale dependent "
                             "growth functions you must specify a transfer function file")
    if not cp.RadiationOn and mnu_sum > 0:
        raise ValueError("You want massive neutrinos but no background radiation: "
                         "this will give an inconsistent cosmology.")

    if c.Nmesh == 0:
        c.Nmesh = 2 * ngrid
    c.TimeIC = 1 / (1 + redshift)
    unit_time_in_s = c.UnitLength_in_cm / c.UnitVelocity_in_cm_per_s
    cp.Hubble = HUBBLE * unit_time_in_s
    return c, cp
