"""Compare the power spectra of IC snapshots and a reference IC with linear theory.

Usage: python plot_matter_pk.py IC [IC ...] [--ref ICS_ref/IC] [--nmesh N] [--out FILE]

Each IC is an MP-GenIC / pygenic BigFile (e.g. ICS_grid/IC; the output directory ICS_grid
works too).
There is one column of panels for the total matter and, if the ICs have gas, one for the
gas alone. The top panel shows the measured P(k) of each IC and of the reference. The
bottom panel shows each of them divided by the linear input, averaged over the same modes
as the measurement.

The linear input for the total matter is the inputspec_*.txt the generator writes next to
the IC. For the gas it is that spectrum times the square of the baryon transfer function
(DELTA_BAR in the ICTransfers block of the IC, which is relative to the total matter).

The total matter density is the mass-weighted sum of gas (type 0) and dark matter (type 1).
Particles are weighted by their Mass block if they have one (PerturbMass = 1) and by the
header MassTable otherwise. A Mass block is ignored, with a warning, if the header says
PerturbMass = 0.

The density is assigned with a TSC kernel on a mesh four times finer than the particle
grid, and the kernel is deconvolved. Cloud-in-cell on a coarser mesh is biased at the
percent level for particles on a grid (see genic/pygenic/README.md).
"""
import argparse
import glob
import os

import bigfile
import matplotlib.pyplot as plt
import numpy as np

CM_PER_MPC = 3.085678e24
GAS, DM = 0, 1
TITLES = {"total": "Total matter", "gas": "Gas"}

# Colours: the reference is neutral ink, each compared IC takes the next hue in fixed order.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def open_ic(path):
    """Open an IC BigFile. Accepts the snapshot itself or the directory holding IC/."""
    path = path.rstrip("/")
    if not os.path.isdir(os.path.join(path, "Header")):
        inner = os.path.join(path, "IC")
        if not os.path.isdir(os.path.join(inner, "Header")):
            raise SystemExit(f"{path}: no BigFile snapshot here (no Header block)")
        path = inner
    return path, bigfile.File(path)


def tsc_mass_mesh(pos, mass, box, nmesh):
    """Mass on a periodic nmesh^3 mesh, assigned with the triangular shaped cloud kernel."""
    x = (pos % box) / box * nmesh
    centre = np.rint(x).astype(np.int64)
    d = x - centre
    # Weights of the cells at centre - 1, centre and centre + 1, for each axis.
    w = np.stack([0.5 * (0.5 - d) ** 2, 0.75 - d ** 2, 0.5 * (0.5 + d) ** 2])
    rho = np.zeros(nmesh ** 3)
    for ix in range(3):
        for iy in range(3):
            for iz in range(3):
                idx = (centre + [ix - 1, iy - 1, iz - 1]) % nmesh
                flat = (idx[:, 0] * nmesh + idx[:, 1]) * nmesh + idx[:, 2]
                weight = mass * w[ix, :, 0] * w[iy, :, 1] * w[iz, :, 2]
                rho += np.bincount(flat, weights=weight, minlength=nmesh ** 3)
    return rho.reshape((nmesh,) * 3)


def power_spectrum(delta, box, kmax_fund, theory=None):
    """Spherically averaged P(k) of an overdensity mesh, with the TSC window deconvolved.

    Bins are one fundamental mode wide, up to kmax_fund (in units of the fundamental).
    Returns k (in 1 / units of box) and P (in units of box^3).

    theory is an optional function P(k). It is averaged over the same modes as the
    measurement and returned as a third value (None without theory). P(k) is steep, so its
    average over a bin is not P at the mean k: in the lowest bin they differ by about 10%.
    """
    n = delta.shape[0]
    dk = np.fft.rfftn(delta) / n ** 3
    kf = np.fft.fftfreq(n, d=1.0 / n)
    kz = np.fft.rfftfreq(n, d=1.0 / n)
    kx, ky, kzz = kf[:, None, None], kf[None, :, None], kz[None, None, :]
    window = (np.sinc(kx / n) * np.sinc(ky / n) * np.sinc(kzz / n)) ** 3
    p = np.abs(dk / window) ** 2 * box ** 3
    kmag = np.sqrt(kx ** 2 + ky ** 2 + kzz ** 2)
    # The real FFT stores half of the modes: those with 0 < kz < n/2 stand for two.
    weight = np.broadcast_to(np.where((kzz > 0) & (kzz < n / 2), 2.0, 1.0), kmag.shape)
    edges = np.arange(0.5, kmax_fund + 1, 1.0)
    nbins = len(edges) - 1
    ibin = np.digitize(kmag, edges)
    used = (ibin >= 1) & (ibin <= nbins)
    ibin, weight, kmag, p = ibin[used] - 1, weight[used], kmag[used] * 2 * np.pi / box, p[used]

    def bin_sum(values):
        return np.bincount(ibin, weights=values * weight, minlength=nbins)

    count = bin_sum(1.0)
    good = count > 0
    ptheory = None if theory is None else bin_sum(theory(kmag))[good] / count[good]
    return bin_sum(kmag)[good] / count[good], bin_sum(p)[good] / count[good], ptheory


def read_header(f):
    attrs = f["Header"].attrs
    npart = np.asarray(attrs["TotNumPart"])
    return {
        "box": float(attrs["BoxSize"][0]),
        "redshift": float(attrs["Redshift"][0]),
        "mpc": float(attrs["UnitLength_in_cm"][0]) / CM_PER_MPC,  # length unit in Mpc/h
        "masstable": np.asarray(attrs["MassTable"]),
        "types": [pt for pt in (GAS, DM) if npart[pt] > 0],
        # pygenic says in the header whether the Mass blocks belong to this IC.
        "perturbmass": int(attrs["PerturbMass"][0]) if "PerturbMass" in attrs else None,
        "ngrid": round(max(npart[GAS], npart[DM]) ** (1 / 3)),
    }


def read_input_spectrum(icpath, head):
    """The linear matter power the generator was given, from inputspec_*.txt next to the IC.
    Returns k [h/Mpc] and P [(Mpc/h)^3], or None if the file is not there."""
    found = sorted(glob.glob(os.path.join(os.path.dirname(icpath), "inputspec_*.txt")))
    if not found:
        return None
    k, p = np.loadtxt(found[0], unpack=True)  # k is in h/Mpc, P in internal length^3
    return k, p * head["mpc"] ** 3


def linear_theory(f, spec):
    """The linear input P(k) of each species, as functions of k [h/Mpc] giving (Mpc/h)^3.

    spec = (k, P) is the total matter input. The gas has the baryon transfer function of
    the IC applied. Without a transfer table the gas follows the total matter."""
    logk_in, logp_in = np.log(spec[0]), np.log(spec[1])

    def total(k):  # log-log interpolation of the table
        return np.exp(np.interp(np.log(k), logk_in, logp_in))

    # f.blocks does not list the columns of ICTransfers, so look on disk.
    if not os.path.isdir(os.path.join(f.basename, "ICTransfers", "DELTA_BAR")):
        return {"total": total, "gas": total}
    tlogk = f["ICTransfers/logk"][:]  # log10 of k in h/Mpc
    tbar = f["ICTransfers/DELTA_BAR"][:]  # delta_b / delta_tot
    if len(tlogk) == 0:
        return {"total": total, "gas": total}

    def gas(k):  # the transfer function is held constant beyond its table, as in the generator
        return total(k) * np.interp(np.log10(k), tlogk, tbar) ** 2

    return {"total": total, "gas": gas}


def measure(f, head, nmesh, kmax_fund, theory, species):
    """P(k) of each species of an IC, and its linear input in the same bins.

    Returns {species: (k [h/Mpc], P [(Mpc/h)^3], binned linear input [(Mpc/h)^3])}, and
    whether the particles carry their own masses."""
    box, mpc = head["box"], head["mpc"]
    mesh = {}
    own_mass = False
    for pt in head["types"]:
        pos = f[f"{pt}/Position"][:]
        has_mass = f"{pt}/Mass" in f.blocks
        if has_mass and head["perturbmass"] == 0:
            # Left behind by an earlier PerturbMass = 1 run into the same directory:
            # writing an IC does not remove blocks that the new run does not make.
            print(f"warning: {f.basename}: ignoring the {pt}/Mass block, "
                  f"because the header says PerturbMass = 0")
            has_mass = False
        if has_mass:
            mass = f[f"{pt}/Mass"][:].astype("f8")
            own_mass = True
        else:
            mass = np.full(len(pos), head["masstable"][pt])
        mesh[pt] = tsc_mass_mesh(pos, mass, box, nmesh)
    rho = {"total": sum(mesh.values()), "gas": mesh.get(GAS)}
    out = {}
    for name in species:
        k, p, pin = power_spectrum(rho[name] / rho[name].mean() - 1, box, kmax_fund,
                                   lambda kk, name=name: theory[name](kk / mpc))
        out[name] = (k / mpc, p * mpc ** 3, pin)
    return out, own_mass


def label_of(icpath):
    """Name an IC by its output directory, e.g. ICS_ref/IC -> ICS_ref."""
    return os.path.basename(os.path.dirname(os.path.abspath(icpath))) or icpath


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ics", nargs="+", metavar="IC",
                        help="IC(s) to compare against the reference, e.g. ICS_grid")
    parser.add_argument("--ref", default="ICS_ref_grid/IC", help="reference IC (default: ICS_ref_grid/IC)")
    parser.add_argument("--nmesh", type=int, default=None,
                        help="mesh cells per side (default: 4 x the particle grid)")
    parser.add_argument("--ylim", type=float, nargs=2, metavar=("LO", "HI"), default=(0.8, 1.2),
                        help="range of the ratio panels (default: 0.8 1.2). The noise of a "
                             "glass exceeds any fixed range near the Nyquist frequency")
    parser.add_argument("--out", default="matter_pk_comparison.png")
    args = parser.parse_args()
    if len(args.ics) > len(SERIES):
        raise SystemExit(f"at most {len(SERIES)} ICs can be compared in one figure")

    refpath, reffile = open_ic(args.ref)
    refhead = read_header(reffile)
    opened = [open_ic(p) for p in args.ics]
    heads = [read_header(f) for _, f in opened]
    for (path, _), head in zip(opened, heads):
        for key in ("box", "redshift", "mpc"):
            if not np.isclose(head[key], refhead[key]):
                raise SystemExit(f"{path} and {refpath} differ in {key}: "
                                 f"{head[key]:g} vs {refhead[key]:g}")

    # The gas gets its own column only if every IC has gas.
    species = ["total"]
    if all(GAS in h["types"] for h in heads + [refhead]):
        species.append("gas")

    # Measure everything on one mesh and stop at the Nyquist frequency of the coarsest
    # particle grid, so that all spectra share the same k bins.
    ngrid_min = min(h["ngrid"] for h in heads + [refhead])
    nmesh = args.nmesh or 4 * max(h["ngrid"] for h in heads + [refhead])
    kmax_fund = min(ngrid_min, nmesh) // 2
    box_mpc = refhead["box"] * refhead["mpc"]
    knyq = np.pi * ngrid_min / box_mpc

    # Each IC is compared with its own linear input, or the reference's if it has none.
    reflabel = label_of(refpath)
    refspec = read_input_spectrum(refpath, refhead)
    specs = [read_input_spectrum(path, head) or refspec for (path, _), head in zip(opened, heads)]
    if refspec is None:
        refspec = next((s for s in specs if s is not None), None)
        specs = [s or refspec for s in specs]
    if refspec is None:
        raise SystemExit("no inputspec_*.txt next to any of the ICs: no linear input to compare with")

    # (label, {species: (k, P, binned linear input)}, has its own masses); the reference is last.
    measured = []
    for (path, f), head, spec in zip(opened, heads, specs):
        measured.append((label_of(path),)
                        + measure(f, head, nmesh, kmax_fund, linear_theory(f, spec), species))
    reftheory = linear_theory(reffile, refspec)
    measured.append((f"{reflabel} (reference)",)
                    + measure(reffile, refhead, nmesh, kmax_fund, reftheory, species))
    kref = measured[-1][1]["total"][0]

    half = kref < knyq / 2
    print(f"box {box_mpc:g} Mpc/h, z = {refhead['redshift']:g}, TSC mesh {nmesh}^3, "
          f"particle Nyquist {knyq:.2f} h/Mpc")
    for name in species:
        print(f"{TITLES[name]}: measured P(k) / linear input")
        for label, spectra, own_mass in measured:
            _, p, pin = spectra[name]
            ratio = p / pin
            print(f"  {label}{' (masses from the Mass block)' if own_mass else ''}: "
                  f"k < k_Nyq/2: mean {ratio[half].mean():.4f}, "
                  f"max |ratio - 1| {np.abs(ratio[half] - 1).max():.4f}; "
                  f"all k: mean {ratio.mean():.4f}, max |ratio - 1| {np.abs(ratio - 1).max():.4f}")

    plt.rcParams.update({"font.size": 11, "axes.edgecolor": INK_SECONDARY,
                         "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY})
    ncols = len(species)
    fig, axes = plt.subplots(2, ncols, figsize=(1.5 + 6.5 * ncols, 7.5), sharex=True,
                             sharey="row", squeeze=False, facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [2, 1]}, constrained_layout=True)

    # The reference is drawn last, dashed and in neutral ink, on top of the compared ICs.
    styles = [("-", color, 2) for color in SERIES[:len(measured) - 1]] + [("--", INK, 1.5)]
    kin = refspec[0]
    kin = kin[(kin > kref.min() / 1.3) & (kin < knyq * 1.05)]
    for name, (ax1, ax2) in zip(species, axes.T):
        ax1.loglog(kin, reftheory[name](kin), ":", color=INK_SECONDARY, lw=1.5, label="linear input")
        for (label, spectra, _), (ls, color, lw) in zip(measured, styles):
            k, p, pbinned = spectra[name]
            ax1.loglog(k, p, ls, color=color, lw=lw, label=label)
            ax2.semilogx(k, p / pbinned, ls, color=color, lw=lw)
        band = ax2.axhspan(0.99, 1.01, color=GRID, lw=0, zorder=0)
        ax2.axhline(1, color=INK_SECONDARY, lw=1, ls=":")
        ax1.set_title(TITLES[name], fontsize=11, loc="left")
        ax2.set_xlabel(r"$k$  [$h$/Mpc]")
        for ax in (ax1, ax2):
            ax.set_facecolor(SURFACE)
            ax.grid(color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

    # Every panel uses the same line styles, so one legend serves them all.
    axes[0, 0].set_ylabel(r"$P(k)$  [(Mpc/$h$)$^3$]")
    axes[0, 0].legend(frameon=False)
    axes[1, 0].set_ylabel("measured / linear input")
    axes[1, 0].legend([band], ["±1%"], frameon=False, fontsize=9)
    axes[1, 0].set_ylim(*args.ylim)
    fig.suptitle(f"Power spectrum of the initial conditions:  z = {refhead['redshift']:g}, "
                 f"{box_mpc:g} Mpc/h box, {ngrid_min}³ particles per species, "
                 f"up to the particle Nyquist frequency", fontsize=12, x=0.01, ha="left")
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
