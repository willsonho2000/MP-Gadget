#!/usr/bin/env python3
"""Plot the CAMB matter power spectrum at several redshifts.

The cosmology (h, Omega_m, Omega_b, A_s, n_s) is read from the run's
paramfile.gadget / paramfile.genic, so CAMB reproduces the spectrum the ICs
were generated from (checked: CAMB at z=99 matches ../class_pk_99.dat to <0.5%
for k < 100 h/Mpc).

The ICs use WDM_pk_99.dat = T(k)^2 * P_CDM, with the nCDM transfer function
    T(k) = (1 + (alpha k)^beta)^gamma,  (alpha, beta, gamma) = (0.02, 2.2, -4.5)
(see pk_tk_generate.ipynb). By default the same T(k)^2 is applied to the CAMB
linear spectrum; pass --cdm to turn it off or --wdm-params to change it.

--nonlinear adds CAMB's HMcode-2020 non-linear spectrum. CAMB has no WDM
non-linear model, so the curve is CDM: compare with the simulation only where
T(k) ~ 1 (k below a few tens of h/Mpc).

--sim overlays the simulation spectra (total matter, CDM + baryons)
and adds a bottom panel with the ratio P_sim / P_CAMB,linear.

Examples
--------
    python plot_camb_power.py --z 99 10 5 4.6 4.2               # CAMB only
    python plot_camb_power.py --sim keir/output --nonlinear       # z from the sim
    python plot_camb_power.py --sim keir/output keir_highres/output --labels 512 1024
"""
import argparse
import glob
import os
import re

import camb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# Ordinal blue ramp (light -> dark): high z is light, low z is dark.
_RAMP = LinearSegmentedColormap.from_list(
    "zramp", ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
# CAMB uses '--' (linear) and ':' (HMcode); simulations use these.
_SIM_LS = ["-", "-.", (0, (3, 1, 1, 1, 1, 1)), (0, (6, 2))]


def read_param(outdir, name, default):
    """Look up a scalar in the run's paramfile.gadget / paramfile.genic (above output/)."""
    for pf in ("paramfile.gadget", "paramfile.genic"):
        path = os.path.join(os.path.dirname(os.path.abspath(outdir)), pf)
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            for line in fh:
                m = re.match(rf"\s*{name}\s*=\s*([-+0-9.eE]+)", line)
                if m:
                    return float(m.group(1))
    return default


def scale_factors(outdir):
    """Scale-factor tags of the power-DM-<a>.txt files, lowest z first."""
    files = glob.glob(os.path.join(outdir, "power-DM-*.txt"))
    tags = [re.search(r"power-DM-(.+)\.txt$", f).group(1) for f in files]
    return sorted(tags, key=float, reverse=True)


def load_shells(path):
    """(k, P, modes) of the non-empty |k| shells; cached to a .npz next to the file."""
    cache = path[:-4] + ".npz"
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(path):
        d = np.load(cache)
        return d["k"], d["p"], d["n"]
    raw = np.fromfile(path, sep=" ").reshape(-1, 3)  # much faster than loadtxt
    k, p, n = raw.T
    good = np.isfinite(k) & np.isfinite(p) & (k > 0) & (n > 0)
    k, p, n = k[good], p[good], n[good]
    try:
        np.savez(cache, k=k, p=p, n=n)
    except OSError:
        pass
    return k, p, n


def log_bin(k, p, n, nbins):
    """Mode-weighted average of k and P in log-spaced k bins."""
    edges = np.logspace(np.log10(k.min()), np.log10(k.max() * 1.0001), nbins + 1)
    idx = np.digitize(k, edges) - 1
    nsum = np.bincount(idx, weights=n, minlength=nbins)[:nbins]
    ksum = np.bincount(idx, weights=n * k, minlength=nbins)[:nbins]
    psum = np.bincount(idx, weights=n * p, minlength=nbins)[:nbins]
    ok = nsum > 0
    return ksum[ok] / nsum[ok], psum[ok] / nsum[ok], nsum[ok]


def binned_spectrum(outdir, a, species, fb, nbins):
    """Simulation P(k) in h/Mpc, (Mpc/h)^3. Files are in h/kpc, (kpc/h)^3.

    species 'total' combines CDM and baryons assuming perfect correlation:
    P_m = ((1 - f_b) sqrt(P_c) + f_b sqrt(P_b))^2 (no cross spectrum is stored).
    """
    def one(tag):
        k, p, n = load_shells(os.path.join(outdir, f"power-{tag}-{a}.txt"))
        return log_bin(k, p, n, nbins)

    if species == "dm":
        k, p, n = one("DM")
    elif species == "bar":
        k, p, n = one("bar")
    else:
        k, pc, n = one("DM")
        kb, pb, _ = one("bar")
        p = ((1 - fb) * np.sqrt(pc) + fb * np.sqrt(np.interp(k, kb, pb))) ** 2
    return k * 1e3, p * 1e-9, n


def wdm_transfer(k, alpha, beta, gamma):
    """nCDM transfer function, k in h/Mpc."""
    return (1 + (alpha * k) ** beta) ** gamma


def camb_power(param_dir, redshifts, kmin, kmax, npoints, nonlinear):
    """Return k [h/Mpc], z (ascending), P_lin[z, k] and (optionally) P_nl[z, k]."""
    def par(name, default):
        return read_param(param_dir, name, default)

    h = par("HubbleParam", 0.669)
    om, ob = par("Omega0", 0.3209), par("OmegaBaryon", 0.0494)
    pars = camb.set_params(
        H0=100 * h, ombh2=ob * h**2, omch2=(om - ob) * h**2,
        mnu=0, num_massive_neutrinos=0, nnu=3.046,
        TCMB=par("CMBTemperature", 2.7255),
        As=par("PrimordialAmp", 2.215e-9), ns=par("PrimordialIndex", 0.971),
        tau=0.054, halofit_version="mead2020")
    # compute a little past kmax: CAMB's last few k-points are less accurate
    pars.set_matter_power(redshifts=sorted(set(redshifts), reverse=True),
                          kmax=1.5 * kmax * h, nonlinear=False)
    res = camb.get_results(pars)
    k, z, p_lin = res.get_matter_power_spectrum(minkh=kmin, maxkh=kmax, npoints=npoints)
    p_nl = None
    if nonlinear:
        pars.NonLinear = camb.model.NonLinear_both
        res.calc_power_spectra(pars)
        _, _, p_nl = res.get_matter_power_spectrum(minkh=kmin, maxkh=kmax, npoints=npoints)
    return k, np.asarray(z), p_lin, p_nl


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", nargs="+", default=[], metavar="OUTDIR",
                    help="simulation output dir(s) to overlay; also set the redshifts")
    ap.add_argument("--labels", nargs="+", help="legend label per --sim directory")
    ap.add_argument("--z", nargs="+", type=float,
                    help="redshifts (default: those found in the --sim dirs)")
    ap.add_argument("--params", default="keir/output",
                    help="output dir whose ../paramfile.* gives the cosmology "
                         "(default: first --sim dir, else keir/output)")
    ap.add_argument("--cdm", action="store_true", help="do not apply the WDM T(k)^2")
    ap.add_argument("--wdm-params", nargs=3, type=float, default=[0.02, 2.2, -4.5],
                    metavar=("ALPHA", "BETA", "GAMMA"))
    ap.add_argument("--nonlinear", action="store_true", help="add HMcode-2020 (CDM)")
    ap.add_argument("--species", choices=["total", "dm", "bar"], default="total",
                    help="simulation species to overlay (default total)")
    ap.add_argument("--kmin", type=float,
                    help="h/Mpc (default 0.3 with --sim, else 0.01)")
    ap.add_argument("--kmax", type=float, default=300, help="h/Mpc")
    ap.add_argument("--nbins", type=int, default=40, help="log-k bins for the sim")
    ap.add_argument("--delta2", action="store_true", help="plot k^3 P / 2pi^2")
    ap.add_argument("-o", "--output", default="camb_power.png")
    args = ap.parse_args()

    labels = args.labels or [os.path.basename(os.path.dirname(os.path.abspath(d)))
                             for d in args.sim]
    if len(labels) != len(args.sim):
        ap.error("--labels needs one entry per --sim directory")
    param_dir = args.sim[0] if args.sim else args.params
    if args.kmin is None:
        args.kmin = 0.3 if args.sim else 1e-2

    # Redshifts: explicit, else every snapshot in the sim dirs.
    sim_a = {d: scale_factors(d) for d in args.sim}
    if args.z:
        zs = sorted(args.z)
    else:
        zs = sorted({1 / float(a) - 1 for v in sim_a.values() for a in v})
    if not zs:
        ap.error("give --z or at least one --sim directory with power-*.txt files")

    k, zc, p_lin, p_nl = camb_power(param_dir, zs, args.kmin, args.kmax, 400,
                                    args.nonlinear)
    tk2 = 1.0 if args.cdm else wdm_transfer(k, *args.wdm_params) ** 2
    p_lin = p_lin * tk2
    conv = (k**3 / (2 * np.pi**2)) if args.delta2 else 1.0

    def colour(z):  # ordinal: evenly spaced by rank, dark = low z
        r = int(np.argmin(np.abs(np.array(zs) - z)))
        return _RAMP(1 - r / (len(zs) - 1) if len(zs) > 1 else 1.0)

    def camb_at(z):
        return int(np.argmin(np.abs(zc - z)))

    if args.sim:
        fig, (ax, axr) = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                                      gridspec_kw=dict(height_ratios=[3, 1], hspace=0.05))
    else:
        fig, ax = plt.subplots(figsize=(7, 5.2))
        axr = None

    for z in zs:
        i = camb_at(z)
        ax.plot(k, p_lin[i] * conv, color=colour(z), lw=1.5, ls="--" if args.sim else "-")
        if p_nl is not None:
            ax.plot(k, p_nl[i] * conv, color=colour(z), lw=1.5, ls=":")

    sim_min = np.inf
    for j, (outdir, label) in enumerate(zip(args.sim, labels)):
        ls = _SIM_LS[j % len(_SIM_LS)]
        fb = read_param(outdir, "OmegaBaryon", 0.0494) / read_param(outdir, "Omega0", 0.3209)
        for a in sim_a[outdir]:
            z = 1 / float(a) - 1
            if args.z and min(abs(np.array(zs) - z)) > 0.01:
                continue
            ks, ps, _ = binned_spectrum(outdir, a, args.species, fb, args.nbins)
            yconv = (ks**3 / (2 * np.pi**2)) if args.delta2 else 1.0
            ax.plot(ks, ps * yconv, color=colour(z), lw=2, ls=ls)
            sim_min = min(sim_min, np.min(ps * yconv))
            i = camb_at(z)
            axr.plot(ks, ps / np.interp(np.log(ks), np.log(k), p_lin[i]),
                     color=colour(z), lw=2, ls=ls)
            print(f"{label:>16s}  z = {z:5.2f}  k = [{ks.min():.3g}, {ks.max():.3g}] h/Mpc")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$\Delta^2(k)$" if args.delta2 else r"$P(k)\ \ [(h^{-1}\,\mathrm{Mpc})^3]$")
    model = "CDM" if args.cdm else "WDM (IC transfer fn)"
    ax.set_title(f"Matter power spectrum: CAMB linear, {model}")
    bottom = axr if axr is not None else ax
    bottom.set_xlabel(r"$k\ \ [h\,\mathrm{Mpc}^{-1}]$")
    bottom.set_xlim(args.kmin, args.kmax)
    if axr is not None:
        axr.axhline(1, color="0.6", lw=1)
        axr.set_ylabel(r"$P_\mathrm{sim}/P_\mathrm{CAMB,lin}$")
        axr.set_yscale("log")
        # Beyond k ~ 20 h/Mpc the WDM linear spectrum is cut off while non-linear
        # evolution regenerates power in the sim, so the ratio runs off the top.
        axr.set_ylim(0.3, 100)
        ax.set_ylim(bottom=0.1 * sim_min)
    for a_ in filter(None, (ax, axr)):
        a_.grid(True, which="major", color="0.9", lw=0.8)
        a_.set_axisbelow(True)
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)

    handles = [Line2D([], [], color=colour(z), lw=2, label=f"z = {z:.2f}") for z in zs]
    style = [("CAMB linear", "--" if args.sim else "-")]
    if p_nl is not None:
        style.append(("CAMB HMcode (CDM)", ":"))
    if args.sim:
        style += [(l, _SIM_LS[j % len(_SIM_LS)])
                  for j, l in enumerate(labels)]
    handles += [Line2D([], [], color="0.3", lw=1.5, ls=s, label=l) for l, s in style]
    ax.legend(handles=handles, frameon=False, loc="lower left", fontsize=9)

    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
