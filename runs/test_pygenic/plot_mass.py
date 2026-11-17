"""Histogram of the particle masses of an IC with perturbed masses (PerturbMass = 1).

Usage: python plot_mass.py [IC] [--bins N] [--out FILE]

IC is a pygenic BigFile (default ICS_test; the output directory or ICS_test/IC both work).
The histogram shows, for gas and dark matter, the mass of each particle relative to the
mean particle mass of its species (the header MassTable), as a fraction of the particles
per percent of mass.

With PerturbMass = 1 the masses are m_gas = <m_gas> (1 + f_c delta_bc) and
m_cdm = <m_cdm> (1 - f_b delta_bc), so the gas distribution is the mirror image of the
dark matter one, wider by f_c / f_b.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

from plot_matter_pk import GAS, DM, GRID, INK, INK_SECONDARY, SERIES, SURFACE, label_of, open_ic, read_header

# Dark matter and gas keep the same colours in every panel.
SPECIES = {DM: ("Dark matter", SERIES[0]), GAS: ("Gas", SERIES[1])}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ic", nargs="?", default="ICS_test", help="IC to plot (default: ICS_test)")
    parser.add_argument("--bins", type=int, default=200, help="number of mass bins (default: 200)")
    parser.add_argument("--out", default="mass_distribution.png")
    args = parser.parse_args()

    path, f = open_ic(args.ic)
    head = read_header(f)
    attrs = f["Header"].attrs
    if head["perturbmass"] != 1:
        raise SystemExit(f"{path}: the header does not say PerturbMass = 1, "
                         f"so all particles of a species have the same mass")
    f_b = float(attrs["OmegaBaryon"][0]) / float(attrs["Omega0"][0])

    # Mass perturbation of each particle in percent: 100 (m / <m> - 1)
    pert = {pt: 100 * (f[f"{pt}/Mass"][:].astype("f8") / head["masstable"][pt] - 1) for pt in head["types"]}
    print(f"{label_of(path)}: particle masses relative to the mean of each species")
    for pt, p in pert.items():
        print(f"  {SPECIES[pt][0]}: mean mass {head['masstable'][pt]:.6g} (internal units), "
              f"perturbation mean {p.mean():+.1e}%, rms {p.std():.3f}%, "
              f"min {p.min():+.2f}%, max {p.max():+.2f}%")
    if GAS in pert and DM in pert:
        print(f"  rms gas / rms dark matter = {pert[GAS].std() / pert[DM].std():.3f}  "
              f"(expected f_c / f_b = {(1 - f_b) / f_b:.3f})")

    lim = max(np.abs(p).max() for p in pert.values())
    edges = np.linspace(-lim, lim, args.bins + 1)
    centres = 0.5 * (edges[1:] + edges[:-1])

    plt.rcParams.update({"font.size": 11, "axes.edgecolor": INK_SECONDARY,
                         "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY})
    fig, ax = plt.subplots(figsize=(8, 5.2), facecolor=SURFACE, constrained_layout=True)
    for pt, p in pert.items():
        name, color = SPECIES[pt]
        density = np.histogram(p, bins=edges)[0] / len(p) / np.diff(edges)
        ax.fill_between(centres, density, step="mid", color=color, alpha=0.18, lw=0)
        ax.step(centres, density, where="mid", color=color, lw=1.8,
                label=f"{name}:  rms {p.std():.2f}%,  range {p.min():+.1f}% to {p.max():+.1f}%")
    ax.axvline(0, color=INK_SECONDARY, lw=1, ls=":")
    ax.set_xlabel(r"particle mass relative to the mean of its species,  $m / \langle m \rangle - 1$  [%]")
    ax.set_ylabel("fraction of particles per percent")
    ax.set_ylim(0, 1.2 * ax.get_ylim()[1])  # headroom for the legend
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.suptitle(f"Particle masses of {label_of(path)}:  z = {head['redshift']:g}, "
                 f"{head['box'] * head['mpc']:g} Mpc/h box, {head['ngrid']}³ particles per species",
                 fontsize=12, x=0.01, ha="left")
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
