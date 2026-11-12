"""Compare the particle velocity or momentum distributions of IC snapshots and a reference IC.

Usage: python plot_velocity.py IC [IC ...] [--momentum] [--ref ICS_ref_grid/IC] [--bins N] [--out FILE]

Each IC is an MP-GenIC / pygenic BigFile (e.g. ICS_grid/IC; the output directory ICS_grid
works too).
There is one column of panels for the dark matter and, if the ICs have gas, one for the
gas. The top panel shows the distribution of the particle speed |v| of each IC and of the
reference, as a fraction of the particles per km/s. The bottom panel shows each
distribution divided by the reference's, where the reference has enough particles per bin.

Velocities are peculiar velocities in km/s. Every particle counts once, whatever its mass.
The legend gives the rms of one component for each IC.

With --momentum the quantity is the momentum of each particle divided by the mean particle
mass of its species, (m / <m>) |v|, also in km/s. Particles have their own mass m if the
IC has a Mass block (PerturbMass = 1); <m> is the header MassTable. For ICs with equal
particle masses this is the same as the speed. A Mass block is ignored if the header says
PerturbMass = 0, as in plot_matter_pk.py.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

from plot_matter_pk import GAS, DM, GRID, INK, INK_SECONDARY, SERIES, SURFACE, label_of, open_ic, read_header

TITLES = {DM: "Dark matter", GAS: "Gas"}
# Bins of the reference with fewer particles than this are left out of the ratio panel.
MIN_COUNT = 100


def read_velocities(f, pt):
    """Peculiar velocities of one particle type in km/s."""
    attrs = f["Header"].attrs
    vel = f[f"{pt}/Velocity"][:].astype("f8") * (float(attrs["UnitVelocity_in_cm_per_s"][0]) / 1e5)
    # Without UsePeculiarVelocity the ICs hold v / sqrt(a), the Gadget-2 convention.
    if "UsePeculiarVelocity" in attrs and not int(attrs["UsePeculiarVelocity"][0]):
        vel *= np.sqrt(float(attrs["Time"][0]))
    return vel


def read_relative_mass(f, pt, head):
    """Mass of each particle of one type in units of the mean particle mass (the MassTable).
    None if all particles have the same mass."""
    if f"{pt}/Mass" not in f.blocks or head["perturbmass"] == 0:
        return None
    return f[f"{pt}/Mass"][:].astype("f8") / head["masstable"][pt]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ics", nargs="+", metavar="IC",
                        help="IC(s) to compare against the reference, e.g. ICS_grid")
    parser.add_argument("--momentum", action="store_true",
                        help="plot the momentum per mean particle mass instead of the speed")
    parser.add_argument("--ref", default="ICS_ref_grid/IC", help="reference IC (default: ICS_ref_grid/IC)")
    parser.add_argument("--bins", type=int, default=60, help="number of bins (default: 60)")
    parser.add_argument("--ylim", type=float, nargs=2, metavar=("LO", "HI"), default=(0.8, 1.2),
                        help="range of the ratio panels (default: 0.8 1.2)")
    parser.add_argument("--out", default=None,
                        help="default: velocity_comparison.png, or momentum_comparison.png")
    args = parser.parse_args()
    if len(args.ics) > len(SERIES):
        raise SystemExit(f"at most {len(SERIES)} ICs can be compared in one figure")
    out = args.out or ("momentum_comparison.png" if args.momentum else "velocity_comparison.png")

    refpath, reffile = open_ic(args.ref)
    refhead = read_header(reffile)
    opened = [open_ic(p) for p in args.ics] + [(refpath, reffile)]
    heads = [read_header(f) for _, f in opened[:-1]] + [refhead]
    labels = [label_of(path) for path, _ in opened[:-1]] + [f"{label_of(refpath)} (reference)"]

    # The gas gets its own column only if every IC has gas.
    species = [DM] + ([GAS] if all(GAS in h["types"] for h in heads) else [])

    # values[pt] is one array per IC: the speed, or the momentum per mean particle mass.
    values, sigma = {}, {}
    print(f"box {refhead['box'] * refhead['mpc']:g} Mpc/h, z = {refhead['redshift']:g}: "
          f"{'momentum per mean particle mass' if args.momentum else 'particle velocities'} in km/s")
    for pt in species:
        values[pt], sigma[pt] = [], []
        print(f"{TITLES[pt]}:")
        for label, (_, f), head in zip(labels, opened, heads):
            vel = read_velocities(f, pt)
            note = ""
            if args.momentum:
                relmass = read_relative_mass(f, pt, head)
                if relmass is None:
                    note = "equal masses, so the same as the velocity"
                else:
                    speed = np.linalg.norm(vel, axis=1)
                    note = (f"masses vary by {relmass.std():.2%} rms; velocity rms "
                            f"{np.sqrt(np.mean(vel ** 2)):.3f}; correlation of mass and speed "
                            f"{np.corrcoef(relmass, speed)[0, 1]:+.3f}; net momentum "
                            f"{np.linalg.norm((relmass[:, None] * vel).mean(axis=0)):.2e}")
                    vel = relmass[:, None] * vel
            values[pt].append(np.linalg.norm(vel, axis=1))
            sigma[pt].append(np.sqrt(np.mean(vel ** 2)))
            v = values[pt][-1]
            print(f"  {label}: rms of one component {sigma[pt][-1]:.3f}, mean {v.mean():.3f}, "
                  f"median {np.median(v):.3f}, max {v.max():.3f}" + (f"  ({note})" if note else ""))

    # All distributions share the same bins, so that they can be divided.
    edges = np.linspace(0, max(v.max() for pt in species for v in values[pt]), args.bins + 1)
    centres = 0.5 * (edges[1:] + edges[:-1])

    plt.rcParams.update({"font.size": 11, "axes.edgecolor": INK_SECONDARY,
                         "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY})
    ncols = len(species)
    fig, axes = plt.subplots(2, ncols, figsize=(1.5 + 6.5 * ncols, 7.5), sharex=True,
                             sharey="row", squeeze=False, facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [2, 1]}, constrained_layout=True)

    # The reference is drawn last, dashed and in neutral ink, on top of the compared ICs.
    styles = [("-", color, 2) for color in SERIES[:len(labels) - 1]] + [("--", INK, 1.5)]
    for pt, (ax1, ax2) in zip(species, axes.T):
        counts = [np.histogram(v, bins=edges)[0] for v in values[pt]]
        enough = counts[-1] >= MIN_COUNT
        for label, count, sig, (ls, color, lw) in zip(labels, counts, sigma[pt], styles):
            density = count / count.sum() / np.diff(edges)
            ax1.plot(centres, density, ls, color=color, lw=lw,
                     label=f"{label}:  σ = {sig:.2f} km/s")
            refdensity = counts[-1] / counts[-1].sum() / np.diff(edges)
            ax2.plot(centres[enough], density[enough] / refdensity[enough], ls, color=color, lw=lw)
        band = ax2.axhspan(0.95, 1.05, color=GRID, lw=0, zorder=0)
        ax2.axhline(1, color=INK_SECONDARY, lw=1, ls=":")
        ax1.set_title(TITLES[pt], fontsize=11, loc="left")
        ax1.legend(frameon=False, fontsize=9.5, title="rms of one component", title_fontsize=9.5,
                   alignment="left")
        ax2.set_xlabel(r"momentum / mean particle mass  $(m / \langle m \rangle)\,|v|$  [km/s]"
                       if args.momentum else r"speed $|v|$  [km/s]")
        for ax in (ax1, ax2):
            ax.set_facecolor(SURFACE)
            ax.grid(color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

    axes[0, 0].set_ylabel("fraction of particles per km/s")
    axes[0, 0].set_ylim(bottom=0)
    axes[1, 0].set_ylabel("distribution / reference")
    axes[1, 0].legend([band], ["±5%"], frameon=False, fontsize=9)
    axes[1, 0].set_ylim(*args.ylim)
    ngrid = min(h["ngrid"] for h in heads)
    fig.suptitle(f"Particle {'momenta' if args.momentum else 'velocities'} of the initial conditions:  "
                 f"z = {refhead['redshift']:g}, {refhead['box'] * refhead['mpc']:g} Mpc/h box, "
                 f"{ngrid}³ particles per species", fontsize=12, x=0.01, ha="left")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
