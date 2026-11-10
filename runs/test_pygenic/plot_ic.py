"""Plot the DM and gas distribution in the pygenic test ICs.

Usage: python plot_ic.py [path/to/IC] [--slab THICKNESS_KPC] [--out FILE]

Particle type 0 is gas and type 1 is dark matter (MP-Gadget convention).
"""
import argparse

import bigfile
import matplotlib.pyplot as plt
import numpy as np

SPECIES = {0: ("Gas", "Oranges", "#d9730d"), 1: ("Dark matter", "Blues", "#2a6fdb")}


def load(path):
    f = bigfile.File(path)
    attrs = f["Header"].attrs
    box = float(attrs["BoxSize"][0])
    redshift = float(attrs["Redshift"][0])
    data = {}
    for ptype in SPECIES:
        if f"{ptype}/Position" not in f.blocks:
            continue
        data[ptype] = {
            "pos": f[f"{ptype}/Position"][:],
            "prepos": f[f"{ptype}/PrePosition"][:],
            "vel": f[f"{ptype}/Velocity"][:],
        }
    return box, redshift, data


def overdensity(pos, box, nmesh):
    """Cloud-in-cell overdensity 1 + delta on a periodic nmesh^3 mesh.

    CIC (rather than nearest-grid-point) avoids aliasing against the
    particle lattice, which otherwise dominates the signal in the ICs."""
    x = (pos % box) / box * nmesh - 0.5
    i0 = np.floor(x).astype(np.int64)
    f = x - i0
    rho = np.zeros((nmesh,) * 3)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (np.where(dx, f[:, 0], 1 - f[:, 0])
                     * np.where(dy, f[:, 1], 1 - f[:, 1])
                     * np.where(dz, f[:, 2], 1 - f[:, 2]))
                idx = (i0 + [dx, dy, dz]) % nmesh
                np.add.at(rho, (idx[:, 0], idx[:, 1], idx[:, 2]), w)
    return rho / rho.mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ic", nargs="?", default="ICS/IC")
    parser.add_argument("--slab", type=float, default=None,
                        help="slab thickness in z (kpc/h); default: box/8")
    parser.add_argument("--nmesh", type=int, default=None,
                        help="CIC mesh cells per side; default: half the particle grid")
    parser.add_argument("--out", default="ic_distribution.png")
    args = parser.parse_args()

    box, redshift, data = load(args.ic)
    ngrid = round(len(next(iter(data.values()))["pos"]) ** (1 / 3))
    nmesh = args.nmesh or ngrid // 2
    slab = args.slab if args.slab is not None else box / 8
    nslab = max(1, round(slab / box * nmesh))
    slab = nslab * box / nmesh

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), constrained_layout=True)

    # Row 1: projected density maps (shared color scale), plus the 1D PDF.
    rho = {ptype: overdensity(d["pos"], box, nmesh) for ptype, d in data.items()}
    maps = {ptype: r[:, :, :nslab].mean(axis=2) for ptype, r in rho.items()}
    vmin = min(m.min() for m in maps.values())
    vmax = max(m.max() for m in maps.values())

    for col, (ptype, m) in enumerate(maps.items()):
        name, cmap, _ = SPECIES[ptype]
        ax = axes[0, col]
        im = ax.imshow(m.T, origin="lower", extent=[0, box / 1e3, 0, box / 1e3],
                       cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"{name}: projected 1+δ  (z < {slab / 1e3:g} Mpc/h)")
        ax.set_xlabel("x [Mpc/h]")
        ax.set_ylabel("y [Mpc/h]")
        fig.colorbar(im, ax=ax, shrink=0.85, label="1 + δ")

    ax = axes[0, 2]
    for ptype, r in rho.items():
        name, _, color = SPECIES[ptype]
        ax.hist(r.ravel() - 1, bins=80, histtype="step", lw=2, color=color,
                label=f"{name}  σ = {np.std(r):.4f}")
    ax.set_xlabel("δ (CIC, %d³ mesh)" % nmesh)
    ax.set_ylabel("cells")
    ax.set_title("Overdensity PDF")
    ax.legend(frameon=False, loc="upper left")

    # Row 2: thin-slab particle scatter, displacement and velocity distributions.
    ax = axes[1, 0]
    thin = box / ngrid
    for ptype, d in data.items():
        name, _, color = SPECIES[ptype]
        pos = d["pos"] % box
        sel = pos[:, 2] < thin
        ax.scatter(pos[sel, 0] / 1e3, pos[sel, 1] / 1e3, s=10, color=color,
                   alpha=0.8, lw=0, label=name)
    ax.set_xlim(0, box / 4e3)
    ax.set_ylim(0, box / 4e3)
    ax.set_aspect("equal")
    ax.set_xlabel("x [Mpc/h]")
    ax.set_ylabel("y [Mpc/h]")
    ax.set_title(f"Particles in slab z < {thin:.0f} kpc/h (zoom)")
    ax.legend(loc="upper right", markerscale=2, framealpha=0.9)

    ax = axes[1, 1]
    for ptype, d in data.items():
        name, _, color = SPECIES[ptype]
        disp = d["pos"] - d["prepos"]
        disp -= box * np.round(disp / box)
        ax.hist(np.linalg.norm(disp, axis=1), bins=60, histtype="step", lw=2,
                color=color, label=name)
    ax.set_xlabel("|x − q| [kpc/h]")
    ax.set_ylabel("particles")
    ax.set_title("Displacement from grid")
    ax.legend(frameon=False)

    ax = axes[1, 2]
    for ptype, d in data.items():
        name, _, color = SPECIES[ptype]
        ax.hist(np.linalg.norm(d["vel"], axis=1), bins=60, histtype="step", lw=2,
                color=color, label=name)
    ax.set_xlabel("|v| [internal velocity units]")
    ax.set_ylabel("particles")
    ax.set_title("Peculiar velocity")
    ax.legend(frameon=False)

    for ax in axes[:, 1:].flat:
        ax.grid(alpha=0.25, lw=0.5)
    fig.suptitle(f"pygenic test ICs  -  box {box / 1e3:g} Mpc/h, z = {redshift:g}, "
                 f"{ngrid}³ per species")
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
