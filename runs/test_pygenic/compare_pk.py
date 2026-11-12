"""Compare measured P(k) of two MP-GenIC style IC bigfiles (per species + total)."""
import sys
import bigfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

A, B = sys.argv[1], sys.argv[2]
INSPEC = sys.argv[3]
OUT = sys.argv[4]
import os
LA = os.path.basename(os.path.dirname(A.rstrip("/")))
LB = os.path.basename(os.path.dirname(B.rstrip("/")))
# A fine mesh with a TSC kernel. Cloud-in-cell on a mesh only twice as fine as the particle
# grid responds differently to particle masses and to displacements, which biases the
# gas-to-DM ratio of mass-perturbed ICs by a few per cent at half the particle Nyquist.
NMESH = 256
# Largest k shown, in units of the fundamental mode (twice the Nyquist of a 64^3 grid)
KMAX = 64


def paint(pos, mass, box, n):
    """Overdensity of particles with the given masses on an n^3 mesh (TSC assignment)."""
    x = (pos % box) / box * n
    i0 = np.rint(x).astype(np.int64)
    d = x - i0
    w = [0.5 * (0.5 - d) ** 2, 0.75 - d ** 2, 0.5 * (0.5 + d) ** 2]
    mass = np.broadcast_to(mass, (len(pos),))
    rho = np.zeros(n ** 3)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                idx = (i0 + [a - 1, b - 1, c - 1]) % n
                flat = (idx[:, 0] * n + idx[:, 1]) * n + idx[:, 2]
                rho += np.bincount(flat, weights=mass * w[a][:, 0] * w[b][:, 1] * w[c][:, 2], minlength=n ** 3)
    rho = rho.reshape((n,) * 3)
    return rho / rho.mean() - 1


def power(delta, box, n):
    dk = np.fft.rfftn(delta) / n ** 3
    kf = np.fft.fftfreq(n, d=1.0 / n)
    kz = np.fft.rfftfreq(n, d=1.0 / n)
    KX, KY, KZ = np.meshgrid(kf, kf, kz, indexing="ij")
    # deconvolve TSC window
    W = (np.sinc(KX / n) * np.sinc(KY / n) * np.sinc(KZ / n)) ** 3
    p = np.abs(dk / W) ** 2 * box ** 3
    kmag = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
    # rfft double counting: kz>0 and kz<n/2 count twice
    wt = np.where((KZ > 0) & (KZ < n / 2), 2.0, 1.0)
    edges = np.arange(0.5, min(n / 2, KMAX) + 1, 1.0)
    ib = np.digitize(kmag.ravel(), edges)
    nb = len(edges)
    cnt = np.bincount(ib, weights=wt.ravel(), minlength=nb + 1)[1:nb]
    ps = np.bincount(ib, weights=(p * wt).ravel(), minlength=nb + 1)[1:nb]
    ks = np.bincount(ib, weights=(kmag * wt).ravel(), minlength=nb + 1)[1:nb]
    good = cnt > 0
    return ks[good] / cnt[good], ps[good] / cnt[good], cnt[good]


def measure(path):
    f = bigfile.File(path)
    at = f["Header"].attrs
    box = float(at["BoxSize"][0])
    mass = at["MassTable"]
    out, deltas, w = {}, {}, {}
    for pt in (0, 1):
        pos = f[f"{pt}/Position"][:]
        # Mass-perturbed ICs have a mass for each particle; otherwise all are equal.
        pmass = f[f"{pt}/Mass"][:].astype(float) if f"{pt}/Mass" in f.blocks else 1.0
        d = paint(pos, pmass, box, NMESH)
        deltas[pt] = d
        w[pt] = mass[pt] * len(pos)
        out[pt] = power(d, box, NMESH)
    tot = (w[0] * deltas[0] + w[1] * deltas[1]) / (w[0] + w[1])
    out["tot"] = power(tot, box, NMESH)
    return box, out, f


boxA, pa, fa = measure(A)
boxB, pb, fb = measure(B)
assert boxA == boxB
box_mpc = boxA / 1e3
# convert k from units of fundamental to h/Mpc, P from (kpc/h)^3 to (Mpc/h)^3
for d in (pa, pb):
    for key, (k, p, c) in d.items():
        d[key] = (k * 2 * np.pi / box_mpc, p / 1e9, c)

ins = np.loadtxt(INSPEC)
kin, pin = ins[:, 0], ins[:, 1] / 1e9  # (kpc/h)^3 -> (Mpc/h)^3

# Particle-level comparison
print("=== Particle-level comparison ===")
for pt in (0, 1):
    for blk in ("ID", "Position", "PrePosition", "Velocity"):
        a = fa[f"{pt}/{blk}"][:]
        b = fb[f"{pt}/{blk}"][:]
        if blk == "Position":
            diff = a - b
            diff -= boxA * np.round(diff / boxA)
        else:
            diff = a.astype(float) - b.astype(float)
        scale = np.abs(b.astype(float)).max() if blk != "Position" else boxA / 64
        print(f"type {pt} {blk:12s}: max|diff| = {np.abs(diff).max():.3e}  "
              f"rms diff = {np.sqrt((diff**2).mean()):.3e}  (ref scale {scale:.3e})")
    da = fa[f"{pt}/Position"][:] - fa[f"{pt}/PrePosition"][:]
    db = fb[f"{pt}/Position"][:] - fb[f"{pt}/PrePosition"][:]
    for x in (da, db):
        x -= boxA * np.round(x / boxA)
    print(f"type {pt} displacement rms: {LA}={np.sqrt((da**2).mean()):.4f}  "
          f"{LB}={np.sqrt((db**2).mean()):.4f} kpc/h; "
          f"corr={np.corrcoef(da.ravel(), db.ravel())[0,1]:.6f}")

print(f"\n=== P(k) ratio {LA} / {LB} ===")
names = {0: "gas", 1: "DM", "tot": "total"}
knyq = np.pi * 64 / box_mpc
for key in (0, 1, "tot"):
    k, p1, _ = pa[key]
    _, p2, _ = pb[key]
    r = p1 / p2
    sel = k < knyq
    print(f"{names[key]:5s}: k<k_Nyq(64)={knyq:.1f}: mean ratio {r[sel].mean():.5f}, "
          f"max |ratio-1| {np.abs(r[sel]-1).max():.3e}")
print(f"\nk[h/Mpc]   gas: {LA}, {LB} | DM: {LA}, {LB} | total: {LA}, {LB} | ratio_tot  P_in")
k = pa["tot"][0]
pint = np.exp(np.interp(np.log(k), np.log(kin), np.log(pin)))
for i in range(0, len(k), 3):
    print(f"{k[i]:8.3f}  " + "  ".join(f"{pa[s][1][i]:.3e}  {pb[s][1][i]:.3e}" for s in (0, 1, "tot"))
          + f"  {pa['tot'][1][i]/pb['tot'][1][i]:.5f}  {pint[i]:.3e}")

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]},
                               constrained_layout=True)
cols = {0: "#d9730d", 1: "#2a6fdb", "tot": "#333333"}
for key in (1, 0, "tot"):
    k, p1, _ = pa[key]
    _, p2, _ = pb[key]
    ax1.loglog(k, p1, "-", color=cols[key], lw=2, label=f"{names[key]} {LA}")
    ax1.loglog(k, p2, "--", color=cols[key], lw=1.5, alpha=0.8, label=f"{names[key]} {LB}")
    ax2.loglog(k, p1 / p2, "-", color=cols[key], lw=1.5, label=names[key])
m = (kin > k.min() / 1.5) & (kin < k.max() * 1.5)
ax1.loglog(kin[m], pin[m], ":", color="k", lw=1, label="input (inputspec)")
for ax in (ax1, ax2):
    ax.axvline(knyq, color="gray", lw=0.8, ls=":")
ax1.set_ylabel(r"$P(k)$ [(Mpc/h)$^3$]")
ax1.legend(ncol=2, fontsize=8, frameon=False)
ax1.set_title(f"IC power spectra, z=99, {box_mpc:g} Mpc/h, TSC {NMESH}³ (deconvolved)")
ax2.axhline(1, color="gray", lw=0.8)
ax2.set_ylabel(f"{LA} / {LB}")
ax2.set_xlabel("k [h/Mpc]")
ax2.legend(fontsize=8, frameon=False)
fig.savefig(OUT, dpi=140)
print("saved", OUT)
