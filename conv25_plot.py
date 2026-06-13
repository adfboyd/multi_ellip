"""
Convergence/energy dashboard for the t=0..25 single-ellipsoid study.

Compares the OLD 1st-order lagged dphi/dt scheme (oldscheme_conv25/) against
the NEW 2nd-order BDF2 scheme, and overlays the exact added-mass (Kirchhoff)
reference (exact_conv25.csv from exact_added_mass.py).

NOTE: the new-scheme ndiv=4 runs are numerically UNSTABLE (explicit
added-mass / BDF2 multistep instability) and blow up, so only ndiv=2,3 are
shown for the new scheme. The old scheme is stable at all ndiv.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# BEM (.dat) single-body layout
B = {'t': 0, 'ke': 1, 'px': 6, 'py': 7, 'pz': 8, 'vx': 9, 'vy': 10, 'vz': 11,
     'ofx': 20, 'ofy': 21, 'ofz': 22}
# exact (.csv) layout
E = {'t': 0, 'px': 1, 'py': 2, 'pz': 3, 'vx': 4, 'vy': 5, 'vz': 6, 'ke': 7,
     'ofx': 10, 'ofy': 11, 'ofz': 12}

def newd(nd, suf): return np.loadtxt(f'conv25_nd{nd}_{suf}.dat', delimiter=',', skiprows=1)
def oldd(nd, suf): return np.loadtxt(f'oldscheme_conv25/conv25_nd{nd}_{suf}.dat', delimiter=',', skiprows=1)
ex = np.loadtxt('exact_conv25.csv', delimiter=',', skiprows=1)

def dev(d, kecol):
    t, ke = d[:, 0], d[:, kecol]
    m = t >= 0.5
    return t[m], 100.0 * (ke[m] - ke[m][0]) / ke[m][0]

def bis(fn, lo, hi):
    flo = fn(lo)
    for _ in range(200):
        m = 0.5 * (lo + hi)
        if flo * fn(m) < 0: hi = m
        else: lo, flo = m, fn(m)
        if hi - lo < 1e-12: break
    return 0.5 * (lo + hi)

def torder(runs, t, col):
    """temporal order from a (dt0.02, dt0.01, dt0.005) triplet at time t."""
    def at(d): return d[d[:, 0] <= t + 1e-9][-1][col]
    f2, f1, f5 = at(runs[0]), at(runs[1]), at(runs[2])
    d1, d2 = f1 - f2, f5 - f1
    if abs(d1) < 1e-14 or abs(d2) < 1e-14 or d1 * d2 < 0: return np.nan
    return np.log2(abs(d1 / d2))

dts = [('dt0p02', 0.02, 'C0'), ('dt0p01', 0.01, 'C1'), ('dt0p005', 0.005, 'C2')]

fig, ax = plt.subplots(2, 3, figsize=(18, 10))

# (0,0) headline: old vs new energy deviation, ndiv=3, three dt
for suf, dt, c in dts:
    t, dv = dev(oldd(3, suf), B['ke']); ax[0, 0].plot(t, dv, c, ls='--', lw=1.0, alpha=0.7)
    t, dv = dev(newd(3, suf), B['ke']); ax[0, 0].plot(t, dv, c, ls='-', lw=1.4, label=f'dt={dt}')
ax[0, 0].axhline(0, color='k', lw=0.8, label='exact (conserved)')
ax[0, 0].set(title='Energy deviation, ndiv=3 — OLD (dashed, 1st-order) vs NEW (solid, 2nd-order)',
             xlabel='t', ylabel='ΔKE/KE₀ (%)')
ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=0.3)

# (0,1) new-scheme energy deviation: nd2 vs nd3 (mesh independence of the small floor)
for nd, c in ((2, 'C0'), (3, 'C3')):
    t, dv = dev(newd(nd, 'dt0p005'), B['ke']); ax[0, 1].plot(t, dv, c, lw=1.3, label=f'new ndiv={nd}')
ax[0, 1].axhline(0, color='k', lw=0.8)
ax[0, 1].set(title='NEW scheme energy deviation (dt=0.005) — small, ~flat',
             xlabel='t', ylabel='ΔKE/KE₀ (%)')
ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=0.3)

# (0,2) temporal order vs time, old vs new (ndiv=3 dt-triplet).
# Early time only: at long t the dt-triplet trajectories diverge and the
# Richardson order estimate becomes meaningless (same caveat as spatial order).
tg = np.arange(0.3, 3.001, 0.05)
new3 = [newd(3, s) for s, _, _ in dts]
old3 = [oldd(3, s) for s, _, _ in dts]
for runs, ls, lab in ((new3, '-', 'new'), (old3, ':', 'old')):
    for col, qc, qn in ((B['py'], 'C3', 'py'), (B['vz'], 'C4', 'vz')):
        p = np.array([torder(runs, t, col) for t in tg])
        ax[0, 2].plot(tg, p, qc, ls=ls, lw=1.3, label=f'{qn} {lab}')
ax[0, 2].axhline(2, color='gray', ls=':', lw=0.8); ax[0, 2].axhline(1, color='gray', ls=':', lw=0.8)
ax[0, 2].set(title='Temporal order vs time (ndiv=3, t≤3): new≈2, old≈1', xlabel='t', ylabel='order p', ylim=(0, 3))
ax[0, 2].legend(fontsize=7, ncol=2); ax[0, 2].grid(alpha=0.3)

# (1,0) vz(t): exact vs new nd2/nd3  (documents the BEM–exact gap)
ax[1, 0].plot(ex[:, E['t']], ex[:, E['vz']], 'k', lw=1.6, label='exact (Kirchhoff)')
for nd, c in ((2, 'C0'), (3, 'C3')):
    d = newd(nd, 'dt0p005'); ax[1, 0].plot(d[:, 0], d[:, B['vz']], c, lw=1.0, label=f'BEM ndiv={nd}')
ax[1, 0].set(title='vz(t): BEM (new) vs exact added-mass — large gap (rotating case)',
             xlabel='t', ylabel='vz'); ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=0.3)

# (1,1) py(t): exact vs new
ax[1, 1].plot(ex[:, E['t']], ex[:, E['py']], 'k', lw=1.6, label='exact')
for nd, c in ((2, 'C0'), (3, 'C3')):
    d = newd(nd, 'dt0p005'); ax[1, 1].plot(d[:, 0], d[:, B['py']], c, lw=1.0, label=f'BEM ndiv={nd}')
ax[1, 1].set(title='py(t): BEM (new) vs exact', xlabel='t', ylabel='py')
ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=0.3)

# (1,2) speed |v|(t): exact vs new
spd_ex = np.sqrt(ex[:, E['vx']]**2 + ex[:, E['vy']]**2 + ex[:, E['vz']]**2)
ax[1, 2].plot(ex[:, E['t']], spd_ex, 'k', lw=1.6, label='exact')
for nd, c in ((2, 'C0'), (3, 'C3')):
    d = newd(nd, 'dt0p005')
    spd = np.sqrt(d[:, B['vx']]**2 + d[:, B['vy']]**2 + d[:, B['vz']]**2)
    ax[1, 2].plot(d[:, 0], spd, c, lw=1.0, label=f'BEM ndiv={nd}')
ax[1, 2].set(title='speed |v|(t)', xlabel='t', ylabel='|v|'); ax[1, 2].legend(fontsize=8); ax[1, 2].grid(alpha=0.3)

plt.suptitle('Single ellipsoid (1:0.8:0.6), t=0..25 — 2nd-order scheme + exact added-mass reference\n'
             '(new-scheme ndiv=4 omitted: BDF2 instability)', fontsize=13)
plt.tight_layout()
fig.savefig('convergence25.png', dpi=130, bbox_inches='tight')
print('Saved convergence25.png')

# ---------------- orbits ----------------
fig2 = plt.figure(figsize=(13, 6))
axA = fig2.add_subplot(1, 2, 1, projection='3d')
axA.plot(ex[:, E['px']], ex[:, E['py']], ex[:, E['pz']], 'k', lw=1.6, label='exact')
for nd, c in ((2, 'C0'), (3, 'C3')):
    d = newd(nd, 'dt0p005'); axA.plot(d[:, B['px']], d[:, B['py']], d[:, B['pz']], c, lw=1.0, label=f'BEM nd{nd}')
axA.set(title='Centre-of-mass path', xlabel='x', ylabel='y', zlabel='z'); axA.legend(fontsize=8)

axB = fig2.add_subplot(1, 2, 2, projection='3d')
u, v = np.linspace(0, 2*np.pi, 40), np.linspace(0, np.pi, 20)
sx = np.outer(np.cos(u), np.sin(v)); sy = np.outer(np.sin(u), np.sin(v)); sz = np.outer(np.ones_like(u), np.cos(v))
axB.plot_wireframe(sx, sy, sz, color='lightgray', alpha=0.25, lw=0.3)
axB.plot(ex[:, E['ofx']], ex[:, E['ofy']], ex[:, E['ofz']], 'k', lw=1.4, label='exact')
for nd, c in ((2, 'C0'), (3, 'C3')):
    d = newd(nd, 'dt0p005'); axB.plot(d[:, B['ofx']], d[:, B['ofy']], d[:, B['ofz']], c, lw=0.9, alpha=0.85, label=f'BEM nd{nd}')
axB.set(title='Orientation marker (ofix) orbit', xlabel='X', ylabel='Y', zlabel='Z'); axB.legend(fontsize=8)

plt.suptitle('Trajectory & orientation orbit vs exact added-mass (dt=0.005)', fontsize=13)
plt.tight_layout()
fig2.savefig('convergence25_orbits.png', dpi=130, bbox_inches='tight')
print('Saved convergence25_orbits.png')
