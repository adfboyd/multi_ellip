import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# New single-body CSV layout
C = {'t': 0, 'ke_total': 1, 'ke_fluid': 2, 'ke_solid': 3, 'ke_lin': 4, 'ke_rot': 5,
     'px': 6, 'py': 7, 'pz': 8, 'vx': 9, 'vy': 10, 'vz': 11,
     'w1': 16, 'w2': 17, 'w3': 18, 'ofx': 20, 'ofy': 21, 'ofz': 22}

def load(nd, suf):
    return np.loadtxt(f'conv25_nd{nd}_{suf}.dat', delimiter=',', skiprows=1)

D = {(nd, suf): load(nd, suf) for nd in (2, 3, 4)
     for suf in ('dt0p02', 'dt0p01', 'dt0p005')}

def val_at(d, t, col):
    return d[d[:, 0] <= t + 1e-9][-1][col]

def ke_dev(d, tref=0.5):
    """Relative deviation (%) of total KE from its early-time (post-startup)
    value, vs time. Skips t<tref to avoid the first-step transient where the
    fluid KE is not yet established."""
    t, ke = d[:, 0], d[:, C['ke_total']]
    mask = t >= tref
    ke0 = ke[mask][0]
    return t[mask], 100.0 * (ke[mask] - ke0) / ke0

def bis(fn, lo, hi):
    flo = fn(lo)
    for _ in range(200):
        m = 0.5 * (lo + hi)
        if flo * fn(m) < 0:
            hi = m
        else:
            lo, flo = m, fn(m)
        if hi - lo < 1e-12:
            break
    return 0.5 * (lo + hi)

def spatial_p(t, col, suf='dt0p005'):
    f2, f3, f4 = (val_at(D[(n, suf)], t, col) for n in (2, 3, 4))
    d23, d34 = f3 - f2, f4 - f3
    if abs(d23) < 1e-14 or abs(d34) < 1e-14 or d23 * d34 < 0:
        return np.nan
    r = d23 / d34
    try:
        return bis(lambda p: (3**-p - 2**-p) / (4**-p - 3**-p) - r, 0.1, 12.0)
    except Exception:
        return np.nan

def temporal_p(t, col, nd=4):
    f2, f3, f4 = (val_at(D[(nd, s)], t, col) for s in ('dt0p02', 'dt0p01', 'dt0p005'))
    d1, d2 = f3 - f2, f4 - f3
    if abs(d1) < 1e-14 or abs(d2) < 1e-14 or d1 * d2 < 0:
        return np.nan
    return np.log2(abs(d1 / d2))

dts = [('dt0p02', 0.02, 'C0'), ('dt0p01', 0.01, 'C1'), ('dt0p005', 0.005, 'C2')]
nds = [(2, 'C0'), (3, 'C1'), (4, 'C2')]
ref = D[(4, 'dt0p005')]   # reference run

# ================= Figure 1: dashboard (3x3) =================
fig, ax = plt.subplots(3, 3, figsize=(18, 14))

# (0,0) energy deviation from initial vs time, ndiv=4, three dt
for suf, dt, c in dts:
    tt, dev = ke_dev(D[(4, suf)])
    ax[0, 0].plot(tt, dev, c, lw=1.1, label=f'dt={dt}')
ax[0, 0].axhline(0, color='gray', lw=0.6)
ax[0, 0].set(title='Energy deviation from initial (ndiv=4) — grows with dt', xlabel='t', ylabel='ΔKE/KE₀ (%)')
ax[0, 0].legend(); ax[0, 0].grid(alpha=0.3)

# (0,1) energy deviation from initial vs time, dt=0.005, three ndiv
for nd, c in nds:
    tt, dev = ke_dev(D[(nd, 'dt0p005')])
    ax[0, 1].plot(tt, dev, c, lw=1.1, label=f'ndiv={nd}')
ax[0, 1].axhline(0, color='gray', lw=0.6)
ax[0, 1].set(title='Energy deviation from initial (dt=0.005) — ~mesh independent', xlabel='t', ylabel='ΔKE/KE₀ (%)')
ax[0, 1].legend(); ax[0, 1].grid(alpha=0.3)

# (0,2) |drift| vs dt log-log
dtv = np.array([0.02, 0.01, 0.005])
drift = []
for suf, dt, c in dts:
    d = D[(4, suf)]; t, ke = d[:, 0], d[:, C['ke_total']]
    mm = t >= 1.0
    drift.append(abs(100 * (ke[-1] - ke[mm][0]) / ke[mm][0]))
drift = np.array(drift)
ax[0, 2].loglog(dtv, drift, 'ko-', label='measured')
ax[0, 2].loglog(dtv, drift[-1] * (dtv / dtv[-1]), 'r--', label='slope 1')
ax[0, 2].set(title='|energy drift t=1..25| vs dt (ndiv=4)', xlabel='dt', ylabel='|drift| %')
ax[0, 2].legend(); ax[0, 2].grid(alpha=0.3, which='both')

# (1,0) KE components for reference run
m = ref[:, 0] >= 0.5
ax[1, 0].plot(ref[m, 0], ref[m, C['ke_total']], 'k', lw=1.3, label='total')
ax[1, 0].plot(ref[m, 0], ref[m, C['ke_lin']], 'C0', lw=1.0, label='solid lin')
ax[1, 0].plot(ref[m, 0], ref[m, C['ke_rot']], 'C1', lw=1.0, label='solid rot')
ax[1, 0].plot(ref[m, 0], ref[m, C['ke_fluid']], 'C2', lw=1.0, label='fluid')
ax[1, 0].set(title='KE components (ndiv=4, dt=0.005)', xlabel='t', ylabel='KE')
ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=0.3)

# (1,1) spatial order vs time
tg = np.arange(0.2, 1.2, 0.01)
sp_py = np.array([spatial_p(t, C['py']) for t in tg])
sp_vz = np.array([spatial_p(t, C['vz']) for t in tg])
ax[1, 1].plot(tg, sp_py, 'C3.', ms=3, label='py')
ax[1, 1].plot(tg, sp_vz, 'C4.', ms=3, label='vz')
ax[1, 1].axhline(2, color='gray', ls=':', lw=0.8); ax[1, 1].axhline(1.5, color='gray', ls=':', lw=0.8)
ax[1, 1].set(title='Spatial order vs time (clean only for t<~1)', xlabel='t', ylabel='order p', ylim=(0, 6))
ax[1, 1].legend(); ax[1, 1].grid(alpha=0.3)

# (1,2) temporal order vs time (full range)
tg_tp = np.arange(0.2, 25.01, 0.2)
tp_py = np.array([temporal_p(t, C['py']) for t in tg_tp])
tp_vz = np.array([temporal_p(t, C['vz']) for t in tg_tp])
ax[1, 2].plot(tg_tp, tp_py, 'C3.', ms=3, label='py')
ax[1, 2].plot(tg_tp, tp_vz, 'C4.', ms=3, label='vz')
ax[1, 2].axhline(1, color='gray', ls=':', lw=0.8)
ax[1, 2].set(title='Temporal order vs time (~1, all times)', xlabel='t', ylabel='order p', ylim=(0, 2.5))
ax[1, 2].legend(); ax[1, 2].grid(alpha=0.3)

# (2,0) velocity components: reference vs coarse
for nd, ls in [(4, '-'), (2, ':')]:
    d = D[(nd, 'dt0p005')]
    lab = f'nd{nd}'
    ax[2, 0].plot(d[:, 0], d[:, C['vx']], 'C0', ls=ls, lw=1.0, label=f'vx {lab}')
    ax[2, 0].plot(d[:, 0], d[:, C['vy']], 'C1', ls=ls, lw=1.0, label=f'vy {lab}')
    ax[2, 0].plot(d[:, 0], d[:, C['vz']], 'C2', ls=ls, lw=1.0, label=f'vz {lab}')
ax[2, 0].set(title='Velocity components (solid nd4, dotted nd2)', xlabel='t', ylabel='v')
ax[2, 0].legend(fontsize=7, ncol=2); ax[2, 0].grid(alpha=0.3)

# (2,1) angular velocity components (reference)
ax[2, 1].plot(ref[:, 0], ref[:, C['w1']], 'C0', lw=1.0, label='wx')
ax[2, 1].plot(ref[:, 0], ref[:, C['w2']], 'C1', lw=1.0, label='wy')
ax[2, 1].plot(ref[:, 0], ref[:, C['w3']], 'C2', lw=1.0, label='wz')
ax[2, 1].set(title='Angular velocity (ndiv=4, dt=0.005)', xlabel='t', ylabel='omega')
ax[2, 1].legend(fontsize=8); ax[2, 1].grid(alpha=0.3)

# (2,2) py(t) for all ndiv -> divergence
for nd, c in nds:
    d = D[(nd, 'dt0p005')]
    ax[2, 2].plot(d[:, 0], d[:, C['py']], c, lw=1.0, label=f'ndiv={nd}')
ax[2, 2].set(title='py(t) across meshes — diverge for t>~2', xlabel='t', ylabel='py')
ax[2, 2].legend(); ax[2, 2].grid(alpha=0.3)

plt.suptitle('Convergence dashboard t=0..25 — single ellipsoid (1:0.8:0.6), rotating', fontsize=15)
plt.tight_layout()
fig.savefig('convergence25.png', dpi=130, bbox_inches='tight')
print('Saved convergence25.png')

# ================= Figure 2: 3D trajectory + ofix orbit =================
fig2 = plt.figure(figsize=(18, 6))

# 3D position path for the three meshes
axA = fig2.add_subplot(1, 3, 1, projection='3d')
for nd, c in nds:
    d = D[(nd, 'dt0p005')]
    axA.plot(d[:, C['px']], d[:, C['py']], d[:, C['pz']], c, lw=1.0, label=f'ndiv={nd}')
axA.set(title='Centre-of-mass path', xlabel='x', ylabel='y', zlabel='z')
axA.legend(fontsize=8)

# ofix orbit on unit sphere (orientation marker), three meshes
axB = fig2.add_subplot(1, 3, 2, projection='3d')
u, v = np.linspace(0, 2 * np.pi, 40), np.linspace(0, np.pi, 20)
sx = np.outer(np.cos(u), np.sin(v)); sy = np.outer(np.sin(u), np.sin(v)); sz = np.outer(np.ones_like(u), np.cos(v))
axB.plot_wireframe(sx, sy, sz, color='lightgray', alpha=0.25, lw=0.3)
for nd, c in nds:
    d = D[(nd, 'dt0p005')]
    axB.plot(d[:, C['ofx']], d[:, C['ofy']], d[:, C['ofz']], c, lw=0.8, alpha=0.85, label=f'ndiv={nd}')
axB.set(title='Orientation marker (ofix) orbit on unit sphere', xlabel='X', ylabel='Y', zlabel='Z')
axB.legend(fontsize=8)

# ofix orbit: early time only (t<=3, before divergence) to show clean convergence
axC = fig2.add_subplot(1, 3, 3, projection='3d')
axC.plot_wireframe(sx, sy, sz, color='lightgray', alpha=0.25, lw=0.3)
for nd, c in nds:
    d = D[(nd, 'dt0p005')]
    e = d[d[:, 0] <= 3.0]
    axC.plot(e[:, C['ofx']], e[:, C['ofy']], e[:, C['ofz']], c, lw=1.3, label=f'ndiv={nd}')
axC.set(title='ofix orbit, t<=3 (meshes still agree)', xlabel='X', ylabel='Y', zlabel='Z')
axC.legend(fontsize=8)

plt.suptitle('Trajectory & orientation orbits (dt=0.005)', fontsize=14)
plt.tight_layout()
fig2.savefig('convergence25_orbits.png', dpi=130, bbox_inches='tight')
print('Saved convergence25_orbits.png')
