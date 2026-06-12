import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COL = {'py': 7, 'vz': 11, 'ke_total': 1, 'vx': 9, 'vy': 10}

def load(nd, suf):
    return np.loadtxt(f'conv25_nd{nd}_{suf}.dat', delimiter=',', skiprows=1)

dts = [('dt0p02', 0.02, 'C0'), ('dt0p01', 0.01, 'C1'), ('dt0p005', 0.005, 'C2')]
nds = [(2, 'C0'), (3, 'C1'), (4, 'C2')]

fig, ax = plt.subplots(2, 2, figsize=(13, 9))

# (a) KE(t) at ndiv=4 for the three dt -> drift shrinks with dt
for suf, dt, c in dts:
    d = load(4, suf)
    m = d[:, 0] >= 0.5
    ax[0, 0].plot(d[m, 0], d[m, COL['ke_total']], c, lw=1.2, label=f'dt={dt}')
ax[0, 0].set_title('Total KE vs time  (ndiv=4)  —  drift halves with dt')
ax[0, 0].set_xlabel('t'); ax[0, 0].set_ylabel('KE_total'); ax[0, 0].legend(); ax[0, 0].grid(alpha=0.3)

# (b) KE(t) at dt=0.005 for the three ndiv -> mesh independent (overlap)
for nd, c in nds:
    d = load(nd, 'dt0p005')
    m = d[:, 0] >= 0.5
    ax[0, 1].plot(d[m, 0], d[m, COL['ke_total']], c, lw=1.2, label=f'ndiv={nd}')
ax[0, 1].set_title('Total KE vs time  (dt=0.005)  —  ~mesh independent')
ax[0, 1].set_xlabel('t'); ax[0, 1].set_ylabel('KE_total'); ax[0, 1].legend(); ax[0, 1].grid(alpha=0.3)

# (c) |drift| vs dt log-log at ndiv=4 -> slope 1
dtv = np.array([0.02, 0.01, 0.005])
drift = []
for suf, dt, c in dts:
    d = load(4, suf)
    t, ke = d[:, 0], d[:, COL['ke_total']]
    mm = t >= 1.0
    drift.append(abs(100 * (ke[-1] - ke[mm][0]) / ke[mm][0]))
drift = np.array(drift)
ax[1, 0].loglog(dtv, drift, 'ko-', label='measured')
ax[1, 0].loglog(dtv, drift[-1] * (dtv / dtv[-1])**1, 'r--', label='slope 1 (1st order)')
ax[1, 0].set_title('|energy drift over t=1..25| vs dt  (ndiv=4)')
ax[1, 0].set_xlabel('dt'); ax[1, 0].set_ylabel('|drift| (%)'); ax[1, 0].legend(); ax[1, 0].grid(alpha=0.3, which='both')

# (d) py(t), vz(t) trajectory for the finest run + spatial divergence illustration
d4 = load(4, 'dt0p005')
ax[1, 1].plot(d4[:, 0], d4[:, COL['py']], 'C3', lw=1.0, label='py (ndiv=4)')
ax[1, 1].plot(d4[:, 0], d4[:, COL['vz']], 'C4', lw=1.0, label='vz (ndiv=4)')
d2 = load(2, 'dt0p005')
ax[1, 1].plot(d2[:, 0], d2[:, COL['py']], 'C3', lw=0.8, ls=':', label='py (ndiv=2)')
ax[1, 1].set_title('Trajectory components (finest vs coarse) — diverge at long t')
ax[1, 1].set_xlabel('t'); ax[1, 1].set_ylabel('value'); ax[1, 1].legend(); ax[1, 1].grid(alpha=0.3)

plt.suptitle('Convergence study t=0..25 — single ellipsoid (1:0.8:0.6), rotating', fontsize=13)
plt.tight_layout()
fig.savefig('convergence25.png', dpi=140, bbox_inches='tight')
print('Saved convergence25.png')
