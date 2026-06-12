import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

cases = [
    ("conv_nd2_dt0p02.dat",  "ndiv=2  dt=0.020", "C0", "--"),
    ("conv_nd2_dt0p01.dat",  "ndiv=2  dt=0.010", "C1", "-"),
    ("conv_nd2_dt0p005.dat", "ndiv=2  dt=0.005", "C2", ":"),
    ("conv_nd3_dt0p01.dat",  "ndiv=3  dt=0.010", "C3", "-."),
    ("conv_nd3_dt0p005.dat", "ndiv=3  dt=0.005", "C4", (0,(3,1,1,1))),
    ("conv_nd4_dt0p01.dat",  "ndiv=4  dt=0.010", "C5", "-")
]

T_CMP = 0.25

datasets = []
for fname, label, col, ls in cases:
    d = np.loadtxt(fname, delimiter=',', skiprows=1)
    mask = d[:,0] <= T_CMP + 1e-9
    d = d[mask]
    datasets.append({
        "t": d[:,0], "label": label, "col": col, "ls": ls,
        "vx": d[:,10], "vy": d[:,11], "vz": d[:,12],
        "ox": d[:,31], "oy": d[:,32], "oz": d[:,33],
        "o1": d[:,43], "o2": d[:,44], "o3": d[:,45],
        "px": d[:,1],  "py": d[:,2],  "pz": d[:,3],
        "ke_lin": d[:,56], "ke_rot": d[:,57], "ke_fluid": d[:,55],
    })
    spd  = np.sqrt(d[:,10]**2 + d[:,11]**2 + d[:,12]**2)
    omag = np.sqrt(d[:,31]**2 + d[:,32]**2 + d[:,33]**2)
    ke   = d[:,56] + d[:,57] + d[:,55]
    ke0  = ke[d[:,0] >= 0.05][0]
    drift = 100*(ke[-1] - ke0) / ke0
    t_end = d[-1, 0]
    print(f"{label:22s}  steps={len(d)-1:5d}  |v|(t={t_end:.1f})={spd[-1]:.5f}"
          f"  |w|={omag[-1]:.5f}  KE drift={drift:+.2f}%"
          f"  py={d[-1,2]:+.5f}")

fig = plt.figure(figsize=(16, 12))

# ---- 3D ofix sphere path ----
ax3d = fig.add_subplot(2, 3, 1, projection='3d')
u, v = np.linspace(0, 2*np.pi, 30), np.linspace(0, np.pi, 20)
sx = np.outer(np.cos(u), np.sin(v))
sy = np.outer(np.sin(u), np.sin(v))
sz = np.outer(np.ones_like(u), np.cos(v))
ax3d.plot_wireframe(sx, sy, sz, color='lightgray', alpha=0.2, lw=0.3)
for ds in datasets:
    ax3d.plot(ds["o1"], ds["o2"], ds["o3"], color=ds["col"],
              ls=ds["ls"], lw=1.0, label=ds["label"], alpha=0.85)
ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
ax3d.set_title('Body marker on unit sphere (ofix)')
ax3d.legend(fontsize=7, loc='upper left')

# ---- vx ----
ax1 = fig.add_subplot(2, 3, 2)
for ds in datasets:
    ax1.plot(ds["t"], ds["vx"], color=ds["col"], ls=ds["ls"], lw=1.3, label=ds["label"])
ax1.set_xlabel('time (s)'); ax1.set_ylabel('vx (m/s)')
ax1.set_title('Velocity x'); ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)

# ---- vy ----
ax2 = fig.add_subplot(2, 3, 3)
for ds in datasets:
    ax2.plot(ds["t"], ds["vy"], color=ds["col"], ls=ds["ls"], lw=1.3, label=ds["label"])
ax2.set_xlabel('time (s)'); ax2.set_ylabel('vy (m/s)')
ax2.set_title('Velocity y'); ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)

# ---- speed ----
ax4 = fig.add_subplot(2, 3, 4)
for ds in datasets:
    spd = np.sqrt(ds["vx"]**2 + ds["vy"]**2 + ds["vz"]**2)
    ax4.plot(ds["t"], spd, color=ds["col"], ls=ds["ls"], lw=1.3, label=ds["label"])
ax4.set_xlabel('time (s)'); ax4.set_ylabel('|v| (m/s)')
ax4.set_title('Speed |v|  (should be conserved)'); ax4.legend(fontsize=7); ax4.grid(True, alpha=0.3)

# ---- KE total ----
ax5 = fig.add_subplot(2, 3, 5)
for ds in datasets:
    ke = ds["ke_lin"] + ds["ke_rot"] + ds["ke_fluid"]
    mask = ds["t"] >= 0.05
    ax5.plot(ds["t"][mask], ke[mask], color=ds["col"], ls=ds["ls"], lw=1.3, label=ds["label"])
ax5.set_xlabel('time (s)'); ax5.set_ylabel('J')
ax5.set_title('Total KE (skip startup transient)'); ax5.legend(fontsize=7); ax5.grid(True, alpha=0.3)

# ---- vz ----
ax6 = fig.add_subplot(2, 3, 6)
for ds in datasets:
    ax6.plot(ds["t"], ds["vz"], color=ds["col"], ls=ds["ls"], lw=1.3, label=ds["label"])
ax6.set_xlabel('time (s)'); ax6.set_ylabel('vz (m/s)')
ax6.set_title('Velocity z'); ax6.legend(fontsize=7); ax6.grid(True, alpha=0.3)

plt.suptitle(f'Convergence test — rotating ellipsoid (1:0.8:0.6), t=0..{T_CMP} s', fontsize=12)
plt.tight_layout()
fig.savefig('convergence.png', dpi=150, bbox_inches='tight')
print('Saved convergence.png')
