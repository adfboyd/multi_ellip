import numpy as np
import glob

for f in sorted(glob.glob('be_*.dat')):
    eps = f.replace('be_', '').replace('.dat', '')
    d = np.loadtxt(f, delimiter=',', skiprows=1)
    t, ke = d[:, 0], d[:, 1]
    ke1 = ke[t >= 1][0]
    ratio = ke[-1] / ke1
    mvz = np.max(np.abs(d[:, 11]))
    status = 'STABLE' if abs(ratio - 1) < 0.2 else 'BLOWUP'
    print(f'eps={eps:>4}: tmax={t[-1]:.1f} KE(end)/KE(1)={ratio:.4e} max|vz|={mvz:.3f}  {status}')
