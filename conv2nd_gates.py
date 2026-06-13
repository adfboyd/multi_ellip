import numpy as np

# CSV layout: 0=time, 1=ke_total, 6=px, 7=py, 8=pz, 9=vx, 10=vy, 11=vz
PY, VZ, KE = 7, 11, 1

def load(f):
    return np.loadtxt(f, delimiter=',', skiprows=1)

def at(d, t, c):
    return d[d[:, 0] <= t + 1e-9][-1][c]

print('================ G2: temporal order at t=2 (new scheme) ================')
n02 = load('conv2nd_nd2_t2_dt0p02.dat')
n01 = load('conv2nd_nd2_t2_dt0p01.dat')
n005 = load('conv2nd_nd2_t2_dt0p005.dat')
g2 = {}
for name, c in (('py', PY), ('vz', VZ)):
    f2, f1, f05 = at(n02, 2, c), at(n01, 2, c), at(n005, 2, c)
    p = np.log2(abs((f1 - f2) / (f05 - f1)))
    g2[name] = p
    print(f'  {name}: dt0.02={f2:+.7f}  dt0.01={f1:+.7f}  dt0.005={f05:+.7f}  ->  p={p:.2f}')
print(f'  GATE (p>=1.7 both): {"PASS" if all(v >= 1.7 for v in g2.values()) else "FAIL"}')

print()
print('================ G3: same dt->0 limit as old scheme ================')
o01 = load('conv25_nd2_dt0p01.dat')   # OLD scheme reference data
o005 = load('conv25_nd2_dt0p005.dat')
ok3 = True
for name, c in (('py', PY), ('vz', VZ)):
    fo1, fo05 = at(o01, 2, c), at(o005, 2, c)
    f_inf_old = 2 * fo05 - fo1          # 1st-order Richardson
    f_new = at(n005, 2, c)
    gap_old = abs(fo05 - f_inf_old)     # old residual at dt=0.005
    diff = abs(f_new - f_inf_old)
    rel = diff / max(abs(f_inf_old), 1e-12)
    passed = (rel < 0.01) or (diff < 4 * gap_old)
    ok3 &= passed
    print(f'  {name}: old f_inf={f_inf_old:+.7f}  new(0.005)={f_new:+.7f}  |diff|={diff:.2e}'
          f'  (old 0.005 gap {gap_old:.2e})  rel={100*rel:.3f}%  {"ok" if passed else "FAIL"}')
print(f'  GATE: {"PASS" if ok3 else "FAIL"}')

print()
print('================ G4: energy drift t=1..25 (ndiv=2) ================')
print('  old scheme: dt0.02 -5.85%   dt0.01 -2.99%   dt0.005 -1.49%')
drifts = {}
for dt, f in (('0.02', 'conv2nd_nd2_t25_dt0p02.dat'),
              ('0.01', 'conv2nd_nd2_t25_dt0p01.dat'),
              ('0.005', 'conv2nd_nd2_t25_dt0p005.dat')):
    d = load(f)
    t, ke = d[:, 0], d[:, KE]
    m = t >= 1.0
    ke0 = ke[m][0]
    drift = 100 * (ke[-1] - ke0) / ke0
    drifts[dt] = drift
    print(f'  new dt={dt:>5}: KE(1)={ke0:.5f}  KE(25)={ke[-1]:.5f}  drift={drift:+.4f}%')
r1 = abs(drifts['0.02'] / drifts['0.01']) if drifts['0.01'] != 0 else float('inf')
r2 = abs(drifts['0.01'] / drifts['0.005']) if drifts['0.005'] != 0 else float('inf')
small = abs(drifts['0.01']) < 0.05 and abs(drifts['0.005']) < 0.05
ratio_ok = small or (2.5 <= r2 <= 8)
mag_ok = abs(drifts['0.01']) <= 0.5
print(f'  ratios: drift(0.02)/drift(0.01)={r1:.2f}  drift(0.01)/drift(0.005)={r2:.2f}')
print(f'  GATE (|drift(0.01)|<=0.5% and ~4x scaling): {"PASS" if (mag_ok and ratio_ok) else "FAIL"}')
