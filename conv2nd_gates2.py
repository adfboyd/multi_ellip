import numpy as np

PY, VZ, KE = 7, 11, 1

def load(f):
    return np.loadtxt(f, delimiter=',', skiprows=1)

def at(d, t, c):
    return d[d[:, 0] <= t + 1e-9][-1][c]

print('====== G2 extended: temporal order at t=2, successive dt pairs ======')
runs = {dt: load(f'conv2nd_nd2_t2_dt0p{suf}.dat')
        for dt, suf in ((0.02, '02'), (0.01, '01'), (0.005, '005'), (0.0025, '0025'))}
for name, c in (('py', PY), ('vz', VZ)):
    v = {dt: at(d, 2, c) for dt, d in runs.items()}
    print(f'  {name}: ' + '  '.join(f'dt={dt}: {v[dt]:+.8f}' for dt in (0.02, 0.01, 0.005, 0.0025)))
    p1 = np.log2(abs((v[0.01] - v[0.02]) / (v[0.005] - v[0.01])))
    p2 = np.log2(abs((v[0.005] - v[0.01]) / (v[0.0025] - v[0.005])))
    print(f'        order (0.02/0.01/0.005): p={p1:.2f}    order (0.01/0.005/0.0025): p={p2:.2f}')

print()
print('====== G4 revised: SECULAR drift via linear regression over t=1..25 ======')
print('  (endpoint metric is contaminated by the dt/2 offset between the')
print('   fluid-KE sample (half-step) and solid-KE sample (full step);')
print('   regression over many oscillations isolates the secular trend)')
print('  old scheme endpoint drift: dt0.02 -5.85%  dt0.01 -2.99%  dt0.005 -1.49%')
sl = {}
for dt, f in ((0.02, 'conv2nd_nd2_t25_dt0p02.dat'),
              (0.01, 'conv2nd_nd2_t25_dt0p01.dat'),
              (0.005, 'conv2nd_nd2_t25_dt0p005.dat')):
    d = load(f)
    t, ke = d[:, 0], d[:, KE]
    m = t >= 1.0
    A = np.polyfit(t[m], ke[m], 1)
    ke_mean = ke[m].mean()
    secular = 100 * A[0] * 24 / ke_mean   # % change over t=1..25 from the trend
    sl[dt] = secular
    print(f'  new dt={dt:<6}: secular drift over t=1..25 = {secular:+.4f}%   (KE mean {ke_mean:.5f})')
r = abs(sl[0.01] / sl[0.005]) if sl[0.005] != 0 else float('inf')
r0 = abs(sl[0.02] / sl[0.01]) if sl[0.01] != 0 else float('inf')
print(f'  scaling: drift(0.02)/drift(0.01)={r0:.2f}   drift(0.01)/drift(0.005)={r:.2f}   (2nd order -> ~4)')

print()
print('====== sanity: old-scheme secular drift, same metric ======')
for dt, f in ((0.01, 'conv25_nd2_dt0p01.dat'), (0.005, 'conv25_nd2_dt0p005.dat')):
    d = load(f)
    t, ke = d[:, 0], d[:, KE]
    m = t >= 1.0
    A = np.polyfit(t[m], ke[m], 1)
    secular = 100 * A[0] * 24 / ke[m].mean()
    print(f'  old dt={dt:<6}: secular drift = {secular:+.4f}%')
