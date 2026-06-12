import numpy as np

COL = {'py': 7, 'vz': 11}

def load(nd, suf):
    return np.loadtxt(f'conv25_nd{nd}_{suf}.dat', delimiter=',', skiprows=1)

dat = {nd: load(nd, 'dt0p005') for nd in (2, 3, 4)}

def val(d, t, c):
    return d[d[:, 0] <= t + 1e-9][-1][c]

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

def rich(f2, f3, f4):
    d23, d34 = f3 - f2, f4 - f3
    if abs(d23) < 1e-15 or abs(d34) < 1e-15 or d23 * d34 < 0:
        return None
    r = d23 / d34
    try:
        return bis(lambda p: (3**-p - 2**-p) / (4**-p - 3**-p) - r, 0.1, 12.0)
    except Exception:
        return None

print('Early-time SPATIAL order (dt=0.005, ndiv 2/3/4):')
for tc in (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0):
    out = f'  t={tc:>5}:'
    for q in ('py', 'vz'):
        f2 = val(dat[2], tc, COL[q])
        f3 = val(dat[3], tc, COL[q])
        f4 = val(dat[4], tc, COL[q])
        p = rich(f2, f3, f4)
        out += f'  {q} p={p:.2f}' if p else f'  {q} non-monotone'
    print(out)
