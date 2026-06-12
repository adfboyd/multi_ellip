import numpy as np

cases = [
    ('conv_nd2_dt0p01.dat', 2),
    ('conv_nd3_dt0p01.dat', 3),
    ('conv_nd4_dt0p01.dat', 4),
]

vals = {}
for fname, n in cases:
    d = np.loadtxt(fname, delimiter=',', skiprows=1)
    row = d[d[:,0] <= 0.25 + 1e-9][-1]
    vals[n] = {'py': row[2], 'vz': row[12], 'vy': row[11],
               'spd': (row[10]**2+row[11]**2+row[12]**2)**0.5}

print("At t=0.25 s  (dt=0.01, temporally converged):")
for n in [2, 3, 4]:
    v = vals[n]
    print(f"  ndiv={n}: py={v['py']:+.6f}  vz={v['vz']:+.6f}  |v|={v['spd']:.6f}")
print()

def bisect_root(fn, lo, hi, tol=1e-12):
    for _ in range(100):
        mid = (lo + hi) / 2
        if fn(lo) * fn(mid) < 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2

for qty in ['py', 'vz']:
    f2, f3, f4 = vals[2][qty], vals[3][qty], vals[4][qty]
    d23 = f3 - f2
    d34 = f4 - f3
    if abs(d34) < 1e-14 or abs(d23) < 1e-14:
        print(f"{qty}: increments too small to fit")
        continue

    # f_n = f_inf + C * n^-p
    # d23 = C*(3^-p - 2^-p),  d34 = C*(4^-p - 3^-p)
    # ratio = d23/d34 = (3^-p - 2^-p)/(4^-p - 3^-p)
    ratio = d23 / d34
    p = bisect_root(lambda p: (3**-p - 2**-p) / (4**-p - 3**-p) - ratio, 0.2, 10.0)

    # C from d23 = C*(3^-p - 2^-p)
    C = d23 / (3**-p - 2**-p)
    # f_inf from f4 = f_inf + C*4^-p
    f_inf = f4 - C * 4**-p

    # Errors relative to extrapolated truth
    for n, fn in [(2, f2), (3, f3), (4, f4)]:
        err = 100 * (fn - f_inf) / f_inf if abs(f_inf) > 1e-12 else float('nan')
        print(f"  ndiv={n}  {qty}={fn:+.6f}  err={err:+.1f}%")
    print(f"  f_inf (Richardson, p={p:.2f}) = {f_inf:+.6f}")
    print()
