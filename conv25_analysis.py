import numpy as np

# New single-body CSV layout: time=0, ke_total=1, ke_fluid=2, ke_solid=3,
# ke_lin_solid=4, ke_rot_solid=5, body0 block from col 6: px=6,py=7,pz=8,vx=9,vy=10,vz=11
COL = {'py': 7, 'vz': 11, 'vx': 9, 'vy': 10, 'ke_total': 1}

def load(nd, suf):
    return np.loadtxt(f'conv25_nd{nd}_{suf}.dat', delimiter=',', skiprows=1)

dat = {(nd, suf): load(nd, suf)
       for nd in (2, 3, 4)
       for suf in ('dt0p02', 'dt0p01', 'dt0p005')}

def val_at(d, t, col):
    r = d[d[:, 0] <= t + 1e-9][-1]
    return r[col]

def bisect(fn, lo, hi, tol=1e-12):
    flo = fn(lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if flo * fn(mid) < 0:
            hi = mid
        else:
            lo, flo = mid, fn(mid)
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)

def richardson3(f2, f3, f4, base=(2, 3, 4)):
    """Fit f_n = f_inf + C n^-p from three refinements n=base."""
    a, b, c = base
    d23, d34 = f3 - f2, f4 - f3
    if abs(d23) < 1e-15 or abs(d34) < 1e-15:
        return None
    ratio = d23 / d34
    try:
        p = bisect(lambda p: (b**-p - a**-p) / (c**-p - b**-p) - ratio, 0.1, 12.0)
    except Exception:
        return None
    C = d23 / (b**-p - a**-p)
    finf = f4 - C * c**-p
    return p, finf

print("=" * 64)
print("SPATIAL convergence (vary ndiv 2,3,4) at fixed dt")
print("=" * 64)
for dt in ('dt0p005', 'dt0p01'):
    print(f"\n  --- dt = {dt} ---")
    for tc in (5.0, 10.0, 25.0):
        for q in ('py', 'vz'):
            f2 = val_at(dat[(2, dt)], tc, COL[q])
            f3 = val_at(dat[(3, dt)], tc, COL[q])
            f4 = val_at(dat[(4, dt)], tc, COL[q])
            r = richardson3(f2, f3, f4)
            if r:
                p, finf = r
                print(f"    t={tc:>5}  {q}:  nd2={f2:+.5f} nd3={f3:+.5f} nd4={f4:+.5f}  ->  p={p:.2f}")
            else:
                print(f"    t={tc:>5}  {q}:  increments too small / non-monotone")

print("\n" + "=" * 64)
print("TEMPORAL convergence (vary dt 0.02,0.01,0.005) at fixed ndiv")
print("=" * 64)
for nd in (4, 3):
    print(f"\n  --- ndiv = {nd} ---")
    for tc in (5.0, 10.0, 25.0):
        for q in ('py', 'vz'):
            f2 = val_at(dat[(nd, 'dt0p02')], tc, COL[q])
            f3 = val_at(dat[(nd, 'dt0p01')], tc, COL[q])
            f4 = val_at(dat[(nd, 'dt0p005')], tc, COL[q])
            # successive halving: order from increment ratio
            d1, d2 = f3 - f2, f4 - f3
            if abs(d2) > 1e-15 and abs(d1) > 1e-15:
                p = np.log2(abs(d1 / d2))
                print(f"    t={tc:>5}  {q}:  dt02={f2:+.5f} dt01={f3:+.5f} dt005={f4:+.5f}  ->  p={p:.2f}")
            else:
                print(f"    t={tc:>5}  {q}:  increments too small")

print("\n" + "=" * 64)
print("ENERGY drift (KE_total, skip t<1 startup transient)")
print("=" * 64)
for nd in (2, 3, 4):
    for suf, dt in (('dt0p02', 0.02), ('dt0p01', 0.01), ('dt0p005', 0.005)):
        d = dat[(nd, suf)]
        t, ke = d[:, 0], d[:, COL['ke_total']]
        m = t >= 1.0
        ke0, keend = ke[m][0], ke[-1]
        drift = 100 * (keend - ke0) / ke0
        print(f"  nd{nd} dt={dt:<6}: KE(1)={ke0:.5f} KE(25)={keend:.5f}  drift={drift:+.3f}%")
