import numpy as np

# BEM .dat cols: 0=t,1=ke_total,6=px,7=py,8=pz,9=vx,10=vy,11=vz
# exact .csv cols: 0=t,1=px,2=py,3=pz,4=vx,5=vy,6=vz,7=ke_total
def bem(f): return np.loadtxt(f, delimiter=',', skiprows=1)
def ex(f):  return np.loadtxt(f, delimiter=',', skiprows=1)

def bv(d, t, c): return d[d[:, 0] <= t + 1e-9][-1][c]
def ev(d, t, c): return d[d[:, 0] <= t + 1e-9][-1][c]

def rms_vel_err(bemd, exd, tmax):
    """RMS over time of |v_BEM - v_exact| up to tmax, sampled on BEM rows."""
    errs = []
    for row in bemd:
        t = row[0]
        if t > tmax: break
        vb = row[[9, 10, 11]]
        ve = np.array([ev(exd, t, 4), ev(exd, t, 5), ev(exd, t, 6)])
        errs.append(np.linalg.norm(vb - ve))
    return np.sqrt(np.mean(np.array(errs) ** 2))

print("="*70)
print("TEST 1 — pure translation (omega(0)=0; Munk develops rotation)")
print("  BEM-default vs exact: should agree at small t, drift as omega grows")
print("="*70)
bt = bem('transtest_nd3.dat'); et = ex('exact_transtest.csv')
print("  t     vz_BEM      vz_exact   |  py_BEM      py_exact")
for t in (0.5, 1, 2, 3, 5):
    print(f"  {t:<4} {bv(bt,t,11):+.6f}  {ev(et,t,6):+.6f}  |  {bv(bt,t,7):+.6f}  {ev(et,t,2):+.6f}")
print(f"  RMS |v_BEM - v_exact| over t=0..2: {rms_vel_err(bt,et,2.0):.5f}")
print(f"  RMS |v_BEM - v_exact| over t=0..5: {rms_vel_err(bt,et,5.0):.5f}")

print()
print("="*70)
print("TEST 2 — rotating (omega=(1,1,0)): does omega x L close the gap?")
print("="*70)
bd = bem('rottest_nd3_default.dat'); btr = bem('rottest_nd3_transport.dat'); er = ex('exact_rottest.csv')
print("  t     vz_default   vz_transport  vz_exact")
for t in (0.25, 0.5, 1, 2, 3, 5):
    print(f"  {t:<5} {bv(bd,t,11):+.6f}   {bv(btr,t,11):+.6f}    {ev(er,t,6):+.6f}")
print()
print("  RMS |v_BEM - v_exact| over t=0..2:")
print(f"    default       : {rms_vel_err(bd,er,2.0):.5f}")
print(f"    +omega x L     : {rms_vel_err(btr,er,2.0):.5f}")
print("  RMS |v_BEM - v_exact| over t=0..5:")
print(f"    default       : {rms_vel_err(bd,er,5.0):.5f}")
print(f"    +omega x L     : {rms_vel_err(btr,er,5.0):.5f}")
print()
print("  KE drift (exact is flat): exact=%.5f  default(t5)=%.5f  transport(t5)=%.5f"
      % (ev(er,5,7), bv(bd,5,1), bv(btr,5,1)))
