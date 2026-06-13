"""
Exact single-body reference: a free ellipsoid in unbounded potential flow.

For ONE rigid ellipsoid the fluid force is an analytic added-mass effect (Lamb),
so the exact motion is the force-free Kirchhoff system with constant body-frame
effective-mass tensors:

    M = m_s I3 + M_a        (translational: solid + added mass)
    J = I_s + I_a           (rotational: solid + added inertia)

with momenta P = M u_b, Pi = J w_b (body frame). The force-free Kirchhoff
equations are
    dP/dt  = P x w_b
    dPi/dt = Pi x w_b + P x u_b
and the configuration is reconstructed from
    dq/dt  = 1/2 q * (0, w_b)        (q: body->lab orientation)
    dX/dt  = R(q) u_b                (lab COM position)

Total kinetic energy T = 1/2 (u_b.M.u_b + w_b.J.w_b) is conserved exactly, so
this is a clean truth line for the BEM single-body runs (which should converge
to it as ndiv->inf, dt->0). USELESS for multi-body (added mass is then
configuration dependent).

Tensors match hamiltonian.rs: mf_calc (added mass) and if_calc (added inertia).
The shape factors alpha_i = abc * int_0^inf dt / ((a_i^2+t) sqrt(prod(a_j^2+t)))
reproduce mf_calc's alpha/(2-alpha) added masses (sphere -> 1/2 rho V).
"""
import sys
import numpy as np


# ---------- input parsing (same key=value format as the solver) ----------
def parse_input(path):
    vals = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                k, v = line.split('=', 1)
                try:
                    vals[k.strip()] = float(v.strip())
                except ValueError:
                    pass
    return vals


def body_params(vals, i=1):
    g = lambda name, d=0.0: vals.get(f'{name}{i}', d)
    shx, shy, shz = g('shx', 1.0), g('shy', 1.0), g('shz', 1.0)
    req = g('req', 1.0)
    # main.rs normalisation: scale shape so (a b c)^(1/3) == req
    sf = req / (shx * shy * shz) ** (1.0 / 3.0)
    a, b, c = shx * sf, shy * sf, shz * sf
    rho_s = g('rhos', 1.0)
    pos = np.array([g('cex'), g('cey'), g('cez')])
    # orientation quaternion (w, x, y, z), normalised
    q = np.array([g('oriw', 1.0), g('orii'), g('orij'), g('orik')])
    q = q / np.linalg.norm(q)
    u_lab = np.array([g('lvx'), g('lvy'), g('lvz')])
    w_lab = np.array([g('avx'), g('avy'), g('avz')])
    return dict(abc=(a, b, c), rho_s=rho_s, pos=pos, q=q, u_lab=u_lab, w_lab=w_lab)


# ---------- ellipsoid added-mass / added-inertia tensors ----------
_GL_X, _GL_W = np.polynomial.legendre.leggauss(400)  # nodes/weights on [-1,1]

def shape_factors(a, b, c):
    """alpha_i = abc * int_0^inf dt/((a_i^2+t) sqrt(prod(a_j^2+t))) via the
    substitution t = z/(1-z), z in [0,1] (integrand -> 0 smoothly at z=1)."""
    abc = a * b * c
    z = 0.5 * (_GL_X + 1.0)        # map [-1,1] -> [0,1]
    w = 0.5 * _GL_W
    t = z / (1.0 - z)
    jac = 1.0 / (1.0 - z) ** 2     # dt/dz
    root = np.sqrt((a**2 + t) * (b**2 + t) * (c**2 + t))
    def alpha(ax2):
        return abc * np.sum(w * jac / ((ax2 + t) * root))
    return alpha(a**2), alpha(b**2), alpha(c**2)


def tensors(abc, rho_s, rho_f):
    a, b, c = abc
    V = 4.0 / 3.0 * np.pi * a * b * c
    m_s = rho_s * V

    al, be, ga = shape_factors(a, b, c)
    # added mass (mf_calc): V rho_f diag(al/(2-al), ...)
    M_a = rho_f * V * np.diag([al / (2 - al), be / (2 - be), ga / (2 - ga)])
    M = m_s * np.eye(3) + M_a

    # solid inertia (is_calc): m_s/5 diag(b^2+c^2, a^2+c^2, a^2+b^2)
    I_s = m_s / 5.0 * np.diag([b**2 + c**2, a**2 + c**2, a**2 + b**2])

    # added inertia (if_calc): 0.2 V rho_f diag(e1, e2, e3).
    # Each e -> 0 when the two relevant axes are equal (a rotation about the
    # symmetry axis of a spheroid/sphere has no added inertia); guard the 0/0.
    def e_term(d2, fac_diff, sum2):
        num = d2 ** 2 * fac_diff
        den = 2 * d2 + (-fac_diff) * sum2
        return 0.0 if abs(den) < 1e-14 else num / den
    e1 = e_term(b**2 - c**2, ga - be, b**2 + c**2)
    e2 = e_term(a**2 - c**2, ga - al, a**2 + c**2)
    e3 = e_term(a**2 - b**2, be - al, a**2 + b**2)
    I_a = 0.2 * rho_f * V * np.diag([e1, e2, e3])
    J = I_s + I_a

    return dict(V=V, m_s=m_s, M=M, M_a=M_a, I_s=I_s, I_a=I_a, J=J,
                shape_factors=(al, be, ga))


# ---------- quaternion helpers (w, x, y, z); q maps body -> lab ----------
def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w),     2 * (x*z + y*w)],
        [2 * (x*y + z*w),     1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w),     2 * (y*z + x*w),     1 - 2 * (x*x + y*y)],
    ])


def quat_mul(p, q):
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return np.array([
        pw*qw - px*qx - py*qy - pz*qz,
        pw*qx + px*qw + py*qz - pz*qy,
        pw*qy - px*qz + py*qw + pz*qx,
        pw*qz + px*qy - py*qx + pz*qw,
    ])


# ---------- Kirchhoff RHS ----------
def make_rhs(M, J):
    Minv, Jinv = np.linalg.inv(M), np.linalg.inv(J)

    def rhs(t, s):
        P = s[0:3]       # body-frame linear momentum
        Pi = s[3:6]      # body-frame angular momentum
        q = s[6:10]      # orientation (body->lab)
        # X = s[10:13]   # lab position
        u = Minv @ P
        w = Jinv @ Pi
        dP = np.cross(P, w)
        dPi = np.cross(Pi, w) + np.cross(P, u)
        dq = 0.5 * quat_mul(q, np.array([0.0, w[0], w[1], w[2]]))
        dX = quat_to_R(q) @ u
        return np.concatenate([dP, dPi, dq, dX])

    return rhs


def run(input_path, t_end=None, n_out=2001, omega_frame='lab', omega_scale=1.0, q_conj=False):
    vals = parse_input(input_path)
    bp = body_params(vals)
    rho_f = vals.get('rhof', 1.0)
    if t_end is None:
        t_end = vals.get('tend', 25.0)

    T = tensors(bp['abc'], bp['rho_s'], rho_f)
    M, J = T['M'], T['J']
    q0 = bp['q']
    if q_conj:                       # try body<->lab orientation either way
        q0 = np.array([q0[0], -q0[1], -q0[2], -q0[3]])
    R0 = quat_to_R(q0)
    w_lab = bp['w_lab'] * omega_scale

    # ICs: lab-frame velocities -> body frame.
    u_b0 = R0.T @ bp['u_lab']
    if omega_frame == 'lab':
        w_b0 = R0.T @ w_lab
    else:
        w_b0 = w_lab.copy()
    P0 = M @ u_b0
    Pi0 = J @ w_b0
    s0 = np.concatenate([P0, Pi0, q0, bp['pos']])

    rhs = make_rhs(M, J)
    # Fixed-step RK4, normalising the quaternion each output interval. nsub
    # substeps per recorded point keeps the reference far tighter than any BEM run.
    H = t_end / (n_out - 1)
    nsub = 20
    h = H / nsub
    Minv, Jinv = np.linalg.inv(M), np.linalg.inv(J)
    m_s, I_s = T['m_s'], T['I_s']

    def record(t, s):
        P, Pi = s[0:3], s[3:6]
        q = s[6:10] / np.linalg.norm(s[6:10])
        X = s[10:13]
        u_b, w_b = Minv @ P, Jinv @ Pi
        R = quat_to_R(q)
        u_lab = R @ u_b
        ke_total = 0.5 * (u_b @ M @ u_b + w_b @ J @ w_b)
        ke_solid = 0.5 * (m_s * u_lab @ u_lab + w_b @ I_s @ w_b)
        ke_fluid = ke_total - ke_solid
        ofix = R @ np.array([1.0, 0.0, 0.0])   # body x-axis marker in lab
        return [t, *X, *u_lab, ke_total, ke_fluid, ke_solid, *ofix]

    s = s0.copy()
    t = 0.0
    rows = [record(t, s)]
    for _ in range(n_out - 1):
        for _ in range(nsub):
            k1 = rhs(t, s)
            k2 = rhs(t + 0.5 * h, s + 0.5 * h * k1)
            k3 = rhs(t + 0.5 * h, s + 0.5 * h * k2)
            k4 = rhs(t + h, s + h * k3)
            s = s + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        s[6:10] /= np.linalg.norm(s[6:10])   # keep orientation a unit quaternion
        rows.append(record(t, s))

    out = np.array(rows)
    header = "time,px,py,pz,vx,vy,vz,ke_total,ke_fluid,ke_solid,ofx,ofy,ofz"
    return out, header, T


if __name__ == '__main__':
    inp = sys.argv[1] if len(sys.argv) > 1 else 'conv25_nd4_dt0p005.txt'
    out, header, T = run(inp)
    print(f"input: {inp}")
    print(f"  semi-axes a,b,c (from req-normalised shape) baked into tensors")
    print(f"  V={T['V']:.5f}  m_s={T['m_s']:.5f}  shape factors (a,b,c)={tuple(round(x,4) for x in T['shape_factors'])}")
    print(f"  added-mass diag      = {np.round(np.diag(T['M_a']),5)}")
    print(f"  total trans-mass diag= {np.round(np.diag(T['M']),5)}")
    print(f"  solid inertia diag   = {np.round(np.diag(T['I_s']),5)}")
    print(f"  added inertia diag   = {np.round(np.diag(T['I_a']),5)}")
    np.savetxt('exact_reference.csv', out, delimiter=',', header=header, comments='')
    print(f"  wrote exact_reference.csv  ({out.shape[0]} rows, t=0..{out[-1,0]:.1f})")
    print(f"  KE_total drift over run = {1e2*(out[-1,7]-out[0,7])/out[0,7]:+.2e}%  (should be ~0)")
