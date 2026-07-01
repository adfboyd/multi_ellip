//! Reduced Kirchhoff-metric integrator, milestone 2: body-frame Kirchhoff
//! equations with an implicit-midpoint discretization.
//!
//! Fixes the milestone-1 prototype's rotational drift. The prototype
//! differentiated the kinetic energy w.r.t. orientation at fixed *lab* angular
//! velocity, which double-counts the R-omega kinematic link and injects a
//! spurious -omega x L torque for anisotropic bodies. Here we work in the body
//! frame, where each body's self-metric block is constant, and evolve the
//! Kirchhoff equations
//!     dP_b/dt = P_b x Omega_b + F_b^cfg
//!     dPi_b/dt = Pi_b x Omega_b + P_b x V_b + T_b^cfg
//! with (P_b, Pi_b) the body-frame linear/angular impulse-momenta and
//! (V_b, Omega_b) the body-frame velocities. F/T^cfg are the body-frame
//! configuration forces from the *interaction* part of the metric (zero for a
//! single body, so a free body reduces to the classic Kirchhoff top and
//! implicit midpoint conserves energy essentially exactly).
//!
//! Metric: M_lab(q) = M_a(q) + M_body assembled as before; body-frame metric
//! M_bf = T(R)^T M_lab T(R), T = blkdiag(R_b, R_b) per body. Momentum
//! mu = M_bf zeta, zeta = (V_b, Omega_b).

use multi_ellip::bem::bem_for_ode::{BemSolver, BodyState};
use multi_ellip::ellipsoids::body::Body;
use multi_ellip::system::fluid::Fluid;
use multi_ellip::system::hamiltonian::is_calc;
use multi_ellip::system::system::Simulation;
use nalgebra::{DMatrix, DVector, Matrix3, Quaternion, UnitQuaternion, Vector3};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::time::Instant;

fn read_values(path: &str) -> HashMap<String, f64> {
    let text = fs::read_to_string(path).expect("failed to read input");
    let mut values = HashMap::new();
    for line in text.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        values.insert(
            key.trim().to_string(),
            value.trim().parse::<f64>().expect("failed to parse value"),
        );
    }
    values
}

fn get(values: &HashMap<String, f64>, key: &str, default: f64) -> f64 {
    *values.get(key).unwrap_or(&default)
}

fn read_body(values: &HashMap<String, f64>, i: usize) -> Body {
    let g = |name: &str, default: f64| -> f64 {
        *values.get(&format!("{}{}", name, i)).unwrap_or(&default)
    };
    let position = Vector3::new(g("cex", 0.0), g("cey", 0.0), g("cez", 0.0));
    let orientation = Quaternion::from_parts(
        g("oriw", 1.0),
        Vector3::new(g("orii", 0.0), g("orij", 0.0), g("orik", 0.0)),
    )
    .normalize();
    let lin_velocity = Vector3::new(g("lvx", 0.0), g("lvy", 0.0), g("lvz", 0.0));
    let ang_velocity = Quaternion::from_parts(
        0.0,
        Vector3::new(g("avx", 0.0), g("avy", 0.0), g("avz", 0.0)),
    );
    let (shx, shy, shz) = (g("shx", 1.0), g("shy", 1.0), g("shz", 1.0));
    let req = g("req", 1.0);
    let scale = req / (shx * shy * shz).powf(1.0 / 3.0);
    let shape = Vector3::new(shx * scale, shy * scale, shz * scale);
    let rho_s = g("rhos", 1.0);
    let mut body = Body {
        density: rho_s,
        shape,
        position,
        orientation,
        linear_momentum: lin_velocity,
        angular_momentum: ang_velocity,
        inertia: is_calc(Matrix3::from_diagonal(&shape), rho_s),
    };
    body.set_linear_velocity(lin_velocity);
    body.set_angular_velocity(ang_velocity);
    body
}

#[derive(Clone)]
struct Config {
    pos: Vec<Vector3<f64>>,
    ori: Vec<UnitQuaternion<f64>>,
}
impl Config {
    fn nbody(&self) -> usize {
        self.pos.len()
    }
}

struct BodyProps {
    mass: f64,
    inertia_body: Matrix3<f64>,
}

/// Build a solver BodyState from a config plus a *lab-frame* generalized
/// velocity `z_lab` (v_lab, omega_lab per body).
fn body_state_from(cfg: &Config, z_lab: &DVector<f64>) -> BodyState {
    let n = cfg.nbody();
    let mut pos = DVector::zeros(3 * n);
    let mut vel = DVector::zeros(3 * n);
    let mut q = Vec::with_capacity(n);
    let mut omega = Vec::with_capacity(n);
    for b in 0..n {
        for c in 0..3 {
            pos[3 * b + c] = cfg.pos[b][c];
            vel[3 * b + c] = z_lab[6 * b + c];
        }
        q.push(*cfg.ori[b].quaternion());
        omega.push(Quaternion::from_imag(Vector3::new(
            z_lab[6 * b + 3],
            z_lab[6 * b + 4],
            z_lab[6 * b + 5],
        )));
    }
    BodyState {
        lin: (pos, vel),
        ang: (q, omega),
    }
}

/// Convert body-frame zeta = (V_b, Omega_b) to lab z = (R V_b, R Omega_b).
fn body_to_lab_vel(cfg: &Config, zeta: &DVector<f64>) -> DVector<f64> {
    let n = cfg.nbody();
    let mut z = DVector::zeros(6 * n);
    for b in 0..n {
        let r = cfg.ori[b].to_rotation_matrix();
        let v = Vector3::new(zeta[6 * b], zeta[6 * b + 1], zeta[6 * b + 2]);
        let w = Vector3::new(zeta[6 * b + 3], zeta[6 * b + 4], zeta[6 * b + 5]);
        let vl = r * v;
        let wl = r * w;
        for c in 0..3 {
            z[6 * b + c] = vl[c];
            z[6 * b + 3 + c] = wl[c];
        }
    }
    z
}

/// Assemble lab-frame metric M_lab = M_a + M_body at `cfg` (M_a via unit-velocity
/// impulses, columns negated).
fn assemble_m_lab(solver: &mut BemSolver, cfg: &Config, props: &[BodyProps]) -> DMatrix<f64> {
    let n = cfg.nbody();
    let dof = 6 * n;
    let mut m = DMatrix::zeros(dof, dof);
    for j in 0..dof {
        let mut ej = DVector::zeros(dof);
        ej[j] = 1.0;
        solver.set_state(&body_state_from(cfg, &ej));
        let (l_lin, l_ang) = solver.impulse();
        for b in 0..n {
            for c in 0..3 {
                m[(6 * b + c, j)] = -l_lin[3 * b + c];
                m[(6 * b + 3 + c, j)] = -l_ang[3 * b + c];
            }
        }
    }
    let mt = m.transpose();
    m = 0.5 * (&m + &mt);
    for b in 0..n {
        let r = cfg.ori[b].to_rotation_matrix();
        let ang = r.matrix() * props[b].inertia_body * r.matrix().transpose();
        for rr in 0..3 {
            for cc in 0..3 {
                if rr == cc {
                    m[(6 * b + rr, 6 * b + cc)] += props[b].mass;
                }
                m[(6 * b + 3 + rr, 6 * b + 3 + cc)] += ang[(rr, cc)];
            }
        }
    }
    m
}

/// Body-frame metric M_bf = T^T M_lab T, T = blkdiag(R_b, R_b).
fn to_body_frame(m_lab: &DMatrix<f64>, cfg: &Config) -> DMatrix<f64> {
    let n = cfg.nbody();
    let dof = 6 * n;
    let mut t = DMatrix::zeros(dof, dof);
    for b in 0..n {
        let r = cfg.ori[b].to_rotation_matrix();
        for i in 0..3 {
            for j in 0..3 {
                t[(6 * b + i, 6 * b + j)] = r.matrix()[(i, j)];
                t[(6 * b + 3 + i, 6 * b + 3 + j)] = r.matrix()[(i, j)];
            }
        }
    }
    &t.transpose() * m_lab * &t
}

/// Total kinetic energy at (cfg, body-frame zeta): solid analytic + fluid solve.
/// Fluid KE = 0.5 z_lab^T M_a z_lab from one impulse solve.
fn total_ke(
    solver: &mut BemSolver,
    cfg: &Config,
    props: &[BodyProps],
    zeta: &DVector<f64>,
) -> f64 {
    let z_lab = body_to_lab_vel(cfg, zeta);
    solver.set_state(&body_state_from(cfg, &z_lab));
    let _ = solver.impulse();
    let fluid = solver.fluid_kinetic_energy();
    let mut solid = 0.0;
    for b in 0..cfg.nbody() {
        let v = Vector3::new(zeta[6 * b], zeta[6 * b + 1], zeta[6 * b + 2]);
        let w = Vector3::new(zeta[6 * b + 3], zeta[6 * b + 4], zeta[6 * b + 5]);
        solid += 0.5 * props[b].mass * v.dot(&v) + 0.5 * w.dot(&(props[b].inertia_body * w));
    }
    solid + fluid
}

/// Body-frame configuration force (F_b, T_b) per body: d/d(body-pose_b) of the
/// total KE at fixed body-frame zeta, central differences. Body-frame
/// translation of body b along axis c: x_b -> x_b + delta * R_b e_c. Body-frame
/// rotation: R_b -> R_b exp(delta e_c). Zero for a single body (metric const).
fn config_force(
    solver: &mut BemSolver,
    cfg: &Config,
    props: &[BodyProps],
    zeta: &DVector<f64>,
    eps: f64,
    project: bool,
) -> DVector<f64> {
    let n = cfg.nbody();
    let dof = 6 * n;
    let mut f = DVector::zeros(dof);
    if n < 2 {
        return f; // no interaction, no configuration force
    }
    for b in 0..n {
        let r = cfg.ori[b].to_rotation_matrix();
        for c in 0..3 {
            // translation along body axis c (lab direction R e_c)
            let mut cfg_p = cfg.clone();
            let mut cfg_m = cfg.clone();
            let dir = r * Vector3::new(
                if c == 0 { 1.0 } else { 0.0 },
                if c == 1 { 1.0 } else { 0.0 },
                if c == 2 { 1.0 } else { 0.0 },
            );
            cfg_p.pos[b] += eps * dir;
            cfg_m.pos[b] -= eps * dir;
            let ep = total_ke(solver, &cfg_p, props, zeta);
            let em = total_ke(solver, &cfg_m, props, zeta);
            f[6 * b + c] = (ep - em) / (2.0 * eps);
        }
        for c in 0..3 {
            // body-frame rotation about axis c
            let mut axis = Vector3::zeros();
            axis[c] = eps;
            let dqp = UnitQuaternion::from_scaled_axis(axis);
            let dqm = UnitQuaternion::from_scaled_axis(-axis);
            let mut cfg_p = cfg.clone();
            let mut cfg_m = cfg.clone();
            cfg_p.ori[b] = cfg.ori[b] * dqp;
            cfg_m.ori[b] = cfg.ori[b] * dqm;
            let ep = total_ke(solver, &cfg_p, props, zeta);
            let em = total_ke(solver, &cfg_m, props, zeta);
            f[6 * b + 3 + c] = (ep - em) / (2.0 * eps);
        }
    }
    // 2c: project out the 6 global rigid-motion directions. The interaction
    // energy is invariant under a rigid motion of the whole configuration (at
    // fixed body-frame zeta), so the true config force is orthogonal to those
    // directions; the finite difference violates this at truncation order and
    // that violation is exactly the spurious net force/torque that drifts the
    // momentum. Removing it enforces sum_b F_b = 0 and sum_b (x_b x F_b + T_b)=0.
    if project {
        project_out_rigid_modes(&mut f, cfg);
    }
    f
}

/// The 6 infinitesimal global rigid-motion directions expressed in the
/// body-frame perturbation coordinates used by `config_force` (translation
/// coordinate `6b+c` = lab move `R_b e_c`; rotation coordinate `6b+3+c` =
/// body-frame right rotation about axis c). Global lab translation `a` ->
/// per body body-frame translation `R_b^T a`; global lab rotation `theta` about
/// the origin -> body-frame translation `R_b^T (theta x x_b)` and body-frame
/// rotation `R_b^T theta`.
fn rigid_mode_directions(cfg: &Config) -> Vec<DVector<f64>> {
    let n = cfg.nbody();
    let dof = 6 * n;
    let mut dirs = Vec::with_capacity(6);
    for axis in 0..3 {
        let mut a = Vector3::zeros();
        a[axis] = 1.0;
        let mut d = DVector::zeros(dof);
        for b in 0..n {
            let rt = cfg.ori[b].to_rotation_matrix().transpose();
            let tb = rt.matrix() * a;
            for c in 0..3 {
                d[6 * b + c] = tb[c];
            }
        }
        dirs.push(d);
    }
    for axis in 0..3 {
        let mut theta = Vector3::zeros();
        theta[axis] = 1.0;
        let mut d = DVector::zeros(dof);
        for b in 0..n {
            let rt = cfg.ori[b].to_rotation_matrix().transpose();
            let trans = rt.matrix() * theta.cross(&cfg.pos[b]);
            let rot = rt.matrix() * theta;
            for c in 0..3 {
                d[6 * b + c] = trans[c];
                d[6 * b + 3 + c] = rot[c];
            }
        }
        dirs.push(d);
    }
    dirs
}

/// Remove the component of `f` along the global rigid-motion subspace (Euclidean
/// projection via Gram-Schmidt orthonormalisation of the mode directions).
fn project_out_rigid_modes(f: &mut DVector<f64>, cfg: &Config) {
    let dirs = rigid_mode_directions(cfg);
    let mut basis: Vec<DVector<f64>> = Vec::with_capacity(dirs.len());
    for d in dirs {
        let mut u = d;
        for b in &basis {
            let proj = b.dot(&u);
            u -= proj * b;
        }
        let nrm = u.norm();
        if nrm > 1.0e-10 {
            u /= nrm;
            basis.push(u);
        }
    }
    for b in &basis {
        let c = b.dot(f);
        *f -= c * b;
    }
}

/// Kirchhoff RHS (body frame) given midpoint momentum mu and velocity zeta and
/// configuration force f_cfg: per body
///   dP = P x Omega + F,  dPi = Pi x Omega + P x V + T.
fn kirchhoff_rhs(mu: &DVector<f64>, zeta: &DVector<f64>, f_cfg: &DVector<f64>) -> DVector<f64> {
    let n = mu.len() / 6;
    let mut d = DVector::zeros(6 * n);
    for b in 0..n {
        let p = Vector3::new(mu[6 * b], mu[6 * b + 1], mu[6 * b + 2]);
        let pi = Vector3::new(mu[6 * b + 3], mu[6 * b + 4], mu[6 * b + 5]);
        let v = Vector3::new(zeta[6 * b], zeta[6 * b + 1], zeta[6 * b + 2]);
        let w = Vector3::new(zeta[6 * b + 3], zeta[6 * b + 4], zeta[6 * b + 5]);
        let f = Vector3::new(f_cfg[6 * b], f_cfg[6 * b + 1], f_cfg[6 * b + 2]);
        let t = Vector3::new(f_cfg[6 * b + 3], f_cfg[6 * b + 4], f_cfg[6 * b + 5]);
        let dp = p.cross(&w) + f;
        let dpi = pi.cross(&w) + p.cross(&v) + t;
        for c in 0..3 {
            d[6 * b + c] = dp[c];
            d[6 * b + 3 + c] = dpi[c];
        }
    }
    d
}

fn advance_config(cfg: &Config, zeta_mid: &DVector<f64>, dt: f64) -> Config {
    let mut out = cfg.clone();
    for b in 0..cfg.nbody() {
        let v = Vector3::new(zeta_mid[6 * b], zeta_mid[6 * b + 1], zeta_mid[6 * b + 2]);
        let w = Vector3::new(zeta_mid[6 * b + 3], zeta_mid[6 * b + 4], zeta_mid[6 * b + 5]);
        // midpoint rotation for the lab displacement of the body-frame velocity
        let axis = w * (0.5 * dt);
        let r_mid = cfg.ori[b] * UnitQuaternion::from_scaled_axis(axis);
        out.pos[b] = cfg.pos[b] + dt * (r_mid.to_rotation_matrix() * v);
        out.ori[b] = cfg.ori[b] * UnitQuaternion::from_scaled_axis(w * dt);
    }
    out
}

fn half_config(cfg: &Config, zeta_mid: &DVector<f64>, dt: f64) -> Config {
    let mut out = cfg.clone();
    for b in 0..cfg.nbody() {
        let v = Vector3::new(zeta_mid[6 * b], zeta_mid[6 * b + 1], zeta_mid[6 * b + 2]);
        let w = Vector3::new(zeta_mid[6 * b + 3], zeta_mid[6 * b + 4], zeta_mid[6 * b + 5]);
        let axis = w * (0.25 * dt);
        let r_q = cfg.ori[b] * UnitQuaternion::from_scaled_axis(axis);
        out.pos[b] = cfg.pos[b] + 0.5 * dt * (r_q.to_rotation_matrix() * v);
        out.ori[b] = cfg.ori[b] * UnitQuaternion::from_scaled_axis(w * (0.5 * dt));
    }
    out
}

fn solve(m: &DMatrix<f64>, rhs: &DVector<f64>) -> DVector<f64> {
    m.clone().lu().solve(rhs).expect("metric solve failed")
}

fn total_linear_momentum_lab(cfg: &Config, mu: &DVector<f64>) -> Vector3<f64> {
    // P is body frame; lab linear momentum = R_b P_b.
    let mut s = Vector3::zeros();
    for b in 0..cfg.nbody() {
        let r = cfg.ori[b].to_rotation_matrix();
        s += r * Vector3::new(mu[6 * b], mu[6 * b + 1], mu[6 * b + 2]);
    }
    s
}

/// Total spatial angular momentum about the lab origin:
/// sum_b [ R_b Pi_b + x_b x (R_b P_b) ]. Conserved by the exact dynamics.
fn total_angular_momentum_lab(cfg: &Config, mu: &DVector<f64>) -> Vector3<f64> {
    let mut s = Vector3::zeros();
    for b in 0..cfg.nbody() {
        let r = cfg.ori[b].to_rotation_matrix();
        let p_lab = r * Vector3::new(mu[6 * b], mu[6 * b + 1], mu[6 * b + 2]);
        let pi_lab = r * Vector3::new(mu[6 * b + 3], mu[6 * b + 4], mu[6 * b + 5]);
        s += pi_lab + cfg.pos[b].cross(&p_lab);
    }
    s
}

/// SE(3) coadjoint Ad*_f acting on a single body's momentum mu_b = (P, Pi):
///   Ad*_f mu = ( R_f^T P,  R_f^T (Pi + P x p_f) )
/// with f = (R_f, p_f) the body-frame relative increment. This is the exact
/// group coadjoint; composing two half-increments reproduces the full step and
/// keeps the spatial momentum map Ad*_{g^-1} mu invariant (milestone 2b).
fn ad_star_se3(
    rf: &Matrix3<f64>,
    pf: &Vector3<f64>,
    p: &Vector3<f64>,
    pi: &Vector3<f64>,
) -> (Vector3<f64>, Vector3<f64>) {
    let p_new = rf.transpose() * p;
    let pi_new = rf.transpose() * (pi + p.cross(pf));
    (p_new, pi_new)
}

/// Exact coadjoint transport of the stacked momentum from configuration
/// `cfg_a` to `cfg_b`, using the true per-body relative increment
/// f = (R_a^T R_b, R_a^T (x_b - x_a)). Because these legs are the exact group
/// increments, composing leg (a->half) with leg (half->b) reproduces the full
/// increment (a->b) exactly, so the spatial momentum map is preserved to machine
/// precision (unlike an approximate exp-of-average increment).
fn transport_exact(mu: &DVector<f64>, cfg_a: &Config, cfg_b: &Config) -> DVector<f64> {
    let n = cfg_a.nbody();
    let mut out = DVector::zeros(6 * n);
    for b in 0..n {
        let ra = cfg_a.ori[b].to_rotation_matrix();
        let rb = cfg_b.ori[b].to_rotation_matrix();
        let rf = ra.matrix().transpose() * rb.matrix();
        let pf = ra.matrix().transpose() * (cfg_b.pos[b] - cfg_a.pos[b]);
        let p = Vector3::new(mu[6 * b], mu[6 * b + 1], mu[6 * b + 2]);
        let pi = Vector3::new(mu[6 * b + 3], mu[6 * b + 4], mu[6 * b + 5]);
        let (p_new, pi_new) = ad_star_se3(&rf, &pf, &p, &pi);
        for c in 0..3 {
            out[6 * b + c] = p_new[c];
            out[6 * b + 3 + c] = pi_new[c];
        }
    }
    out
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        panic!("usage: reduced_metric_kirchhoff <input.txt> [cfg_eps] [fp_iters] [--vi]");
    }
    let input = &args[1];
    let use_vi = args.iter().any(|s| s == "--vi");
    let project = !args.iter().any(|s| s == "--noproject");
    let numeric: Vec<&String> = args
        .iter()
        .skip(2)
        .filter(|s| !s.starts_with("--"))
        .collect();
    let cfg_eps = numeric
        .first()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(1.0e-4);
    let fp_iters = numeric
        .get(1)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(20);

    let values = read_values(input);
    let nbody = get(&values, "nbody", 2.0) as usize;
    let ndiv = get(&values, "ndiv", 2.0) as u32;
    let rho_f = get(&values, "rhof", 1.0);
    let dt = get(&values, "dt", 0.05);
    let tend = get(&values, "tend", 1.0);
    let nsteps = (tend / dt).round() as usize;
    let tprint = (get(&values, "tprint", 1.0) as usize).max(1);

    let bodies: Vec<Body> = (1..=nbody).map(|i| read_body(&values, i)).collect();
    let props: Vec<BodyProps> = bodies
        .iter()
        .map(|b| BodyProps {
            mass: b.mass(),
            inertia_body: b.inertia,
        })
        .collect();

    let mut cfg = Config {
        pos: bodies.iter().map(|b| b.position).collect(),
        ori: bodies
            .iter()
            .map(|b| UnitQuaternion::from_quaternion(b.orientation))
            .collect(),
    };
    // Initial body-frame velocity zeta from lab velocities.
    let mut zeta = DVector::zeros(6 * nbody);
    for (b, body) in bodies.iter().enumerate() {
        let r = cfg.ori[b].to_rotation_matrix();
        let v = r.transpose() * body.linear_velocity();
        let w = r.transpose() * body.angular_velocity().imag();
        for c in 0..3 {
            zeta[6 * b + c] = v[c];
            zeta[6 * b + 3 + c] = w[c];
        }
    }

    let fluid = Fluid {
        density: rho_f,
        kinetic_energy: 0.0,
    };
    let sys = Simulation::new(fluid, bodies.clone(), ndiv);
    let mut solver = BemSolver::new(sys);

    let scheme = if use_vi {
        "milestone 2b, Ad* variational transport"
    } else {
        "milestone 2, implicit-midpoint ODE"
    };
    println!("reduced Kirchhoff-metric integrator ({scheme})");
    println!("input={input}  nbody={nbody}  ndiv={ndiv}  dt={dt}  tend={tend}  steps={nsteps}");
    println!("cfg_eps={cfg_eps:.3e}  fp_iters={fp_iters}");
    println!();

    let m_bf0 = to_body_frame(&assemble_m_lab(&mut solver, &cfg, &props), &cfg);
    let mut mu = &m_bf0 * &zeta;
    let e0 = 0.5 * zeta.dot(&mu);
    let plin0 = total_linear_momentum_lab(&cfg, &mu);
    let pang0 = total_angular_momentum_lab(&cfg, &mu);

    println!(
        "{:>7} {:>14} {:>13} {:>11} {:>11} {:>10}",
        "t", "E_total", "E_drift_rel", "|dPlin|", "|dPang|", "fp_res"
    );
    println!(
        "{:7.3} {:14.8} {:13.3e} {:11.3e} {:11.3e} {:>10}",
        0.0, e0, 0.0, 0.0, 0.0, "-"
    );

    let t_run = Instant::now();
    for step in 0..nsteps {
        // Fixed point on zeta_mid.
        let mut zeta_mid = zeta.clone();
        let mut last_res = f64::NAN;
        for _ in 0..fp_iters {
            let cfg_half = half_config(&cfg, &zeta_mid, dt);
            let m_bf = to_body_frame(&assemble_m_lab(&mut solver, &cfg_half, &props), &cfg_half);
            let f_cfg = config_force(&mut solver, &cfg_half, &props, &zeta_mid, cfg_eps, project);
            let zeta_mid_new = if use_vi {
                // 2b: mu_mid = Ad*_{n->half}(mu_n) + 0.5 dt f_cfg; zeta_mid = M^-1 mu_mid.
                let mu_mid = transport_exact(&mu, &cfg, &cfg_half) + 0.5 * dt * &f_cfg;
                solve(&m_bf, &mu_mid)
            } else {
                // 2: implicit midpoint on the Kirchhoff ODE.
                let mu_mid = &m_bf * &zeta_mid;
                let rhs = kirchhoff_rhs(&mu_mid, &zeta_mid, &f_cfg);
                let mu_next = &mu + dt * &rhs;
                solve(&m_bf, &(0.5 * (&mu + &mu_next)))
            };
            last_res = (&zeta_mid_new - &zeta_mid).norm();
            zeta_mid = zeta_mid_new;
            if last_res < 1e-12 {
                break;
            }
        }

        // Commit.
        let cfg_half = half_config(&cfg, &zeta_mid, dt);
        let f_cfg = config_force(&mut solver, &cfg_half, &props, &zeta_mid, cfg_eps, project);
        let cfg_new = advance_config(&cfg, &zeta_mid, dt);
        if use_vi {
            // Apply the full impulse once, at the half configuration, then
            // transport n->n+1 via two exact half-legs. This makes the per-step
            // spatial-momentum change exactly dt * Ad*_{g_half^-1} f_cfg, whose
            // net (sum over bodies) is zeroed by the rigid-mode projection, so
            // total momentum is conserved to machine precision. (Splitting the
            // impulse across the two legs instead leaves the second half in the
            // n+1 frame, an O(dt^2)-per-step / O(dt)-accumulated momentum leak.)
            let mu_mid = transport_exact(&mu, &cfg, &cfg_half) + dt * &f_cfg;
            mu = transport_exact(&mu_mid, &cfg_half, &cfg_new);
        } else {
            let m_bf = to_body_frame(&assemble_m_lab(&mut solver, &cfg_half, &props), &cfg_half);
            let mu_mid = &m_bf * &zeta_mid;
            let rhs = kirchhoff_rhs(&mu_mid, &zeta_mid, &f_cfg);
            mu = &mu + dt * &rhs;
        }
        cfg = cfg_new;
        let m_bf_new = to_body_frame(&assemble_m_lab(&mut solver, &cfg, &props), &cfg);
        zeta = solve(&m_bf_new, &mu);

        let e = 0.5 * zeta.dot(&mu);
        let plin = total_linear_momentum_lab(&cfg, &mu);
        let pang = total_angular_momentum_lab(&cfg, &mu);
        if (step + 1) % tprint == 0 {
            println!(
                "{:7.3} {:14.8} {:13.3e} {:11.3e} {:11.3e} {:10.2e}",
                (step + 1) as f64 * dt,
                e,
                (e - e0).abs() / e0.abs().max(1e-30),
                (plin - plin0).norm(),
                (pang - pang0).norm(),
                last_res
            );
        }
    }

    let wall = t_run.elapsed().as_secs_f64();
    let e = 0.5 * zeta.dot(&mu);
    println!();
    println!("wall = {:.2} s   per step = {:.3} s", wall, wall / nsteps as f64);
    println!("final E_drift_rel  = {:.3e}", (e - e0).abs() / e0.abs().max(1e-30));
    println!(
        "final |dPlin|      = {:.3e}",
        (total_linear_momentum_lab(&cfg, &mu) - plin0).norm()
    );
    println!(
        "final |dPang|      = {:.3e}",
        (total_angular_momentum_lab(&cfg, &mu) - pang0).norm()
    );
    println!();
    println!("final positions / orientations:");
    for b in 0..nbody {
        println!(
            "  body {}: pos={:?}  ori={:?}",
            b,
            cfg.pos[b],
            cfg.ori[b].quaternion()
        );
    }
}
