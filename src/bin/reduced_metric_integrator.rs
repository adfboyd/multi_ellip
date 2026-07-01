//! Reduced Kirchhoff-metric integrator (prototype).
//!
//! Advances N rigid ellipsoids in potential flow as the finite-dimensional
//! Lagrangian system  L(q, z) = 0.5 z^T M(q) z,  M(q) = M_body + M_a(q),  where
//! z = (v_1, w_1, ..., v_N, w_N) is the lab-frame generalized velocity and
//! M_a(q) is the added-mass matrix assembled by the unit-velocity method (see
//! `added_mass_matrix_probe` and `docs/reduced_metric_integrator.md`). The fluid
//! has no independent DOF, so this is geodesic motion in the metric M(q).
//!
//! Scheme: implicit-midpoint on the reduced Hamiltonian. Per step, with
//! momentum p = M(q) z:
//!   z_half fixed point:  M(q_half) z_half = p_n + 0.5 dt F(q_half, z_half)
//!   q_{n+1} = q_n (+) dt z_half        (positions += dt v_half; ori via lab exp)
//!   p_{n+1} = p_n + dt F(q_half, z_half)
//!   z_{n+1} = M(q_{n+1})^{-1} p_{n+1}
//! where F_k = d/dq_k [0.5 z^T M(q) z] at fixed z is the metric force, obtained
//! by central differences of the total kinetic energy (solid analytic + fluid
//! from one BEM solve) along each configuration DOF.
//!
//! Correctness note: the lab-frame metric-force torque omits the Euler-Poincare
//! gyroscopic term for anisotropic bodies, so this prototype is *exact only when
//! that term vanishes* -- spheres (isotropic inertia, orientation-independent
//! added mass) or frozen rotation. Use `input_two_sphere.txt` as the clean
//! energy/momentum-conservation validation. Full-ellipsoid rotation needs a
//! proper SE(3) Lie-group variational update (next milestone).
//!
//! Sign convention: `BemSolver::impulse()` returns L = -M_a z, so columns are
//! negated to build the positive-definite M_a.

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

/// Configuration of all bodies: positions and orientations (no velocities).
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

/// Per-body constant properties needed for the solid mass/inertia blocks.
struct BodyProps {
    mass: f64,
    inertia_body: Matrix3<f64>, // body-frame (constant) inertia
}

fn body_state_from(cfg: &Config, z: &DVector<f64>) -> BodyState {
    let n = cfg.nbody();
    let mut pos = DVector::zeros(3 * n);
    let mut vel = DVector::zeros(3 * n);
    let mut q = Vec::with_capacity(n);
    let mut omega = Vec::with_capacity(n);
    for b in 0..n {
        for c in 0..3 {
            pos[3 * b + c] = cfg.pos[b][c];
            vel[3 * b + c] = z[6 * b + c];
        }
        q.push(*cfg.ori[b].quaternion());
        omega.push(Quaternion::from_imag(Vector3::new(
            z[6 * b + 3],
            z[6 * b + 4],
            z[6 * b + 5],
        )));
    }
    BodyState {
        lin: (pos, vel),
        ang: (q, omega),
    }
}

/// Lab-frame solid mass-inertia block contribution to M for body b:
/// linear block = mass * I3, angular block = R I_body R^T.
fn m_body_block(props: &BodyProps, ori: &UnitQuaternion<f64>) -> (Matrix3<f64>, Matrix3<f64>) {
    let r = ori.to_rotation_matrix();
    let lin = Matrix3::identity() * props.mass;
    let ang = r.matrix() * props.inertia_body * r.matrix().transpose();
    (lin, ang)
}

/// Solid kinetic energy at configuration/velocity (z fixed), analytic.
fn solid_ke(cfg: &Config, props: &[BodyProps], z: &DVector<f64>) -> f64 {
    let mut e = 0.0;
    for b in 0..cfg.nbody() {
        let v = Vector3::new(z[6 * b], z[6 * b + 1], z[6 * b + 2]);
        let w = Vector3::new(z[6 * b + 3], z[6 * b + 4], z[6 * b + 5]);
        let (mlin, mang) = m_body_block(&props[b], &cfg.ori[b]);
        e += 0.5 * v.dot(&(mlin * v)) + 0.5 * w.dot(&(mang * w));
    }
    e
}

/// Fluid kinetic energy at (cfg, z): one BEM impulse solve. Equals 0.5 z^T M_a z.
fn fluid_ke(solver: &mut BemSolver, cfg: &Config, z: &DVector<f64>) -> f64 {
    solver.set_state(&body_state_from(cfg, z));
    let _ = solver.impulse();
    solver.fluid_kinetic_energy()
}

/// Assemble the full 6N x 6N metric M(cfg) = M_body + M_a. M_a via unit-velocity
/// impulses (columns negated: impulse returns -M_a z), reusing `solver`.
fn assemble_m(solver: &mut BemSolver, cfg: &Config, props: &[BodyProps]) -> DMatrix<f64> {
    let n = cfg.nbody();
    let dof = 6 * n;
    let mut m = DMatrix::zeros(dof, dof);
    // Added mass columns.
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
    // Symmetrize the added-mass part (discretization gives tiny asymmetry).
    let mt = m.transpose();
    m = 0.5 * (&m + &mt);
    // Solid blocks (block diagonal).
    for b in 0..n {
        let (mlin, mang) = m_body_block(&props[b], &cfg.ori[b]);
        for r in 0..3 {
            for c in 0..3 {
                m[(6 * b + r, 6 * b + c)] += mlin[(r, c)];
                m[(6 * b + 3 + r, 6 * b + 3 + c)] += mang[(r, c)];
            }
        }
    }
    m
}

/// Shift configuration DOF `k` (0..6N) by `delta`: translation -> move position;
/// rotation -> lab-frame left-multiply the body orientation by exp(delta e_c).
fn shift_config(cfg: &Config, k: usize, delta: f64) -> Config {
    let mut out = cfg.clone();
    let b = k / 6;
    let local = k % 6;
    if local < 3 {
        out.pos[b][local] += delta;
    } else {
        let c = local - 3;
        let mut axis = Vector3::zeros();
        axis[c] = delta;
        let dq = UnitQuaternion::from_scaled_axis(axis);
        out.ori[b] = dq * cfg.ori[b];
    }
    out
}

/// Metric force F_k = d/dq_k [0.5 z^T M(q) z] at fixed z, central differences of
/// the total kinetic energy (solid analytic + fluid from one solve) per DOF.
fn metric_force(
    solver: &mut BemSolver,
    cfg: &Config,
    props: &[BodyProps],
    z: &DVector<f64>,
    eps: f64,
) -> DVector<f64> {
    let dof = 6 * cfg.nbody();
    let mut f = DVector::zeros(dof);
    for k in 0..dof {
        let cfg_p = shift_config(cfg, k, eps);
        let cfg_m = shift_config(cfg, k, -eps);
        let e_p = solid_ke(&cfg_p, props, z) + fluid_ke(solver, &cfg_p, z);
        let e_m = solid_ke(&cfg_m, props, z) + fluid_ke(solver, &cfg_m, z);
        f[k] = (e_p - e_m) / (2.0 * eps);
    }
    f
}

fn advance_config(cfg: &Config, z_half: &DVector<f64>, dt: f64) -> Config {
    let mut out = cfg.clone();
    for b in 0..cfg.nbody() {
        let v = Vector3::new(z_half[6 * b], z_half[6 * b + 1], z_half[6 * b + 2]);
        let w = Vector3::new(z_half[6 * b + 3], z_half[6 * b + 4], z_half[6 * b + 5]);
        out.pos[b] = cfg.pos[b] + dt * v;
        let dq = UnitQuaternion::from_scaled_axis(w * dt);
        out.ori[b] = dq * cfg.ori[b];
    }
    out
}

fn half_config(cfg: &Config, z_half: &DVector<f64>, dt: f64) -> Config {
    advance_config(cfg, z_half, 0.5 * dt)
}

fn solve_spd(m: &DMatrix<f64>, rhs: &DVector<f64>) -> DVector<f64> {
    m.clone()
        .lu()
        .solve(rhs)
        .expect("metric solve failed (singular M)")
}

fn total_linear_momentum(p: &DVector<f64>, n: usize) -> Vector3<f64> {
    let mut s = Vector3::zeros();
    for b in 0..n {
        s += Vector3::new(p[6 * b], p[6 * b + 1], p[6 * b + 2]);
    }
    s
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        panic!("usage: reduced_metric_integrator <input.txt> [grad_eps] [fp_iters]");
    }
    let input = &args[1];
    let grad_eps = args
        .get(2)
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(1.0e-4);
    let fp_iters = args
        .get(3)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(6);

    let values = read_values(input);
    let nbody = get(&values, "nbody", 2.0) as usize;
    let ndiv = get(&values, "ndiv", 2.0) as u32;
    let rho_f = get(&values, "rhof", 1.0);
    let dt = get(&values, "dt", 0.05);
    let tend = get(&values, "tend", 1.0);
    let nsteps = (tend / dt).round() as usize;

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
    let mut z = DVector::zeros(6 * nbody);
    for (b, body) in bodies.iter().enumerate() {
        let v = body.linear_velocity();
        let w = body.angular_velocity().imag();
        for c in 0..3 {
            z[6 * b + c] = v[c];
            z[6 * b + 3 + c] = w[c];
        }
    }

    let fluid = Fluid {
        density: rho_f,
        kinetic_energy: 0.0,
    };
    let sys = Simulation::new(fluid, bodies.clone(), ndiv);
    let mut solver = BemSolver::new(sys);

    println!("reduced Kirchhoff-metric integrator (prototype)");
    println!("input={input}  nbody={nbody}  ndiv={ndiv}  dt={dt}  tend={tend}  steps={nsteps}");
    println!("grad_eps={grad_eps:.3e}  fp_iters={fp_iters}");
    println!();

    // Step-start momentum and invariants.
    let mut m_now = assemble_m(&mut solver, &cfg, &props);
    let mut p = &m_now * &z;
    let e0 = 0.5 * z.dot(&p);
    let plin0 = total_linear_momentum(&p, nbody);
    println!(
        "{:>7} {:>14} {:>14} {:>14} {:>10}",
        "t", "E_total", "E_drift_rel", "|Plin-Plin0|", "fp_res"
    );
    println!(
        "{:7.3} {:14.8} {:14.3e} {:14.3e} {:>10}",
        0.0, e0, 0.0, 0.0, "-"
    );

    let t_run = Instant::now();
    for step in 0..nsteps {
        // Fixed point for z_half.
        let mut z_half = z.clone();
        let mut last_res = f64::NAN;
        for _ in 0..fp_iters {
            let cfg_half = half_config(&cfg, &z_half, dt);
            let m_half = assemble_m(&mut solver, &cfg_half, &props);
            let f_half = metric_force(&mut solver, &cfg_half, &props, &z_half, grad_eps);
            let rhs = &p + 0.5 * dt * &f_half;
            let z_half_new = solve_spd(&m_half, &rhs);
            last_res = (&z_half_new - &z_half).norm();
            z_half = z_half_new;
            if last_res < 1e-11 {
                break;
            }
        }

        // Commit: advance configuration by z_half, update momentum, recover z.
        let cfg_half = half_config(&cfg, &z_half, dt);
        let f_half = metric_force(&mut solver, &cfg_half, &props, &z_half, grad_eps);
        cfg = advance_config(&cfg, &z_half, dt);
        p = &p + dt * &f_half;
        m_now = assemble_m(&mut solver, &cfg, &props);
        z = solve_spd(&m_now, &p);

        let e = 0.5 * z.dot(&p);
        let plin = total_linear_momentum(&p, nbody);
        let e_drift = (e - e0).abs() / e0.abs().max(1e-30);
        let dplin = (plin - plin0).norm();
        if (step + 1) % (get(&values, "tprint", 1.0) as usize).max(1) == 0 {
            println!(
                "{:7.3} {:14.8} {:14.3e} {:14.3e} {:10.2e}",
                (step + 1) as f64 * dt,
                e,
                e_drift,
                dplin,
                last_res
            );
        }
    }

    let wall = t_run.elapsed().as_secs_f64();
    println!();
    println!("wall = {:.2} s   per step = {:.3} s", wall, wall / nsteps as f64);
    println!("final E_drift_rel = {:.3e}", (0.5 * z.dot(&p) - e0).abs() / e0.abs().max(1e-30));
    println!("final |Plin-Plin0| = {:.3e}", (total_linear_momentum(&p, nbody) - plin0).norm());
    println!();
    println!("final positions:");
    for b in 0..nbody {
        println!("  body {}: {:?}", b, cfg.pos[b]);
    }
}
