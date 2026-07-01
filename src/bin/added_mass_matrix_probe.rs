use multi_ellip::bem::bem_for_ode::{BemSolver, BodyState};
use multi_ellip::ellipsoids::body::Body;
use multi_ellip::system::fluid::Fluid;
use multi_ellip::system::hamiltonian::is_calc;
use multi_ellip::system::system::Simulation;
use nalgebra::{DMatrix, DVector, Matrix3, Quaternion, Vector3};
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

fn read_body(values: &HashMap<String, f64>, i: usize, sep_delta: f64) -> Body {
    let g = |name: &str, default: f64| -> f64 {
        *values.get(&format!("{}{}", name, i)).unwrap_or(&default)
    };

    let mut position = Vector3::new(g("cex", 0.0), g("cey", 0.0), g("cez", 0.0));
    if i == 1 {
        position[0] -= 0.5 * sep_delta;
    } else if i == 2 {
        position[0] += 0.5 * sep_delta;
    }

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

fn body_state(bodies: &[Body]) -> BodyState {
    let nbody = bodies.len();
    let mut pos = DVector::zeros(3 * nbody);
    let mut vel = DVector::zeros(3 * nbody);
    let mut q = Vec::with_capacity(nbody);
    let mut omega = Vec::with_capacity(nbody);
    for (i, body) in bodies.iter().enumerate() {
        for c in 0..3 {
            pos[3 * i + c] = body.position[c];
            vel[3 * i + c] = body.linear_velocity()[c];
        }
        q.push(body.orientation);
        omega.push(Quaternion::from_imag(body.angular_velocity().imag()));
    }
    BodyState {
        lin: (pos, vel),
        ang: (q, omega),
    }
}

fn solid_ke(bodies: &[Body]) -> f64 {
    bodies
        .iter()
        .map(|body| {
            let v = body.linear_velocity();
            let omega = body.angular_velocity().imag();
            let lin = 0.5 * body.mass() * v.dot(&v);
            let rot = 0.5 * omega.dot(&(body.inertia * omega));
            lin + rot
        })
        .sum()
}

/// Pack the true generalized velocity of `bodies` into the 6N ordering used
/// throughout this probe: index `6b+c` (c=0,1,2) is body b's linear velocity
/// component c, and `6b+3+c` is body b's angular velocity component c.
fn true_generalized_velocity(bodies: &[Body]) -> DVector<f64> {
    let nbody = bodies.len();
    let mut z = DVector::zeros(6 * nbody);
    for (b, body) in bodies.iter().enumerate() {
        let v = body.linear_velocity();
        let omega = body.angular_velocity().imag();
        for c in 0..3 {
            z[6 * b + c] = v[c];
            z[6 * b + 3 + c] = omega[c];
        }
    }
    z
}

/// Build a `BodyState` with the true positions/orientations from `bodies` but
/// with only a single generalized-velocity DOF `j` (0-indexed, 0..6N) set to
/// `value` and every other velocity component zeroed.
fn unit_velocity_state(bodies: &[Body], j: usize, value: f64) -> BodyState {
    let mut state = body_state(bodies);
    // Zero out all velocities first.
    state.lin.1.fill(0.0);
    for om in state.ang.1.iter_mut() {
        *om = Quaternion::from_imag(Vector3::zeros());
    }
    let b = j / 6;
    let local = j % 6;
    if local < 3 {
        state.lin.1[3 * b + local] = value;
    } else {
        let c = local - 3;
        let mut imag = state.ang.1[b].imag();
        imag[c] = value;
        state.ang.1[b] = Quaternion::from_imag(imag);
    }
    state
}

/// Assemble the full 6N x 6N added-mass matrix M_a(q) at the configuration
/// described by `values`/`sep_delta`. Because potential-flow impulse is exact
/// and linear in the generalized velocity, column j of M_a is simply the
/// impulse produced by setting generalized velocity = e_j (unit velocity in
/// DOF j, all else zero) at the fixed configuration. The same `BemSolver` (and
/// hence its cached factorisation) is reused across all 6N unit-velocity
/// solves since none of them change the geometry.
///
/// Sign convention: `BemSolver::impulse()` returns the Lamb impulse
/// L = rho*int(phi*n dA), which is the *negative* of the positive-definite
/// added mass acting on the velocity (see the note in bem_for_ode.rs: the body
/// conserves m_s*u - L, i.e. effective mass m_s + M_a). We therefore assemble
/// M_a = -dL/dz so that M_a is symmetric positive-definite and the reduced
/// fluid energy is +0.5 z^T M_a z. Any integrator built on this M_a must keep
/// this negation consistent with whatever it consumes from impulse().
fn assemble_added_mass(
    values: &HashMap<String, f64>,
    sep_delta: f64,
) -> (DMatrix<f64>, DVector<f64>, f64, f64) {
    let nbody = get(values, "nbody", 2.0) as usize;
    let ndiv = get(values, "ndiv", 2.0) as u32;
    let rho_f = get(values, "rhof", 1.0);
    let bodies: Vec<Body> = (1..=nbody)
        .map(|i| read_body(values, i, sep_delta))
        .collect();
    let fluid = Fluid {
        density: rho_f,
        kinetic_energy: 0.0,
    };
    let sys = Simulation::new(fluid, bodies.clone(), ndiv);
    let mut solver = BemSolver::new(sys);

    // True state: record true impulse/fluid KE/solid KE/generalized velocity.
    let true_state = body_state(&bodies);
    solver.set_state(&true_state);
    let _ = solver.impulse();
    let fluid_ke_true = solver.fluid_kinetic_energy();
    let ke_solid = solid_ke(&bodies);
    let z_true = true_generalized_velocity(&bodies);

    // Assemble M column by column via unit-velocity impulses, reusing solver.
    let dof = 6 * nbody;
    let mut m = DMatrix::zeros(dof, dof);
    for j in 0..dof {
        let state = unit_velocity_state(&bodies, j, 1.0);
        solver.set_state(&state);
        let (l_lin, l_ang) = solver.impulse();
        // Negate: M_a = -dL/dz (positive-definite convention, see doc comment).
        for b in 0..nbody {
            for c in 0..3 {
                m[(6 * b + c, j)] = -l_lin[3 * b + c];
                m[(6 * b + 3 + c, j)] = -l_ang[3 * b + c];
            }
        }
    }

    (m, z_true, fluid_ke_true, ke_solid)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        panic!("usage: added_mass_matrix_probe <input.txt> [eps] [--zero-spin] [--zero-linear]");
    }
    let eps = args
        .get(2)
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(1.0e-3);
    let mut values = read_values(&args[1]);
    let zero_spin = args.iter().any(|s| s == "--zero-spin");
    let zero_linear = args.iter().any(|s| s == "--zero-linear");
    let nbody = get(&values, "nbody", 2.0) as usize;
    if zero_spin {
        for i in 1..=nbody {
            values.insert(format!("avx{}", i), 0.0);
            values.insert(format!("avy{}", i), 0.0);
            values.insert(format!("avz{}", i), 0.0);
        }
    }
    if zero_linear {
        for i in 1..=nbody {
            values.insert(format!("lvx{}", i), 0.0);
            values.insert(format!("lvy{}", i), 0.0);
            values.insert(format!("lvz{}", i), 0.0);
        }
    }
    let ndiv = get(&values, "ndiv", 2.0) as u32;
    let sep0 = get(&values, "cex2", 0.0) - get(&values, "cex1", 0.0);
    let dof = 6 * nbody;

    println!("added-mass full matrix assembly probe");
    println!("(M_a = -dL/dz, symmetric positive-definite; fluid KE = 0.5 z^T M_a z)");
    println!("input={}", args[1]);
    println!("nbody={}", nbody);
    println!("ndiv={}", ndiv);
    println!("zero_spin={}", zero_spin);
    println!("zero_linear={}", zero_linear);
    println!("eps={:.12e}", eps);
    println!("sep (cex2-cex1)={:.12e}", sep0);
    println!("dof (6N)={}", dof);

    // ---- Base assembly, timed for the cost demo ----
    let t_assemble_start = Instant::now();
    let (m0, z_true, fluid_ke_true, ke_solid) = assemble_added_mass(&values, 0.0);
    let assemble_elapsed = t_assemble_start.elapsed();

    // ---- +/- eps assemblies for the metric gradient ----
    let (m_minus, _z_m, fluid_ke_minus, _ks_m) = assemble_added_mass(&values, -eps);
    let (m_plus, _z_p, fluid_ke_plus, _ks_p) = assemble_added_mass(&values, eps);

    // ---- Symmetry error of base M ----
    let mut max_abs_asym = 0.0f64;
    let mut max_abs_entry = 0.0f64;
    for r in 0..dof {
        for c in 0..dof {
            let val = m0[(r, c)];
            if val.abs() > max_abs_entry {
                max_abs_entry = val.abs();
            }
            let asym = (m0[(r, c)] - m0[(c, r)]).abs();
            if asym > max_abs_asym {
                max_abs_asym = asym;
            }
        }
    }
    let rel_asym = if max_abs_entry > 0.0 {
        max_abs_asym / max_abs_entry
    } else {
        0.0
    };

    // ---- Zero-velocity sanity check ----
    let nbody_f = get(&values, "nbody", 2.0) as usize;
    let bodies_zero: Vec<Body> = (1..=nbody_f)
        .map(|i| read_body(&values, i, 0.0))
        .collect();
    let fluid_zero = Fluid {
        density: get(&values, "rhof", 1.0),
        kinetic_energy: 0.0,
    };
    let sys_zero = Simulation::new(fluid_zero, bodies_zero.clone(), ndiv);
    let mut solver_zero = BemSolver::new(sys_zero);
    let mut zero_state = body_state(&bodies_zero);
    zero_state.lin.1.fill(0.0);
    for om in zero_state.ang.1.iter_mut() {
        *om = Quaternion::from_imag(Vector3::zeros());
    }
    let t_single_start = Instant::now();
    solver_zero.set_state(&zero_state);
    let (ll_zero, la_zero) = solver_zero.impulse();
    let single_impulse_elapsed = t_single_start.elapsed();
    let mut norm_l_zero_sq = 0.0f64;
    for v in ll_zero.iter() {
        norm_l_zero_sq += v * v;
    }
    for v in la_zero.iter() {
        norm_l_zero_sq += v * v;
    }
    let norm_l_zero = norm_l_zero_sq.sqrt();

    // ---- Energy consistency: 0.5 z^T M z vs fluid_ke_true ----
    let mz = &m0 * &z_true;
    let mut energy_from_m = 0.0f64;
    for i in 0..dof {
        energy_from_m += z_true[i] * mz[i];
    }
    energy_from_m *= 0.5;
    let rel_energy_err = if fluid_ke_true.abs() > 0.0 {
        (energy_from_m - fluid_ke_true).abs() / fluid_ke_true.abs()
    } else {
        (energy_from_m - fluid_ke_true).abs()
    };

    // ---- Metric gradient: 0.5 z^T dM_ds z vs d_fluid_ke_ds (finite diff) ----
    let dm_ds = (&m_plus - &m_minus) / (2.0 * eps);
    let dmz = &dm_ds * &z_true;
    let mut half_zt_dm_z = 0.0f64;
    for i in 0..dof {
        half_zt_dm_z += z_true[i] * dmz[i];
    }
    half_zt_dm_z *= 0.5;
    let d_fluid_ke_ds = (fluid_ke_plus - fluid_ke_minus) / (2.0 * eps);
    let rel_grad_err = if d_fluid_ke_ds.abs() > 0.0 {
        (half_zt_dm_z - d_fluid_ke_ds).abs() / d_fluid_ke_ds.abs()
    } else {
        (half_zt_dm_z - d_fluid_ke_ds).abs()
    };

    // ---- Report ----
    println!();
    println!("-- symmetry check (base M) --");
    println!("max|M - M^T|        = {:.12e}", max_abs_asym);
    println!("max|M| entry        = {:.12e}", max_abs_entry);
    println!("relative asymmetry  = {:.12e}", rel_asym);

    println!();
    println!("-- zero-velocity sanity check --");
    println!("||L|| at z=0        = {:.12e}", norm_l_zero);

    println!();
    println!("-- energy consistency --");
    println!("0.5 z^T M z         = {:.12e}", energy_from_m);
    println!("fluid_ke_true       = {:.12e}", fluid_ke_true);
    println!("solid_ke_true       = {:.12e}", ke_solid);
    println!("relative error      = {:.12e}", rel_energy_err);

    println!();
    println!("-- metric gradient consistency --");
    println!("fluid_ke_minus      = {:.12e}", fluid_ke_minus);
    println!("fluid_ke_plus       = {:.12e}", fluid_ke_plus);
    println!("0.5 z^T dM_ds z     = {:.12e}", half_zt_dm_z);
    println!("d_fluid_ke_ds (FD)  = {:.12e}", d_fluid_ke_ds);
    println!("relative error      = {:.12e}", rel_grad_err);

    println!();
    println!("-- cost demo --");
    println!(
        "single impulse() call        = {:?}",
        single_impulse_elapsed
    );
    println!(
        "full 6N-column base assembly = {:?}",
        assemble_elapsed
    );
    let ratio = assemble_elapsed.as_secs_f64() / single_impulse_elapsed.as_secs_f64().max(1e-12);
    println!(
        "ratio (assembly / single)    = {:.3}  (expected << 6N = {})",
        ratio, dof
    );

    println!();
    println!("-- base added-mass matrix M_a = -dL/dz (6N x 6N) --");
    println!("{:.6}", m0);
}
