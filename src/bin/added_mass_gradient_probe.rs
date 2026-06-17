use multi_ellip::bem::bem_for_ode::{BemSolver, BodyState};
use multi_ellip::ellipsoids::body::Body;
use multi_ellip::system::fluid::Fluid;
use multi_ellip::system::hamiltonian::is_calc;
use multi_ellip::system::system::Simulation;
use nalgebra::{DVector, Matrix3, Quaternion, Vector3};
use std::collections::HashMap;
use std::env;
use std::fs;

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

fn evaluate(
    values: &HashMap<String, f64>,
    sep_delta: f64,
) -> (f64, f64, DVector<f64>, DVector<f64>) {
    let nbody = get(values, "nbody", 2.0) as usize;
    let ndiv = get(values, "ndiv", 2.0) as u32;
    let rho_f = get(values, "rhof", 1.0);
    let bodies: Vec<Body> = (1..=nbody)
        .map(|i| read_body(values, i, sep_delta))
        .collect();
    let state = body_state(&bodies);
    let fluid = Fluid {
        density: rho_f,
        kinetic_energy: 0.0,
    };
    let sys = Simulation::new(fluid, bodies.clone(), ndiv);
    let mut solver = BemSolver::new(sys);
    solver.set_state(&state);
    let (l_lin, l_ang) = solver.impulse();
    (
        solid_ke(&bodies),
        solver.fluid_kinetic_energy(),
        l_lin,
        l_ang,
    )
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        panic!("usage: added_mass_gradient_probe <input.txt> [eps] [--zero-spin] [--zero-linear]");
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
    let sep0 = get(&values, "cex2", 0.0) - get(&values, "cex1", 0.0);

    let (ks_m, kf_m, ll_m, la_m) = evaluate(&values, -eps);
    let (ks_0, kf_0, ll_0, la_0) = evaluate(&values, 0.0);
    let (ks_p, kf_p, ll_p, la_p) = evaluate(&values, eps);

    let d_kf_ds = (kf_p - kf_m) / (2.0 * eps);
    let d_kt_ds = ((ks_p + kf_p) - (ks_m + kf_m)) / (2.0 * eps);
    let d_ll_ds = (&ll_p - &ll_m) / (2.0 * eps);
    let d_la_ds = (&la_p - &la_m) / (2.0 * eps);

    println!("added-mass local separation-gradient probe");
    println!("input={}", args[1]);
    println!("ndiv={}", get(&values, "ndiv", 2.0));
    println!("zero_spin={}", zero_spin);
    println!("zero_linear={}", zero_linear);
    println!("sep={:.12e}", sep0);
    println!("eps={:.12e}", eps);
    println!("solid_ke_minus={:.12e}", ks_m);
    println!("solid_ke_base ={:.12e}", ks_0);
    println!("solid_ke_plus ={:.12e}", ks_p);
    println!("fluid_ke_minus={:.12e}", kf_m);
    println!("fluid_ke_base ={:.12e}", kf_0);
    println!("fluid_ke_plus ={:.12e}", kf_p);
    println!("d_fluid_ke_ds={:.12e}", d_kf_ds);
    println!("d_total_ke_ds={:.12e}", d_kt_ds);
    println!("l_lin_base={}", ll_0.transpose());
    println!("l_ang_base={}", la_0.transpose());
    println!("d_l_lin_ds={}", d_ll_ds.transpose());
    println!("d_l_ang_ds={}", d_la_ds.transpose());
}
