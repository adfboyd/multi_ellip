use chrono::{DateTime, Local};
use nalgebra as na;
use nalgebra::{DVector, Quaternion, Vector3};
use std::{fs, fs::File, io::BufWriter, io::Write, path::Path};

use std::collections::{HashMap, HashSet};
use std::env;
use std::thread;

use nom::{
    bytes::complete::{tag, take_while1},
    character::complete::line_ending,
    combinator::opt,
    multi::many0,
    number::complete::double,
    sequence::{separated_pair, terminated},
    IResult,
};

use multi_ellip::bem::bem_for_ode;
use multi_ellip::ellipsoids::body::Body;
use multi_ellip::ode::rk4pcdm::{self, BodyInfo};
use multi_ellip::system::fluid::Fluid;
use multi_ellip::system::hamiltonian::is_calc;
use multi_ellip::system::system::Simulation;
use multi_ellip::utils::SimName;

fn main() {
    let started_at: DateTime<Local> = Local::now();
    let args: Vec<String> = env::args().collect();

    // Expect at least one argument: the input file path.
    if args.len() < 2 {
        panic!("Usage: program <path_to_input_file> [output_dir]");
    }
    let input_file_path = &args[1];
    let blank_fp = ".".to_string();
    let output_file_path = if args.len() > 2 { &args[2] } else { &blank_fp };

    let input_data = match fs::read_to_string(input_file_path) {
        Ok(content) => content,
        Err(e) => {
            println!("Failed to read file '{}': {:?}", input_file_path, e);
            return;
        }
    };

    // Keep only "key=value" lines before parsing: the nom many0 parser stops at
    // the first line it can't parse (blank line, comment, trailing whitespace),
    // which would silently drop every key after it. Filtering first makes key
    // order and blank lines irrelevant.
    let input_data: String = input_data
        .lines()
        .filter(|l| l.contains('='))
        .collect::<Vec<_>>()
        .join("\n");

    // Parse "key=value" assignments into a map.
    let mut values: HashMap<String, f64> = HashMap::new();
    let mut seen_keys: HashSet<String> = HashSet::new();
    let mut duplicate_keys: Vec<String> = Vec::new();
    match parse_assignments(&input_data) {
        Ok((_, assignments)) => {
            for (variable, value) in assignments {
                if !seen_keys.insert(variable.to_string()) {
                    duplicate_keys.push(variable.to_string());
                }
                values.insert(variable.to_string(), value);
            }
        }
        Err(e) => {
            println!("Failed to parse: {:?}", e);
            return;
        }
    }

    let get = |name: &str, default: f64| -> f64 { *values.get(name).unwrap_or(&default) };

    let rho_f = get("rhof", 1.0);
    let ndiv = get("ndiv", 2.0) as usize;
    let t_end = get("tend", 10.0);
    let dt = get("dt", 0.1);
    let tprint_raw = get("tprint", 1.0);
    // Console progress interval (steps). Decoupled from tprint, which controls
    // how often rows are written to the output .dat file.
    let logevery_raw = get("logevery", 100.0);
    let nbody = get("nbody", 2.0) as usize;
    let mut input_warnings: Vec<String> = Vec::new();

    if dt <= 0.0 {
        println!("Invalid input: dt must be greater than 0.");
        return;
    }
    if t_end <= 0.0 {
        println!("Invalid input: tend must be greater than 0.");
        return;
    }
    if nbody == 0 {
        println!("Invalid input: nbody must be at least 1.");
        return;
    }
    if tprint_raw < 1.0 {
        input_warnings.push(format!(
            "tprint={} is below 1; using 1 step instead.",
            tprint_raw
        ));
    }
    if logevery_raw < 1.0 {
        input_warnings.push(format!(
            "logevery={} is below 1; using 1 step instead.",
            logevery_raw
        ));
    }
    if !duplicate_keys.is_empty() {
        duplicate_keys.sort();
        duplicate_keys.dedup();
        input_warnings.push(format!(
            "duplicate input key(s): {}. The last value for each key was used.",
            duplicate_keys.join(", ")
        ));
    }

    let tprint = tprint_raw.max(1.0) as u32;
    let logevery = logevery_raw.max(1.0) as u32;

    // Build each body from its 1-indexed input variables (cex1, oriw1, shx1, ...).
    let read_body = |i: usize| -> Body {
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

        // Normalise the input shape so its equivalent radius equals `req`.
        let (shx, shy, shz) = (g("shx", 1.0), g("shy", 1.0), g("shz", 1.0));
        let req = g("req", 1.0);
        let req_temp = (shx * shy * shz).powf(1.0 / 3.0);
        let sf = req / req_temp;
        let shape = Vector3::new(shx * sf, shy * sf, shz * sf);
        let rho_s = g("rhos", 1.0);

        let mut body = Body {
            density: rho_s,
            shape,
            position,
            orientation,
            linear_momentum: lin_velocity,
            angular_momentum: ang_velocity,
            inertia: is_calc(na::Matrix3::from_diagonal(&shape), rho_s),
        };
        body.set_linear_velocity(lin_velocity);
        body.set_angular_velocity(ang_velocity);
        body
    };

    let bodies: Vec<Body> = (1..=nbody).map(read_body).collect();

    let fluid = Fluid {
        density: rho_f,
        kinetic_energy: 0.0,
    };

    let mut sys = Simulation::new(fluid, bodies, ndiv as u32);
    sys.step_dt = dt;
    // Lamb impulse-transport term (ω × L). Required for the correct force/torque
    // on a rotating body: F = dL/dt = ρ∮(∂φ/∂t)n̂dA + ω×L. Default ON; set
    // `impulse_transport=0` to drop it (reproduces the old, incomplete force).
    sys.impulse_transport = get("impulse_transport", 1.0) > 0.5;
    sys.added_mass_stab = get("added_mass_stab", 0.0) > 0.5;
    sys.phidot_blend = get("phidot_blend", 0.0);
    let strong_couple = get("strong_couple", 0.0) > 0.5;
    let impulse_scheme = get("impulse_scheme", 0.0) > 0.5;
    let energy_projection = get("energy_projection", 0.0) > 0.5;
    let phidot_blend = sys.phidot_blend;

    // Initial integrator state, stacked over bodies.
    let mut p0 = DVector::zeros(3 * nbody);
    let mut v0 = DVector::zeros(3 * nbody);
    let mut orientations: Vec<(Quaternion<f64>, Quaternion<f64>)> = Vec::with_capacity(nbody);
    let mut inertias: Vec<na::Matrix3<f64>> = Vec::with_capacity(nbody);
    let mut masses: Vec<f64> = Vec::with_capacity(nbody);
    let mut body_info: Vec<BodyInfo> = Vec::with_capacity(nbody);

    for (i, b) in sys.bodies.iter().enumerate() {
        let pos = b.position;
        let vel = b.linear_velocity();
        let lin_ke = b.linear_energy();
        let rot_ke = b.rotational_energy();
        for c in 0..3 {
            p0[3 * i + c] = pos[c];
            v0[3 * i + c] = vel[c];
        }
        orientations.push((b.orientation, b.angular_velocity()));
        inertias.push(b.inertia);
        masses.push(b.mass());
        body_info.push(BodyInfo {
            density: b.density,
            shape: b.shape,
            initial_ke_ratio: if lin_ke.abs() > f64::EPSILON {
                rot_ke / lin_ke
            } else {
                f64::NAN
            },
        });
    }

    let x = (p0, v0);

    // Per-body body-frame added-mass tensors for the optional stabiliser,
    // pre-scaled by the safety factor (input key added_mass_safety, default 2).
    // Constant in the body frame; harmless when the flag is off.
    let added_mass_safety = get("added_mass_safety", 2.0);
    let added_mass_tensors: Vec<na::Matrix3<f64>> = sys
        .added_mass_tensors()
        .iter()
        .map(|m| m * added_mass_safety)
        .collect();
    let added_mass_stab = sys.added_mass_stab;

    let solver = bem_for_ode::BemSolver::new(sys);

    // Impulse takes precedence over strong, then the explicit default (matches the
    // former boolean precedence).
    let scheme = if impulse_scheme {
        rk4pcdm::CouplingScheme::Impulse
    } else if strong_couple {
        rk4pcdm::CouplingScheme::Strong
    } else {
        rk4pcdm::CouplingScheme::Explicit
    };

    let mut stepper = rk4pcdm::Rk4PCDM::new(
        solver,
        0.0,
        x,
        orientations,
        inertias,
        masses,
        body_info,
        added_mass_tensors,
        added_mass_stab,
        t_end,
        dt,
        tprint,   // samp_rate: .dat row every tprint steps
        logevery, // print_rate: console progress every logevery steps
        scheme,
        energy_projection,
    );

    // Per body: an octahedron (8 faces) subdivided 4^ndiv times -> 8*4^ndiv
    // quadratic (6-node) triangles, with 2*nelm + 2 nodes (matches ellip_gridder).
    let elems_per_body = 8_usize * 4_usize.pow(ndiv as u32);
    let nelm_end = elems_per_body * nbody;
    let npts_total = (2 * elems_per_body + 2) * nbody;

    let path_base_str = output_file_path;

    if std::fs::create_dir_all(path_base_str.clone()).is_err() {
        panic!("Could not create output directories\n");
    }

    let path_base = Path::new(&path_base_str);
    let sim_name = SimName::new(path_base);

    // Single body -> single_body_complete.dat; otherwise multiple_body_complete.dat.
    let path = if nbody == 1 {
        sim_name.single_body_path()
    } else {
        sim_name.complete_path()
    };
    let file1 = match File::create(path) {
        Err(e) => {
            println!("Could not open file. Error: {:?}", e);
            return;
        }
        Ok(buf) => buf,
    };
    let output_path = path.to_path_buf();

    let mut buf = BufWriter::new(file1);

    let estimated_steps = ((t_end - 0.0) / dt).ceil() as usize;
    let output_rows = estimated_output_rows(estimated_steps, tprint as usize);
    let available_cores = available_parallelism();
    let rayon_threads = rayon::current_num_threads();
    println!();
    if !input_warnings.is_empty() {
        println!("Input warnings:");
        for warning in &input_warnings {
            println!("  - {}", warning);
        }
        println!();
    }
    println!("================ Simulation setup ================");
    println!("  Run label:         {}", path_base.display());
    println!("  Started:           {}", fmt_timestamp(started_at));
    println!("  Input file:        {}", input_file_path);
    println!("  Output file:       {}", output_path.display());
    println!("  Bodies:            {}", nbody);
    println!("  Fluid density:     {}", rho_f);
    println!(
        "  Mesh:              ndiv {} -> {} triangles / {} nodes",
        ndiv, nelm_end, npts_total
    );
    println!(
        "  Simulated time:    0 -> {}  (dt = {}, ~{} steps)",
        t_end, dt, estimated_steps
    );
    println!("  Output cadence:    every {} step(s)", tprint);
    println!("  Output rows:       ~{}", output_rows);
    println!("  Progress cadence:  every {} step(s)", logevery);
    println!(
        "  CPU cores:         {} available, {} Rayon worker thread(s)",
        available_cores, rayon_threads
    );
    println!("  Strong coupling:   {}", fmt_enabled(strong_couple));
    println!("  Impulse scheme:    {}", fmt_enabled(impulse_scheme));
    println!("  Energy projection: {}", fmt_enabled(energy_projection));
    if added_mass_stab {
        println!(
            "  Added-mass stab:   enabled  (safety factor = {})",
            added_mass_safety
        );
    } else {
        println!("  Added-mass stab:   disabled");
    }
    println!("  Phi-dot blend:     {}", fmt_phidot_blend(phidot_blend));
    print_solver_guidance();
    println!("================================================");
    println!("Solver starting - good luck!");
    println!();

    let res = stepper.integrate_with_writer(&mut buf);

    match res {
        Ok(stats) => {
            let finished_at: DateTime<Local> = Local::now();
            println!("Solver finished successfully - good job!");
            if let Err(e) = buf.flush() {
                println!("Could not write to file. Error: {:?}", e);
                return;
            }
            println!("Results saved successfully.");

            println!();
            println!("================== Run summary ==================");
            println!("  Bodies:            {}", nbody);
            println!("  Triangles / nodes: {} / {}", nelm_end, npts_total);
            println!(
                "  Timesteps:         {}  (dt = {})",
                stats.accepted_steps, dt
            );
            println!("  Simulated time:    0 -> {}", t_end);
            println!("  First-step setup:  {:.3} s", stepper.run_first_step_secs);
            println!(
                "  Mean time/step:    {:.4} s  (excl. first step)",
                stepper.run_steady_per_step
            );
            println!("  Total wall time:   {}", fmt_hms(stepper.run_wall_secs));
            if energy_projection {
                println!(
                    "  Projection max |dz|/|z|:        {:.6e}",
                    stepper.projection_max_corr_rel
                );
                println!(
                    "  Projection max pre-KE error:    {:.6e}",
                    stepper.projection_max_energy_err_rel
                );
                println!(
                    "  Projection max impulse residual: {:.6e}",
                    stepper.projection_max_constraint_resid
                );
            }
            println!(
                "  CPU cores:         {} available, {} Rayon worker thread(s)",
                available_cores, rayon_threads
            );
            println!("  Started:           {}", fmt_timestamp(started_at));
            println!("  Finished:          {}", fmt_timestamp(finished_at));
            println!("  Output file:       {}", output_path.display());
            println!("================================================");
        }
        Err(e) => {
            if let Err(flush_err) = buf.flush() {
                println!(
                    "Could not flush partial output file. Error: {:?}",
                    flush_err
                );
            }
            println!("Solver stopped with error: {e}");
        }
    };
}

fn estimated_output_rows(num_steps: usize, samp_rate: usize) -> usize {
    if num_steps == 0 {
        1
    } else {
        2 + (num_steps - 1) / samp_rate
    }
}

fn fmt_timestamp(timestamp: DateTime<Local>) -> String {
    timestamp.format("%Y-%m-%d %H:%M:%S").to_string()
}

fn fmt_phidot_blend(value: f64) -> String {
    if value.abs() <= f64::EPSILON {
        "0 (off)".to_string()
    } else {
        value.to_string()
    }
}

fn print_solver_guidance() {
    println!();
    println!("  Solver guidance:");
    println!("    - Strong coupling: use for stiff close-coupled runs; slower but more implicit.");
    println!(
        "    - Impulse scheme: use when impulse/momentum conservation or KE drift is the focus. Recommended."
    );
    println!(
        "    - Added-mass stab: use for higher mesh counts (ndiv > 3) to preserve stability - deprecated."
    );
}

fn available_parallelism() -> usize {
    thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

fn fmt_enabled(enabled: bool) -> &'static str {
    if enabled {
        "enabled"
    } else {
        "disabled"
    }
}

/// Format a duration in seconds as a compact "Hh Mm Ss" string.
fn fmt_hms(secs: f64) -> String {
    let total = secs as u64;
    let (h, m, s) = (total / 3600, (total % 3600) / 60, total % 60);
    if h > 0 {
        format!("{}h {}m {}s", h, m, s)
    } else if m > 0 {
        format!("{}m {}s", m, s)
    } else {
        format!("{:.2}s", secs)
    }
}

fn parse_assignment(input: &str) -> IResult<&str, (&str, f64)> {
    // Keys are alphanumeric plus underscore (e.g. impulse_transport); plain
    // alphanumeric1 silently rejected underscore keys and, via many0, dropped
    // every key after the first such line.
    let key = take_while1(|c: char| c.is_alphanumeric() || c == '_');
    separated_pair(key, tag("="), double)(input)
}

fn parse_assignments(input: &str) -> IResult<&str, Vec<(&str, f64)>> {
    many0(terminated(parse_assignment, opt(line_ending)))(input)
}
