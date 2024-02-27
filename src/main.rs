use std::{fs, fs::File, io::BufWriter, io::Write, path::Path};
use std::f64::consts::PI;
// use std::rc::Rc;
use std::sync::{Arc, Mutex};
use nalgebra as na;
use nalgebra::{Quaternion, Vector3, Vector6};

use std::env;
use std::collections::HashMap;

use nom::{
    IResult,
    bytes::complete::tag,
    character::complete::{alphanumeric1, line_ending, multispace0},
    number::complete::double,
    sequence::{preceded, separated_pair, terminated},
    combinator::{map, opt},
    multi::many0,

};

use multi_ellip::ellipsoids::body::Body;
use multi_ellip::system::hamiltonian::is_calc;
use multi_ellip::bem::bem_for_ode;
// use multi_ellip::bem::potentials::*;
use std::time::Instant;
// use indicatif::ParallelProgressIterator;
// use plotters::prelude::LogScalable;
// use rayon::prelude::IntoParallelRefIterator;
// use rayon::iter::*;
// use multi_ellip::bem::geom::{abc_vec, combiner, ellip_gridder, elm_geom, gauss_leg, gauss_trgl};
// use multi_ellip::bem::integ::*;
use multi_ellip::ode::rk4pcdm;
use multi_ellip::system::fluid::Fluid;
use multi_ellip::system::system::Simulation;
use multi_ellip::utils::SimName;

// use plotters::prelude::*;
// use colorous::*;
// use palette::{Srgb, Hsv};
// use palette::rgb::Rgb;
// use palette::FromColor;


type State2 = (Quaternion<f64>, Quaternion<f64>);


type Linear2State = (Vector6<f64>, Vector6<f64>);
type Angular2State = (State2, State2);

type Time = f64;

// fn map_value_to_color(value: f64) -> RGBColor {
//     // Map the value to a hue in the range [0, 240]
//     let hue = ((value + 1.0) / 2.0 * 240.0).min(240.0) as u8;
//
//     // Create an RGBColor with the hue as the red component
//     RGBColor(hue, 0, 0)
// }

// fn main() -> Result<(), Box<dyn std::error::Error>> {
fn main() {
    println!("Hello, world!");


    let args: Vec<String> = env::args().collect();

    // Expect at least one argument: the file path
    if args.len() < 1 {
        println!("Usage: program <path_to_input_file>");
        return;
    }
    let input_file_path = &args[1];
    println!("{:?}", input_file_path);
    let blank_fp = ".".to_string();
    let output_file_path = if args.len() > 2
             {&args[2]}
        else
             {&blank_fp};
    // Read from the file
    let input_data = match fs::read_to_string(input_file_path) {
        Ok(content) => content,
        Err(e) => {
            println!("Failed to read file '{}': {:?}", input_file_path, e);
            return;
        }
    };

    println!("{:?}",input_data);
    let mut position_1 = Vector3::new(0.0, 0.0, 0.0);
    let mut orientation_1 = Quaternion::from_parts(0.0, Vector3::new(0.0, 0.0, 0.0));
    let mut lin_velocity_1 = Vector3::new(0.0, 0.0, 0.0);
    let mut ang_velocity_1 = Quaternion::from_parts(0.0, Vector3::new(0.0, 0.0, 0.0));
    let mut shape_1 = Vector3::new(0.0, 0.0, 0.0);
    let mut req1 = 1.0;
    let mut rho_s1 = 1.0;

    let mut position_2 = Vector3::new(0.0, 0.0, 0.0);
    let mut orientation_2 = Quaternion::from_parts(0.0, Vector3::new(0.0, 0.0, 0.0));
    let mut lin_velocity_2 = Vector3::new(0.0, 0.0, 0.0);
    let mut ang_velocity_2 = Quaternion::from_parts(0.0, Vector3::new(0.0, 0.0, 0.0));
    let mut shape_2 = Vector3::new(0.0, 0.0, 0.0);
    let mut req2 = 1.0;
    let mut rho_s2 = 1.0;

    let mut rho_f = 1.0;
    let mut t_end = 10.0;
    let mut dt = 0.1;
    let mut tprint = 20;
    let mut ndiv :usize = 2;
    let mut nbody :usize = 2;

    match parse_assignments(&input_data) {
        Ok((_, assignments)) => {
            let mut values = HashMap::new();
            for (variable, value) in assignments {
                values.insert(variable.to_string(), value);
            }

            let mut check_and_assign_default = |name: &str| {
                match name {
                    // Specify special default values for specific variables
                    "shx1" => *values.entry(name.to_string()).or_insert(1.0),
                    "shy1" => *values.entry(name.to_string()).or_insert(1.0),
                    "shz1" => *values.entry(name.to_string()).or_insert(1.0),
                    "shx2" => *values.entry(name.to_string()).or_insert(1.0),
                    "shy2" => *values.entry(name.to_string()).or_insert(1.0),
                    "shz2" => *values.entry(name.to_string()).or_insert(1.0),
                    "oriw1" => *values.entry(name.to_string()).or_insert(1.0),
                    "oriw2" => *values.entry(name.to_string()).or_insert(1.0),
                    "rhos1" => *values.entry(name.to_string()).or_insert(1.0),
                    "rhos2" => *values.entry(name.to_string()).or_insert(1.0),
                    "ndiv" => *values.entry(name.to_string()).or_insert(2.0),
                    "tend" => *values.entry(name.to_string()).or_insert(10.0),
                    "dt" => *values.entry(name.to_string()).or_insert(0.1),
                    "nbody" => *values.entry(name.to_string()).or_insert(2.0),
                    // Default case for all other variables
                    _ => *values.entry(name.to_string()).or_insert(0.0),
                }
            };

            // List of expected variables
            let variables = [
                "cex1", "cey1", "cez1", "oriw1", "orii1", "orij1", "orik1",
                "lvx1", "lvy1", "lvz1", "avx1", "avy1", "avz1", "shx1", "shy1", "shz1", "req1", "rhos1",
                "cex2", "cey2", "cez2", "oriw2", "orii2", "orij2", "orik2",
                "lvx2", "lvy2", "lvz2", "avx2", "avy2", "avz2", "rhos2",
                "shx2", "shy2", "shz2", "rhof", "ndiv",
                "tend", "dt", "nbody"
            ];

            // Check each variable and assign default if necessary
            for &var in &variables {
                let value = check_and_assign_default(var);
                println!("'{}' = {}", var, value); // or use value as needed
            }

            println!("{:?}", values);


            let cex1 = *values.get("cex1").unwrap();
            let cey1 = *values.get("cey1").unwrap();
            let cez1 = *values.get("cez1").unwrap();

            position_1 = Vector3::new(cex1, cey1, cez1);
            println!("position_1: {:?}", position_1);

            let oriw1 = *values.get("oriw1").unwrap();
            let orii1 = *values.get("orii1").unwrap();
            let orij1 = *values.get("orij1").unwrap();
            let orik1 = *values.get("orik1").unwrap();

            orientation_1 = Quaternion::from_parts(oriw1, Vector3::new(orii1, orij1, orik1));
            orientation_1 = orientation_1.normalize();
            println!("orientation_1: {:?}", orientation_1);

            let lvx1 = *values.get("lvx1").unwrap();
            let lvy1 = *values.get("lvy1").unwrap();
            let lvz1 = *values.get("lvz1").unwrap();

            lin_velocity_1 = Vector3::new(lvx1, lvy1, lvz1);
            println!("lin_velocity_1: {:?}", lin_velocity_1);

            let avx1 = *values.get("avx1").unwrap();
            let avy1 = *values.get("avy1").unwrap();
            let avz1 = *values.get("avz1").unwrap();

            ang_velocity_1 = Quaternion::from_parts(0.0, Vector3::new(avx1, avy1, avz1));
            println!("ang_velocity_1: {:?}", ang_velocity_1);

            let shx1 = *values.get("shx1").unwrap();
            let shy1 = *values.get("shy1").unwrap();
            let shz1 = *values.get("shz1").unwrap();
            req1 = *values.get("req1").unwrap();


            let req1_temp = (shx1*shy1*shz1).powf(1./3.);
            let sf = 1./req1_temp*req1;
            println!("Equivalent radius of 1 = {:?}", req1);
            shape_1 = Vector3::new(shx1*sf, shy1*sf, shz1*sf);
            println!("shape_1: {:?}", shape_1);

            req1 = *values.get("req1").unwrap();
            println!("Equivalent radius of 1 = {:?}", req1);
            rho_s1 = *values.get("rhos1").unwrap();
            println!("Density of solid 1 = {:?}", rho_s1);

            let cex2 = *values.get("cex2").unwrap();
            let cey2 = *values.get("cey2").unwrap();
            let cez2 = *values.get("cez2").unwrap();

            position_2 = Vector3::new(cex2, cey2, cez2);
            println!("position_2: {:?}", position_2);

            let oriw2 = *values.get("oriw2").unwrap();
            let orii2 = *values.get("orii2").unwrap();
            let orij2 = *values.get("orij2").unwrap();
            let orik2 = *values.get("orik2").unwrap();

            orientation_2 = Quaternion::from_parts(oriw2, Vector3::new(orii2, orij2, orik2));
            orientation_2 = orientation_2.normalize();
            println!("orientation_2: {:?}", orientation_2);

            let lvx2 = *values.get("lvx2").unwrap();
            let lvy2 = *values.get("lvy2").unwrap();
            let lvz2 = *values.get("lvz2").unwrap();

            lin_velocity_2 = Vector3::new(lvx2, lvy2, lvz2);
            println!("lin_velocity_2: {:?}", lin_velocity_2);

            let avx2 = *values.get("avx2").unwrap();
            let avy2 = *values.get("avy2").unwrap();
            let avz2 = *values.get("avz2").unwrap();

            ang_velocity_2 = Quaternion::from_parts(0.0, Vector3::new(avx2, avy2, avz2));
            println!("ang_velocity_2: {:?}", ang_velocity_2);

            let shx2 = *values.get("shx2").unwrap();
            let shy2 = *values.get("shy2").unwrap();
            let shz2 = *values.get("shz2").unwrap();
            req2 = *values.get("req2").unwrap();
            println!("Equivalent radius of 2 = {:?}", req2);

            let req2_temp = (shx2*shy2*shz2).powf(1./3.);
            let sf = 1./req2_temp *  req2;
            shape_2 = Vector3::new(shx2*sf, shy2*sf, shz2*sf);
            println!("shape_2: {:?}", shape_2);


            rho_s2 = *values.get("rhos2").unwrap();
            println!("Density of solid 2 = {:?}", rho_s2);

            rho_f = *values.get("rhof").unwrap();
            println!("\nDensity of fluid = {:?}", rho_f);
            let ndiv_f64 = *values.get("ndiv").unwrap();
            ndiv = ndiv_f64 as usize;
            println!("{:?} divisions.", ndiv);
            t_end = *values.get("tend").unwrap();
            dt = *values.get("dt").unwrap();
            println!("Running with timestep = {:?}, until t={:?}", dt, t_end);
            tprint = *values.get("tprint").unwrap() as u32;
            let nbody_f64 = *values.get("nbody").unwrap();
            nbody = nbody_f64 as usize;
            println!("Running with {:?} body(s)", nbody);
        },
        Err(e) => {
            println!("Failed to parse: {:?}", e);
        }
    }


    let comment = format!("#Setup information to go here.");

    // let den = 1.0;
    // let s = Vector3::new(1.0, 1.0, 1.0);
    // let s0 = Vector3::new(1.0, 1.0, 0.6);
    // let q = Quaternion::from_parts(1.0, Vector3::new(0.0, 0.0, 0.0));
    // let omega = Quaternion::from_imag(Vector3::new(1.0, 1.0, 0.0));
    // let omega2 = Quaternion::from_imag(Vector3::new(1.0, 0.0, -1.0));
    //
    // let o_vec = Vector3::new(-1.0, 0.0, 0.0).normalize();
    // let o_vec2 = Vector3::new(1.0, 1.0, -1.0).normalize();
    // let init_ang_mom = o_vec.cross(&o_vec2).normalize();
    // let ang_mom_q = Quaternion::from_imag(init_ang_mom);
    // let q0 = Quaternion::from_real(0.0);

    let ratio= 20.0;

    let mut body1 = Body {
        density: rho_s1,
        shape: shape_1,
        position: position_1,
        orientation: orientation_1,
        linear_momentum: lin_velocity_1,
        angular_momentum: ang_velocity_1,
        inertia: is_calc(na::Matrix3::from_diagonal(&shape_1), rho_s1),
    };

    body1.set_linear_velocity(lin_velocity_1);
    body1.set_angular_velocity(ang_velocity_1);
    //Normalise initial conditions
    // let init_frequency = body1.rotational_frequency();
    //
    // body1.angular_momentum = body1.angular_momentum / init_frequency;
    //
    //
    // let init_direction = body1.linear_momentum;
    // body1.linear_momentum = body1.ic_generator(init_direction, ratio);


    let mut body2 = Body {
        density: rho_s2,
        shape: shape_2,
        position: position_2,
        orientation: orientation_2,
        linear_momentum:  lin_velocity_2,
        angular_momentum: ang_velocity_2,
        inertia: is_calc(na::Matrix3::from_diagonal(&shape_2), rho_s2),
    };

    body2.set_linear_velocity(lin_velocity_2);
    body2.set_angular_velocity(ang_velocity_2);



    //Setup Fluid
    let fluid = Fluid {
        density: rho_f,
        kinetic_energy: 0.0,
    };

    let ndiv = 0;

    let npts_circ = (4*2_usize.pow(ndiv)) as f64;
    let dx = (4.0 * PI) / npts_circ;
    // let dt_max = dx / body1.linear_velocity().norm();

    // println!("Angular momentum is {:?}", body1.angular_momentum.imag());

    println!("Building simulation");
    // Building System for simulation
    let sys  = Simulation::new(
        fluid,
        body1,
        body2,
        100.0,
        0.0000001,
        ndiv,
        10000,
        nbody,
    );

    println!("Simulation Built");



    let p1 = sys.body1.position;
    let p2 = sys.body2.position;
    let p = Vector6::new(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]);

    let v1 = sys.body1.linear_velocity();
    let v2 = sys.body2.linear_velocity();
    let v = Vector6::new(v1[0], v1[1], v1[2], v2[0], v2[1], v2[2]);

    let x = (p, v);

    let q1 = sys.body1.orientation;
    let q2 = sys.body2.orientation;
    // let q = (q1, q2);

    let omega1 = sys.body1.angular_velocity();
    let omega2 = sys.body2.angular_velocity();
    println!("omega1 = {:?}", omega1);
    // let omega = (omega1, omega2);

    // let o = (q, omega);

    let inertia1 = sys.body1.inertia;
    let inertia2 = sys.body2.inertia;

    let sys_mutex = Arc::new(Mutex::new(sys));

    let linear_system = bem_for_ode::LinearUpdate{
        system: sys_mutex.clone()
    };

    let angular_system = bem_for_ode::AngularUpdate{
        system: sys_mutex.clone(),
    };

    let forcing_system = bem_for_ode::ForceCalculate{
        system: sys_mutex.clone(),
    };


    let mut stepper = rk4pcdm::Rk4PCDM::new(
        linear_system,
        angular_system,
        forcing_system,
        0.0,
        x,
        (q1, omega1),
        (q2, omega2),
        inertia1,
        inertia2,
        t_end,
        dt,
        tprint);

    println!("Solver initialised");

    let nelm_end = 8_usize * 4_usize.pow(ndiv) * 2_usize;

    println!("Total number of elements = {:?}.", nelm_end);

    let res = stepper.integrate();

    match res {
        Ok(_) => {

            // let path_base_str = format!("./output_ellipse_sphere_surfgrad_norot/");
            let path_base_str = output_file_path;
            println!("Saving to {:?}", path_base_str);

            match std::fs::create_dir_all(path_base_str.clone()) {
                Ok(_) => {}
                Err(_) => {
                    panic!("Could not create output directories\n")
                }
            }

            let path_base = Path::new(&path_base_str);
            let sim_name = SimName::new(path_base);

            save(
                &stepper.t_out,
                &stepper.x_out,
                &stepper.o_out,
                &stepper.o_lab_out,
                &sim_name,
                &comment,
            );
        }
        Err(e) => println!("An error occurred {:?}", e),

    };

    // Ok(())

}





//Saving results
pub fn save(
    times: &Vec<Time>,
    states1: &Vec<Linear2State>,
    states2: &Vec<Angular2State>,
    states4: &Vec<Vector6<f64>>,

    filename: &SimName,
    comment: &str,
) {
    let file1 = match File::create(filename.complete_path()) {
        Err(e) => {
            println!("Could not open file. Error: {:?}", e);
            return;
        }
        Ok(buf) => buf,
    };

    let mut buf = BufWriter::new(file1);
    // buf.write_fmt(format_args!("{}", comment)).unwrap();


    buf.write_fmt(format_args!(
        "time,p1_1,p2_1,p3_1,p1_2,p2_2,p3_2,v1_1,v2_1,v3_1,v1_2,v2_2,v3_2,q1_1,q2_1,q3_1,q0_1,q1_2,q2_2,q3_2,q0_2,o1_1,o2_1,o3_1,o0_1,o1_2,o2_2,o3_2,o0_2,ofix1_1,ofix2_1,ofix3_1,ofix1_2,ofix2_2,ofix3_2\n"
    ))
        .unwrap();


    // Write time and state vector in a csv format
    for (i, state) in states1.iter().enumerate() {
        // println!("Iteration {}", i);
        //write time
        buf.write_fmt(format_args!("{}", times[i])).unwrap();
        //write linear quantities
        for val in state.0.iter() {
            buf.write_fmt(format_args!(", {}", val)).unwrap();
        }
        for val in state.1.iter() {
            buf.write_fmt(format_args!(", {}", val)).unwrap();
        }
        //write angular quantities
        let state2 = states2[i];
        for val in state2.0.0.as_vector().iter() {
            buf.write_fmt(format_args!(", {}", val)).unwrap();
        }
        for val in state2.0.1.as_vector().iter() {
            buf.write_fmt(format_args!(", {}", val)).unwrap();
        }
        for val in state2.1.0.as_vector().iter() {
            buf.write_fmt(format_args!(", {}", val)).unwrap();
        }
        for val in state2.1.1.as_vector().iter() {
            buf.write_fmt(format_args!(", {}", val)).unwrap();
        }
        // //write lab frame position
        // let state3 = states3[i];
        // for val in state3.iter() {
        //     buf.write_fmt(format_args!(", {}", val)).unwrap();
        // }
        //write lab orientation
        let state4 = states4[i];
        for val in state4.iter() {
            buf.write_fmt(format_args!(", {}", val)).unwrap();
        }
        buf.write_fmt(format_args!("\n")).unwrap();
    }

    if let Err(e) = buf.flush() {
        println!("Could not write to file. Error: {:?}", e);
    }
}

fn parse_assignment(input: &str) -> IResult<&str, (&str, f64)> {
    separated_pair(alphanumeric1, tag("="), double)(input)
}

fn parse_assignments(input: &str) -> IResult<&str, Vec<(&str, f64)>> {
    many0(terminated(
        parse_assignment,
        opt(line_ending)
    ))(input)
}
// fn parse_assignment(input: &str) -> IResult<&str, (String, f64)> {
//     separated_pair(
//         alphanumeric1,
//         char('='),
//         double
//     )(input)
//         .map(|(next_input, (name, value))| (next_input, (name.to_string(), value)))
// }
//
// fn parse_assignments(input: &str) -> IResult<&str, Vec<(String, f64)>> {
//     many0(terminated(
//         parse_assignment,
//         opt(terminated(multispace1, opt(multispace0)))
//     ))(input)
// }
