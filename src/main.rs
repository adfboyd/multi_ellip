use std::{fs::File, io::BufWriter, io::Write, path::Path};
use std::f64::consts::PI;
// use std::rc::Rc;
use std::sync::{Arc, Mutex};
use nalgebra as na;
use nalgebra::{DMatrix, DVector, Matrix3, Quaternion, UnitQuaternion, Vector3, Vector6};
use multi_ellip::ellipsoids::body::Body;
use multi_ellip::system::hamiltonian::is_calc;
use multi_ellip::bem::bem_for_ode;
use multi_ellip::bem::potentials::*;
use std::time::Instant;
use indicatif::ParallelProgressIterator;
use plotters::prelude::LogScalable;
// use rayon::prelude::IntoParallelRefIterator;
use rayon::iter::*;
use multi_ellip::bem::geom::{abc_vec, combiner, ellip_gridder, elm_geom, gauss_leg, gauss_trgl};
use multi_ellip::bem::integ::*;
use multi_ellip::ode::rk4pcdm;
use multi_ellip::system::fluid::Fluid;
use multi_ellip::system::system::Simulation;
use multi_ellip::utils::SimName;

type State2 = (Quaternion<f64>, Quaternion<f64>);


type Linear2State = (Vector6<f64>, Vector6<f64>);
type Angular2State = (State2, State2);

type Time = f64;



fn main() {
    println!("Hello, world!");

    let comment = format!("#Setup information to go here.");

    let den = 1.0;
    let s = Vector3::new(1.0, 1.0, 1.0);
    let s0 = Vector3::new(1.0, 1.0, 1.0);
    let q = Quaternion::from_parts(1.0, Vector3::new(0.0, 0.0, 0.0));
    let o_vec = Vector3::new(-1.0, 0.0, 0.0).normalize();
    let o_vec2 = Vector3::new(1.0, 1.0, -1.0).normalize();
    let init_ang_mom = o_vec.cross(&o_vec2).normalize();
    let ang_mom_q = Quaternion::from_imag(init_ang_mom);
    let q0 = Quaternion::from_real(0.0);

    let ratio= 20.0;

    let den1 = 1.0;
    let mut body1 = Body {
        density: den1,
        shape: s,
        position: Vector3::new(-100.0, 0.0, 0.0),
        orientation: q.normalize(),
        linear_momentum: Vector3::new(1.0, 0.0, 0.0),
        angular_momentum: q,
        inertia: is_calc(na::Matrix3::from_diagonal(&s), den1),
    };

    body1.set_linear_velocity(Vector3::new(1.0, 0.0, 0.0));
    body1.set_angular_velocity(q);
    //Normalise initial conditions
    // let init_frequency = body1.rotational_frequency();
    //
    // body1.angular_momentum = body1.angular_momentum / init_frequency;
    //
    //
    // let init_direction = body1.linear_momentum;
    // body1.linear_momentum = body1.ic_generator(init_direction, ratio);

    let den2 = 1.0;

    let mut body2 = Body {
        density: den2,
        shape: s0,
        position: Vector3::new(100.0, 0.0, 0.0),
        orientation: q.normalize(),
        linear_momentum:  Vector3::new(-0.0001, 0.0, 0.0),
        angular_momentum: q,
        inertia: is_calc(na::Matrix3::from_diagonal(&s0), den2),
    };

    body2.set_linear_velocity(Vector3::new(-1.0, 0.0, 0.0));
    body2.set_angular_velocity(q);



    //Setup Fluid
    let fluid = Fluid {
        density: den,
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
        ratio,
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
        10.0,
        0.001,
        100);

    println!("Solver initialised");

    let nelm_end = 8_usize * 4_usize.pow(ndiv) * 2_usize;

    println!("Total number of elements = {:?}.", nelm_end);

    let res = stepper.integrate();

    match res {
        Ok(_) => {

            let path_base_str = format!("./output0/");

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

    }

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