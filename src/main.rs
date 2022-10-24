use std::{fs::File, io::BufWriter, io::Write, path::Path};
use std::rc::Rc;
use nalgebra as na;
use nalgebra::{DMatrix, DVector, Quaternion, Vector3, Vector4};
use multi_ellip::ellipsoids::body::Body;
use multi_ellip::system::hamiltonian::is_calc;
use multi_ellip::bem::potentials::{ke_1body, f_1body, phi_1body_serial, f_2body, f_2body_serial, f_finder, phi_eval_1body};
use std::time::Instant;
use multi_ellip::system::fluid::Fluid;
use multi_ellip::system::system::Simulation;

type State = (na::OVector<f64, na::U3>, na::OVector<f64, na::U3>);
type State2 = (Quaternion<f64>, Quaternion<f64>);
type State3 = na::OVector<f64, na::U3>;
type Time = f64;



fn main() {
    println!("Hello, world!");

    let den = 1.0;
    let s = na::Vector3::new(1.0, 0.8, 0.6);
    let q = na::Quaternion::from_parts(1.0, na::Vector3::new(1.0, 0.5, 0.0));
    let o_vec = na::Vector3::new(-1.0, 0.0, 0.0).normalize();
    let o_vec2 = na::Vector3::new(1.0, 1.0, -1.0).normalize();
    let init_ang_mom = o_vec.cross(&o_vec2).normalize();
    let ang_mom_q = na::Quaternion::from_imag(init_ang_mom);
    let q0 = Quaternion::from_real(0.0);

    let ratio= 20.0;

    let mut body1 = Body {
        density: 1.0,
        shape: s,
        position: Vector3::new(1.0, 0.0, 0.0),
        orientation: q.normalize(),
        linear_momentum: o_vec * 6.0,
        angular_momentum: ang_mom_q,
        inertia: is_calc(na::Matrix3::from_diagonal(&s), den),
    };


    body1.linear_momentum = body1.linear_momentum_from_vel(Vector3::new(1.0, 0.0, 0.0));

    body1.print_vel();
    body1.print_mass();
    body1.print_lin_mom();
    //Normalise initial conditions
    let init_frequency = body1.rotational_frequency();
    println!("{:?}", init_frequency);
    body1.print_ang_mom();

    body1.angular_momentum = body1.angular_momentum / init_frequency;
    body1.print_ang_mom();

    let init_direction = body1.linear_momentum;
    body1.linear_momentum = body1.ic_generator(init_direction, ratio);



    let mut body2 = Body {
        density: 1.0,
        shape: s,
        position: Vector3::new(1.0, 0.0, 0.0),
        orientation: q.normalize(),
        linear_momentum: o_vec2,
        angular_momentum: q0,
        inertia: is_calc(na::Matrix3::from_diagonal(&s), den),
    };




    //Setup Fluid
    let fluid = Fluid {
        density: den,
        kinetic_energy: 0.0,
    };

    let ndiv = 3;
    println!("Building simulation");
    // Building System for simulation
    let sys = Rc::new(Simulation::new(
        fluid,
        body1,
        body2,
        100.0,
        0.000001,
        ndiv,
        10000,
        ratio,
    ));
    println!("Simulation Built");

    let v = na::Vector6::new(0.0, 1.0, 1.0, 1.0, 0.0, 0.0);
    let v4 = na::Vector4::new(v[0], v[1], v[2], v[3]);
    println!("{:?}", v4);
    // let vq = na::Quaternion::from_vector(v4);

    let ndiv = 3;
    let (nq, mint) = (12_usize, 13_usize);

    // let sing_par = Instant::now();
    // let f = f_1body(&body1, ndiv, nq, mint);
    // let sing_par_t = sing_par.elapsed();
    //
    // let sing_ser = Instant::now();
    // let f = phi_1body_serial(&body1, ndiv, nq, mint);
    // let sing_ser_t = sing_ser.elapsed();

    // let ke_t = Instant::now();
    // let ke_1 = ke_1body(&body1, ndiv, nq, mint);
    // let ke_elapse = ke_t.elapsed();
    //
    //
    // let p0 = Vector3::new(1.0, 0.0, 0.0);
    //
    // let phi_val = phi_eval_1body(&body1, ndiv, nq, mint, p0);

    let par_before = Instant::now();
    let double_d = f_2body(&body1, &body2, ndiv, nq, mint);
    let par_time = par_before.elapsed();


    // println!("Serial code took {:?}", sing_ser_t);
    // println!("Parallel code took {:?}", sing_par_t);
    //
    // println!("ke is {:?}, took {:?}", ke_1, ke_elapse);
    // println!("Phi value is {:?}", phi_val);
    //
    // println!("Serial code took {:?}", ser_time);
    // println!("Parallel code took {:?}", par_time);
}



//Saving results
pub fn save(
    times: &Vec<Time>,
    states1: &Vec<State>,
    states2: &Vec<State2>,
    states3: &Vec<State3>,
    states4: &Vec<Vector3<f64>>,

    filename: &Path,
    comment: String,
) {
    let file5 = match File::create(filename) {
        Err(e) => {
            println!("Could not open file. Error: {:?}", e);
            return;
        }
        Ok(buf) => buf,
    };

    let mut buf = BufWriter::new(file5);
    buf.write_fmt(format_args!(
        "time,p1,p2,p3,v1,v2,v3,q1,q2,q3,q0,o1,o2,o3,o0,pfix1,pfix2,pfix3,ofix1,ofix2,ofix3\n"
    ))
        .unwrap();
    buf.write_fmt(format_args!("{}", comment)).unwrap();

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
        for val in state2.0.as_vector().iter() {
            buf.write_fmt(format_args!(", {}", val)).unwrap();
        }
        for val in state2.1.as_vector().iter() {
            buf.write_fmt(format_args!(", {}", val)).unwrap();
        }
        //write lab frame position
        let state3 = states3[i];
        for val in state3.iter() {
            buf.write_fmt(format_args!(", {}", val)).unwrap();
        }
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