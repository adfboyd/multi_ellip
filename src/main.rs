use std::{fs::File, io::BufWriter, io::Write, path::Path};
use std::f32::consts::PI;
// use std::rc::Rc;
use std::sync::{Arc, Mutex};
use nalgebra as na;
use nalgebra::{DMatrix, DVector, Quaternion, UnitQuaternion, Vector3, Vector6};
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

    let comment = format!("Testing results");

    let den = 1.0;
    let s = Vector3::new(1.0, 0.8, 0.6);
    let s0 = Vector3::new(1.0, 1.0, 1.0);
    let q = Quaternion::from_parts(1.0, Vector3::new(1.0, 0.5, 0.0));
    let o_vec = Vector3::new(-1.0, 0.0, 0.0).normalize();
    let o_vec2 = Vector3::new(1.0, 1.0, -1.0).normalize();
    let init_ang_mom = o_vec.cross(&o_vec2).normalize();
    let ang_mom_q = Quaternion::from_imag(init_ang_mom);
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


    //Normalise initial conditions
    let init_frequency = body1.rotational_frequency();

    body1.angular_momentum = body1.angular_momentum / init_frequency;


    let init_direction = body1.linear_momentum;
    body1.linear_momentum = body1.ic_generator(init_direction, ratio);



    let body2 = Body {
        density: 1.0,
        shape: s0,
        position: Vector3::new(1.0, 0.0, 0.0),
        orientation: q.normalize(),
        linear_momentum: o_vec2,
        angular_momentum: ang_mom_q,
        inertia: is_calc(na::Matrix3::from_diagonal(&s), den),
    };




    //Setup Fluid
    let fluid = Fluid {
        density: den,
        kinetic_energy: 0.0,
    };

    let ndiv =3;
    println!("Building simulation");
    // Building System for simulation
    let sys  = Simulation::new(
        fluid,
        body1,
        body2,
        100.0,
        0.000001,
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
        100.0,
        0.1,
        10);

    println!("Solver initialised");

    let res = stepper.integrate();

    match res {
        Ok(_) => {

            let path_base_str = format!("./output/");

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

    let ndiv = 3;
    let (nq, mint) = (12_usize, 13_usize);
    let req= 1.0;
    //
    // let (nelm1, npts1, p1, n1) = ellip_gridder(ndiv, req, s, p1, UnitQuaternion::from_quaternion(q1));
    // let (nelm2, npts2, p2, n2) = ellip_gridder(ndiv, req, s0, p2, UnitQuaternion::from_quaternion(q2));
    //
    // let (nelm, npts, p, n) = combiner(nelm1, nelm2, npts1, npts2, &p1, &p2, &n1, &n2);
    let (zz, ww) = gauss_leg(nq);
    let (xiq, etq, wq) = gauss_trgl(mint);
    //
    // let (alpha1, beta1, gamma1) = abc_vec(nelm1, &p1, &n1);
    // let (alpha2, beta2, gamma2) = abc_vec(nelm2, &p2, &n2);
    // let (alpha, beta, gamma) = abc_vec(nelm, &p, &n);
    //
    // let (vna1, vol1, sa1) = elm_geom(npts1, nelm1, mint,
    //                                  &p1, &n1,
    //                                  &alpha1, &beta1, &gamma1,
    //                                  &xiq, &etq, &wq);
    //
    // let (vna2, vol2, sa2) = elm_geom(npts2, nelm2, mint,
    //                                  &p2, &n2,
    //                                  &alpha2, &beta2, &gamma2,
    //                                  &xiq, &etq, &wq);
    //
    // let (vna, vol, sa) = elm_geom(npts, nelm, mint,
    //                                        &p, &n,
    //                                        &alpha, &beta, &gamma,
    //                                        &xiq, &etq, &wq);
    //
    // println!("Vol1, sa1 = {:?}, {:?}", vol1, sa1);
    // println!("Vol2, sa2 = {:?}, {:?}", vol2, sa2);
    // println!("Vol, sa = {:?}, {:?}", vol, sa);
    // println!("4/3 pi = {:?}, \n 4 pi = {:?}", 4.0/3.0 * PI, 4.0 * PI);
    //
    // let phi = phi_eval_1body(&body2, 2, nq, mint, s0*115000.0);
    // println!("phi = {:?}", phi);
    //
    // let dfdn2 = dfdn_single(&body2.position, &body2.linear_velocity(), &body2.angular_velocity().imag(), npts2, &p2, &vna2);

    // let s0 = Vector3::new(1.0, 0.8, 0.6);
    // let q = Quaternion::new(1.0, 1.0, 1.0, 1.0);
    // let req = (s0[0] * s0[1] * s0[2]).as_f64().powf(1.0/3.0);
    //
    // let mut body = Body {
    //     density: 1.0,
    //     shape: s0,
    //     position: Vector3::new(0.0, 0.0, 0.0),
    //     orientation: q.normalize(),
    //     linear_momentum: o_vec2,
    //     angular_momentum: q0 * 0.0,
    //     inertia: is_calc(na::Matrix3::from_diagonal(&s), den),
    // };
    // let v0 = Vector3::new(1.0, 0.0, 0.0);
    // body.set_linear_velocity(v0);
    // println!("LinVel = {:?}", body.linear_velocity());
    // body.print_ang_mom();
    //
    // let (nelm, npts, p, n) = ellip_gridder(3, req, body.shape, body.position, UnitQuaternion::from_quaternion(body.orientation));
    //
    // println!("p = {:?}", p);
    // let (alpha, beta, gamma) = abc_vec(nelm, &p, &n);
    // let (vna, vol, sa) = elm_geom(npts, nelm, mint,
    //                               &p, &n,
    //                               &alpha, &beta, &gamma,
    //                               &xiq, &etq, &wq);
    // let i = 0;
    //
    //
    // println!("vna = {:?}, {:?}, {:?}", vna[(i, 0)], vna[(i,1)], vna[(i,2)]);
    //
    //
    // let dfdn = dfdn_single(&body.position, &body.linear_velocity(), &body2.angular_velocity().imag(), npts, &p, &vna);
    //
    // // println!("dfdn = {:?}", dfdn);
    // let rhs = lslp_3d(npts, nelm, mint, nq,
    //                   &dfdn, &p, &n, &vna,
    //                   &alpha, &beta, &gamma,
    //                   &xiq, &etq, &wq, &zz, &ww);
    //
    // println!("Correct surface area = {:?}", sa);
    // println!("SA estimate = {:?}", body.surface_area_estimate());
    //
    //
    // println!("rhs = {:?}", rhs);
    //
    // let amat_1 = DMatrix::zeros(npts, npts);
    // let amat = Mutex::new(amat_1);
    //
    // let js = (0..npts).collect::<Vec<usize>>();
    //
    // js.par_iter().progress_count(npts as u64).for_each(|&j| {
    //     // println!("Computing column {} of the influence matrix", j);
    //     let mut q = DVector::zeros(npts);
    //
    //     q[j] = 1.0;
    //
    //     let dlp = ldlp_3d(npts, nelm, mint,
    //                       &q, &p, &n, &vna,
    //                       &alpha, &beta, &gamma,
    //                       &xiq, &etq, &wq);
    //
    //     let mut amat = amat.lock().unwrap();
    //
    //     for k in 0..npts {
    //         amat[(k, j)] = dlp[k];
    //     }
    // });
    //
    // let amat_final = amat.into_inner().unwrap();
    //
    // let decomp = amat_final.lu();
    //
    // let f = decomp.solve(&rhs).expect("Linear resolution failed");
    //
    // println!("Linear system solved!");
    //
    // println!("f = {:?}", f);

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
    //
    // let par_before = Instant::now();
    // let _double_d = f_2body(&body1, &body2, 2, nq, mint);
    // let par_time = par_before.elapsed();
    //
    //
    // // println!("Serial code took {:?}", sing_ser_t);
    // // println!("Parallel code took {:?}", sing_par_t);
    // //
    // // println!("ke is {:?}, took {:?}", ke_1, ke_elapse);
    // // println!("Phi value is {:?}", phi_val);
    // //
    // // println!("Serial code took {:?}", ser_time);
    // println!("Parallel code took {:?}", par_time);
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
    buf.write_fmt(format_args!(
        "time,p1_1,p2_1,p3_1,p1_2,p2_2,p3_2,v1_1,v2_1,v3_1,v1_2,v2_2,v3_2,q1_1,q2_1,q3_1,q0_1,q1_2,q2_2,q3_2,q0_2,o1_1,o2_1,o3_1,o0_1,o1_2,o2_2,o3_2,o0_2,ofix1_1,ofix2_1,ofix3_1,ofix1_2,ofix2_2,ofix3_2\n"
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