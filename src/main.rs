use nalgebra as na;
use nalgebra::{DMatrix, DVector, Quaternion, Vector3};
use multi_ellip::ellipsoids::body::Body;
use multi_ellip::system::hamiltonian::is_calc;
use multi_ellip::bem::potentials::{ke_1body, f_1body, phi_1body_serial, f_2body, f_2body_serial, f_finder};
use std::time::Instant;


fn main() {
    println!("Hello, world!");

    let den = 5.0;
    let s = na::Vector3::new(1.0, 0.8, 0.6);
    let q = na::Quaternion::from_parts(1.0, na::Vector3::new(1.0, 0.5, 0.0));
    let o_vec = na::Vector3::new(-1.0, 0.0, 0.0).normalize();
    let o_vec2 = na::Vector3::new(1.0, 1.0, -1.0).normalize();
    let init_ang_mom = o_vec.cross(&o_vec2).normalize();
    let ang_mom_q = na::Quaternion::from_imag(init_ang_mom);
    let q0 = Quaternion::from_real(0.0);


    let body1 = Body {
        density: 1.0,
        shape: s,
        position: Vector3::new(1.0, 0.0, 0.0),
        orientation: q.normalize(),
        linear_momentum: o_vec * 6.0,
        angular_momentum: ang_mom_q * 0.0,
        inertia: is_calc(na::Matrix3::from_diagonal(&s), den),
    };

    let body2 = Body {
        density: 1.0,
        shape: s,
        position: Vector3::new(0.0, 0.0, 0.0),
        orientation: q.normalize(),
        linear_momentum: o_vec2,
        angular_momentum: q0,
        inertia: is_calc(na::Matrix3::from_diagonal(&s), den),
    };


    let ndiv = 3;
    let (nq, mint) = (12_usize, 13_usize);

    let sing_par = Instant::now();
    let f = f_1body(&body1, ndiv, nq, mint);
    let sing_par_t = sing_par.elapsed();

    let sing_ser = Instant::now();
    let f = phi_1body_serial(&body1, ndiv, nq, mint);
    let sing_ser_t = sing_ser.elapsed();

    let ke_t = Instant::now();
    let ke_1 = ke_1body(&body1, ndiv, nq, mint);
    let ke_elapse = ke_t.elapsed();

    // let par_before = Instant::now();
    // let double_d = f_2body(&body1, &body2, ndiv, nq, mint);
    // let par_time = par_before.elapsed();
    //
    // let ser_before = Instant::now();
    // let double_d_ser = f_2body_serial(&body1, &body2, ndiv, nq, mint);
    // let ser_time = ser_before.elapsed();

    // println!("Serial code took {:?}", sing_ser_t);
    // println!("Parallel code took {:?}", sing_par_t);

    println!("ke is {:?}, took {:?}", ke_1, ke_elapse);

    // println!("Serial code took {:?}", ser_time);
    // println!("Parallel code took {:?}", par_time);
}
