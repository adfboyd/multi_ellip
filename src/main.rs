use nalgebra as na;
use nalgebra::{DMatrix, DVector, Quaternion, Vector3};
use multi_ellip::bem::geom::{abc_vec, combiner, ellip_gridder, elm_geom, gauss_leg, gauss_trgl};
use multi_ellip::bem::integ::{ldlp_3d, lslp_3d};
use multi_ellip::ellipsoids::body::Body;
use multi_ellip::system::hamiltonian::is_calc;
use rayon::prelude::*;
use multi_ellip::bem::potentials::{phi_1body, phi_1body_serial, phi_2body, phi_2body_serial, phi_finder};
use std::time::Instant;


fn main() {
    println!("Hello, world!");

    let den = 5.0;
    let s = na::Vector3::new(1.0, 0.6, 0.2);
    let q = na::Quaternion::from_parts(1.0, na::Vector3::new(0.0, 0.0, 0.0));
    let o_vec = na::Vector3::new(1.0, 1.0, 1.0).normalize();
    let o_vec2 = na::Vector3::new(1.0, 1.0, -1.0).normalize();
    let init_ang_mom = o_vec.cross(&o_vec2).normalize();
    let ang_mom_q = na::Quaternion::from_imag(init_ang_mom);


    let body1 = Body {
        density: 1.0,
        shape: s,
        position: Vector3::zeros(),
        orientation: q.normalize(),
        linear_momentum: o_vec,
        angular_momentum: ang_mom_q,
        inertia: is_calc(na::Matrix3::from_diagonal(&s), den),
    };

    let body2 = Body {
        density: 1.0,
        shape: s,
        position: Vector3::new(2.0, 0.0, 0.0),
        orientation: q.normalize(),
        linear_momentum: o_vec2,
        angular_momentum: ang_mom_q,
        inertia: is_calc(na::Matrix3::from_diagonal(&s), den),
    };


    let ndiv = 3;
    let (nq, mint) = (3_usize, 7_usize);

    let sing_par = Instant::now();
    let f = phi_1body(&body1, ndiv, nq, mint);
    let sing_par_t = sing_par.elapsed();

    let sing_ser = Instant::now();
    let f = phi_1body_serial(&body1, ndiv, nq, mint);
    let sing_ser_t = sing_ser.elapsed();

    let par_before = Instant::now();
    let double_d = phi_2body(&body1, &body2, ndiv, nq, mint);
    let par_time = par_before.elapsed();

    let ser_before = Instant::now();
    let double_d_ser = phi_2body_serial(&body1, &body2, ndiv, nq, mint);
    let ser_time = ser_before.elapsed();

    println!("Serial code took {:?}", sing_ser_t);
    println!("Parallel code took {:?}", sing_par_t);

    println!("Serial code took {:?}", ser_time);
    println!("Parallel code took {:?}", par_time);
}
//
//
//
// fn gradp(x :na::Vector3<f64>, p :na::Vector3<f64>, m :f64) -> na::Vector3<f64> {
//     let v = x - p;
//     let denom = v.norm().powi(3);
//
//     let res =  if denom != 0.0 {
//         (m/denom) * v}
//     else {
//         na::Vector3::new(0.0, 0.0, 0.0)
//     };
//     res
// }
//
// fn flux(x :na::Vector3<f64>, p :na::Vector3<f64>, m :f64, body :Body) -> f64 {
//     let n = body.norm_vec(x);
//     let gp = gradp(x, p, m);
//     let res = n.dot(&gp);
//     res
// }
//
// fn dist(x :na::Vector3<f64>) -> f64 {
//     x.norm()
// }