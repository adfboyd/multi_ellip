use nalgebra as na;
use nalgebra::{Quaternion, Vector3};
use multi_ellip::bem::geom::{abc_vec, combiner, ellip_gridder, elm_geom, gauss_leg, gauss_trgl};
// use multi_ellip::ellipsoids::body::Body;
use multi_ellip::system::hamiltonian::is_calc;



fn main() {
    println!("Hello, world!");

    let den = 5.0;
    let s = na::Vector3::new(1.0, 0.8, 0.6);
    let q = na::Quaternion::from_parts(1.0, na::Vector3::new(0.0, 0.0, 0.0));
    let o_vec = na::Vector3::new(1.0, 1.0, 1.0).normalize();
    let o_vec2 = na::Vector3::new(1.0, 1.0, -1.0).normalize();
    let init_ang_mom = o_vec.cross(&o_vec2).normalize();
    let ang_mom_q = na::Quaternion::from_imag(init_ang_mom);


    // let mut body = Body {
    //     density: 1.0,
    //     shape: s,
    //     position: Vector3::zeros(),
    //     orientation: q.normalize(),
    //     linear_momentum: o_vec,
    //     angular_momentum: ang_mom_q,
    //     inertia: is_calc(na::Matrix3::from_diagonal(&s), den),
    // };

    // let sa_est = body.sa_est();
    // let sa = body.sa_calc();
    //
    // println!("Calculated area = {:?} for body of shape {:?}", sa, body.shape);
    // println!("A good estimate should be {:?}", sa_est);
    //
    // let m = 0.25;
    // let p = na::Vector3::new(0.5, 0.4, 0.0);
    //
    // let f = |x :na::Vector3<f64>| -> f64 {
    //     flux(x, p, m, body)
    // };
    //
    // let ave_dist = body.integ(&dist) / sa;
    //
    // println!("Average distance of a point on ellipsoid is {:?}", ave_dist);
    // // for i in 0..100000 {
    // //     let point_flux = body.integ(&f);
    // //     if i%1000 == 0 {
    // //         println!("{:?}", i);
    // //     }
    // //     // println!("Total flux from charge is {:?}", point_flux);
    // // };
    //
    // let pf = body.integ(&f);
    // println!("Total flux from charge is {:?}", pf);
    // let er = pf - std::f64::consts::PI;
    // println!("Error - {:?}", er);
    let p = na::Vector3::new(1.0, 0.0, 0.0);
    let q_unit = na::UnitQuaternion::from_quaternion(q);
    let req1:f64 = (s[0] * s[1] * s[2]);
    let req = req1.powf(1.0/3.0);
    // let (nelm1, npts1, p1, n1) = ellip_gridder(4, req, s, p, q_unit);
    // let (nelm2, npts2, p2, n2) = ellip_gridder(3, req, s, p, q_unit);
    // let (nelm, npts, p, n) = combiner(nelm1, nelm2, npts1, npts2, p1, p2, n1, n2);
    // for i in 0..50{
    //     let (xi, yi, zi) = (p[(i, 0)], p[(i, 1)], p[(i, 2)]);
    //     println!("{:?}, {:?}, {:?}", xi, yi, zi);
    //     let f = ((xi - p[0]) / s[0]).powi(2) + (yi / s[1]).powi(2) + (zi / s[2]).powi(2);
    //     println!("{}", f);
    //
    //     // println!("{}, {}, {}", n[(i, 0)], n[(i, 1)], n[(i, 2)]);
    //
    // }
    let req = 1.0;
    let ndiv = 5;
    let (nelm, npts, p, n) = ellip_gridder(ndiv, req, s, p, q_unit);

    let (nq, mint) = (6_usize, 7_usize);

    let (zz, ww) = gauss_leg(nq);
    let (xiq, etq, wq) = gauss_trgl(mint);

    let (alpha, beta, gamma) = abc_vec(nelm, &p, &n);

    let (vna, vlm, sa) = elm_geom(npts, nelm, mint,
                                  &p, &n,
                                  &alpha, &beta, &gamma,
                                  &xiq, &etq, &wq);

    let vlm_exact = 4.0 / 3.0 * std::f64::consts::PI * req.powi(3);

    println!("Volume approximation error = {}", (vlm - vlm_exact).abs());


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