use itertools::Itertools;
use nalgebra as na;
use nalgebra::{Quaternion, Vector3};
use multi_ellip::ellipsoids::body::Body;
use multi_ellip::system::hamiltonian::is_calc;



fn main() {
    println!("Hello, world!");

    let den = 5.0;
    let s = na::Vector3::new(1.0, 0.8, 0.6);
    let q = na::Quaternion::from_parts(1.0, na::Vector3::new(0.0, 0.0, 0.0));
    let o_vec = na::Vector3::new(1.0, 1.0, 1.0).normalize();
    let o_vec2 = na::Vector3::new(1.0, 1.0, -1.0).normalize();
    let init_ang_mom = o_vec.cross(&o_vec2).normalize();
    let ang_mom_q = init_ang_mom.quaternion();


    let mut body = Body {
        density: 1.0,
        shape: s,
        position: Vector3::zeros(),
        orientation: q.normalize(),
        linear_momentum: o_vec,
        angular_momentum: *ang_mom_q,
        inertia: is_calc(na::Matrix3::from_diagonal(&s), den),
    };

    let surface_area = body.integ(&f0);
    let sa_est = body.sa_est();

    println!("Calculated area = {:?}", surface_area);
    println!("A good estimate should be {:?}", sa_est);

}

fn f0(_x :na::Vector3<f64>) -> f64 {
    1.0
}
