use nalgebra as na;

pub fn body_to_lab(p_body: &na::Quaternion<f64>, q: &na::Quaternion<f64>) -> na::Quaternion<f64> {
    // let &q_quaternion = q.quaternion();
    let q_inv = q.try_inverse().unwrap();
    let p_space = q * (p_body * q_inv);
    p_space
}

pub fn lab_to_body(p_lab: &na::Quaternion<f64>, q: &na::Quaternion<f64>) -> na::Quaternion<f64> {
    // let &q_quaternion = q.quaternion();
    let q_inv = q.try_inverse().unwrap();
    let p_body = q_inv * (p_lab * q);
    p_body
}

pub fn accel_get(
    omega: &na::Quaternion<f64>,
    inertia: &na::Matrix3<f64>,
    torque: &na::Quaternion<f64>,
) -> na::Quaternion<f64> {
    let inertia_inv = inertia.try_inverse().unwrap();
    let omega_v = omega.imag();
    let torque_v = torque.imag();
    let test: na::SVector<f64, 3> = inertia * omega_v;
    let accel = inertia_inv * (torque_v - omega_v.cross(&test));
    na::Quaternion::from_imag(accel)
}
