//! Rigid-body rotation math: frame transforms between lab and body frames, and
//! the body-frame Euler angular acceleration. Single home for helpers that were
//! previously duplicated between `ode::pcdm` and `ode::rk4pcdm`.

use nalgebra::{Matrix3, Quaternion};

/// Rotate a pure (vector) quaternion `p_space` from the lab frame into the body
/// frame of orientation `q`. Guards a degenerate (near-zero-norm) orientation so
/// a blown-up run yields NaN rather than panicking in `try_inverse`.
pub fn lab_to_body(p_space: &Quaternion<f64>, q: &Quaternion<f64>) -> Quaternion<f64> {
    let q_inv = if q.norm() > 0.00001 {
        q.try_inverse().unwrap()
    } else {
        Quaternion::from_real(0.0)
    };
    q_inv * (p_space * q)
}

/// Rotate a pure (vector) quaternion `p_body` from the body frame of orientation
/// `q` into the lab frame. Same degenerate-orientation guard as `lab_to_body`.
pub fn body_to_lab(p_body: &Quaternion<f64>, q: &Quaternion<f64>) -> Quaternion<f64> {
    let q_inv = if q.norm() > 0.00001 {
        q.try_inverse().unwrap()
    } else {
        Quaternion::from_real(0.0)
    };
    q * (p_body * q_inv)
}

/// Body-frame Euler angular acceleration `I^{-1}(N - omega x (I omega))`, where
/// `omega` and `torque` are pure quaternions in the body frame.
pub fn accel_get(
    omega: &Quaternion<f64>,
    inertia: &Matrix3<f64>,
    torque: &Quaternion<f64>,
) -> Quaternion<f64> {
    let inertia_inv = inertia.try_inverse().unwrap();
    let omega_v = omega.imag();
    let torque_v = torque.imag();
    let test: nalgebra::SVector<f64, 3> = inertia * omega_v;
    let accel = inertia_inv * (torque_v - omega_v.cross(&test));
    Quaternion::from_imag(accel)
}
