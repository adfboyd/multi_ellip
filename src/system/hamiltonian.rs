use nalgebra as na;
use std::f64::consts::PI;

use crate::ode::Dop853;

type State = na::Vector3<f64>;
type Time = f64;

struct Ellipsoid {
    a: f64,
    b: f64,
    c: f64,
}

impl crate::ode::System<State> for Ellipsoid {
    fn system(&self, t: Time, _y: &State, dy: &mut State) {
        let k_l = ((self.a.powi(2) + t) * (self.b.powi(2) + t) * (self.c.powi(2) + t)).sqrt();
        let pre_factor = self.a * self.b * self.c;
        dy[0] = pre_factor / ((self.a.powi(2) + t) * k_l);
        dy[1] = pre_factor / ((self.b.powi(2) + t) * k_l);
        dy[2] = pre_factor / ((self.c.powi(2) + t) * k_l);
    }
}

pub(crate) struct ShapeFunction {
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
}

pub fn k_tot(
    //Calculates total kinetic energy ie the Hamiltonian in single-body case
    v: &na::Vector3<f64>,
    omega: &na::Vector3<f64>,
    m: &na::Matrix3<f64>,
    i: &na::Matrix3<f64>,
) -> f64 {
    let m_v = v.dot(&(m * v));
    let ke = 0.5 * m_v;

    let re = 0.5 * omega.dot(&(i * omega));
    ke + re
}

// fn m_matrix(m_f: na::Matrix3<f64>, m_s: na::Matrix3<f64>) -> na::Matrix3<f64> {
//     m_f + m_s
// } //Adds the solid and fluid mass tensors

pub fn mf_calc(alpha: f64, beta: f64, gamma: f64, v_s: f64, rho_f: f64) -> na::Matrix3<f64> {
    //calculates the mass tensor for the fluid
    let m1 = na::Matrix3::from_diagonal(&na::Vector3::new(
        alpha / (2.0 - alpha),
        beta / (2.0 - beta),
        gamma / (2.0 - gamma),
    ));
    let mf = v_s * rho_f * m1;
    mf
}

pub fn ms_calc(a: f64, b: f64, c: f64) -> na::Matrix3<f64> {
    //calculates the mass tensor for the solid
    na::Matrix3::from_diagonal(&na::Vector3::new(a, b, c))
}

pub fn if_calc(
    //calculates the moment of inertia of the fluid
    alpha: f64,
    beta: f64,
    gamma: f64,
    v_s: f64,
    rho_f: f64,
    m_s: na::Matrix3<f64>,
) -> na::Matrix3<f64> {
    let a: f64 = m_s[(0, 0)];
    let b: f64 = m_s[(1, 1)];
    let c: f64 = m_s[(2, 2)];

    let e1 = ((b.powi(2) - c.powi(2)).powi(2) * (gamma - beta))
        / (2.0 * (b.powi(2) - c.powi(2)) + (beta - gamma) * (b.powi(2) + c.powi(2)));
    let e2 = ((a.powi(2) - c.powi(2)).powi(2) * (gamma - alpha))
        / (2.0 * (a.powi(2) - c.powi(2)) + (alpha - gamma) * (a.powi(2) + c.powi(2)));
    let e3 = ((a.powi(2) - b.powi(2)).powi(2) * (beta - alpha))
        / (2.0 * (a.powi(2) - b.powi(2)) + (alpha - beta) * (a.powi(2) + b.powi(2)));

    let i1 = na::Matrix3::from_diagonal(&na::Vector3::new(e1, e2, e3));
    let i_f = 0.2 * v_s * rho_f * i1;
    i_f
}

pub fn is_calc(m_s: na::Matrix3<f64>, rho_s: f64) -> na::Matrix3<f64> {
    //calculates the moment of inertia of the solid
    let mass = mass_calc(m_s, rho_s);

    let a: f64 = m_s[(0, 0)];
    let b: f64 = m_s[(1, 1)];
    let c: f64 = m_s[(2, 2)];

    let i1 = mass * (b.powi(2) + c.powi(2)) * 0.2;
    let i2 = mass * (a.powi(2) + c.powi(2)) * 0.2;
    let i3 = mass * (a.powi(2) + b.powi(2)) * 0.2;

    let i_s = na::Matrix3::from_diagonal(&na::Vector3::new(i1, i2, i3));

    i_s
}

fn mass_calc(m_s: na::Matrix3<f64>, rho_s: f64) -> f64 {
    //calculates the mass of the ellipsoid
    let a: f64 = m_s[(0, 0)];
    let b: f64 = m_s[(1, 1)];
    let c: f64 = m_s[(2, 2)];

    let vol = (4.0 / 3.0) * PI * a * b * c;
    let mass = vol * rho_s;

    mass
}

pub(crate) fn calc_shape_factor(lambda: f64, m_s: na::Matrix3<f64>) -> Option<ShapeFunction> {
    //calculates the shape factor
    let a: f64 = m_s[(0, 0)];
    let b: f64 = m_s[(1, 1)];
    let c: f64 = m_s[(2, 2)];

    // Initial state
    let y0 = State::new(0.0, 0.0, 0.0);

    //setting up integrator
    let system = Ellipsoid { a, b, c };

    let mut stepper = Dop853::new(system, 0.0, lambda, 1e-3, y0, 1e-4, 1e-4);
    let res = stepper.integrate();

    match res {
        Ok(_) => {
            let val = stepper.y_out().last().unwrap();
            Some(ShapeFunction {
                alpha: val[0],
                beta: val[1],
                gamma: val[2],
            })
        }
        Err(e) => {
            println!("An error occured: {:?}", e);
            None
        }
    }
}

/*
fn tests() {
    let (a, b, c) = (1.0, 0.8, 0.6);
    let rho_f = 2.0;

    let m_s = ms_calc(a, b, c);

    let alpha = alpha_calc(10.0.powi(50), M_s);
    let beta = beta_calc(10.0.powi(50), M_s);
    let gamma = gamma_calc(10.0.powi(50), M_s);
    let m_f = mf_calc(alpha, beta, gamma, V_s, rho_f);

    let M: Matrix4<f64> = m_f + m_s;

    let i_f = if_calc(alpha, beta, gamma, V_s, rho_f, m_s);
    let i_s = is_calc(m_s, rho_f);
    let I: Matrix4<f64> = i_f + i_s;

    let ke = k_tot(v, omega, &M, &I);
}
 */
