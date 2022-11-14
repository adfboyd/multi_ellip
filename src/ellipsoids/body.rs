use std::f64::consts::PI;
// use csv::*;
use nalgebra as na;
use nalgebra::{Matrix3, Vector3};
use serde::Serialize;

use crate::ellipsoids::state::State;
use crate::system::hamiltonian::{calc_shape_factor, if_calc, is_calc, mf_calc};

#[derive(Debug, Copy, Clone, Serialize)]
#[serde(rename_all = "PascalCase")]
pub struct Body {
    pub density: f64,
    pub shape: na::Vector3<f64>,
    pub position: na::Vector3<f64>,
    pub orientation: na::Quaternion<f64>,
    pub linear_momentum: na::Vector3<f64>,
    pub angular_momentum: na::Quaternion<f64>,
    pub inertia: na::Matrix3<f64>,
}

impl Body {
    pub fn mass(&self) -> f64 {
        (4.0 / 3.0)
            * std::f64::consts::PI
            * self.density
            * self.shape.iter().fold(1.0, |accum, &x| accum * x)
    }

    pub fn linear_velocity(&self) -> na::Vector3<f64> {
        (1.0 / self.mass()) * self.linear_momentum
    }

    pub fn linear_momentum_from_vel(&self, v :Vector3<f64>) -> Vector3<f64> {
        self.mass() * v
    }

    pub fn angular_velocity(&self) -> na::Quaternion<f64> {
        let omega = self.angular_momentum.imag() * 2.0;
        let vec = self.inertia.try_inverse().map(|m| m * omega).unwrap();
        na::Quaternion::from_imag(vec)
    }

    pub fn q(&self) -> State {
        let q = State {
            v: self.position,
            q: self.orientation,
        };
        q
    }

    pub fn p(&self) -> State {
        let p = State {
            v: self.linear_momentum,
            q: self.angular_momentum,
        };
        p
    }

    pub fn update_q(&mut self, q: State) {
        let State {
            v: p_new,
            q: o_new
        } = q;
        self.position = p_new;
        self.orientation = o_new;
    }

    pub fn update_p(&mut self, p: State) {
        let State {
            v: lin_mom,
            q: ang_mom,
        } = p;
        self.linear_momentum = lin_mom;
        self.angular_momentum = ang_mom;
    }

    // fn print_body<T: std::io::Write>(&self, wtr: &mut csv::Writer<T>) -> Result<()> {
    //     wtr.serialize(self)?;
    //     wtr.flush()?;
    //     Ok(())
    // }

    pub fn linear_energy(&self) -> f64 {
        let lin_mom = self.linear_momentum;
        lin_mom.dot(&lin_mom) / (2.0 * self.mass())
    }

    pub fn rotational_energy(&self) -> f64 {
        let ang_mom = self.angular_momentum.imag();
        let ang_vel_q = self.angular_velocity();
        let ang_vel = ang_vel_q.vector();
        0.5 * ang_vel.dot(&ang_mom)
    }

    pub fn rotational_frequency(&self) -> f64 {
        let ang_vel_q = self.angular_velocity();
        let abs_ang_vel = ang_vel_q.vector().norm();
        abs_ang_vel / (2.0 * PI)
    }


    // pub(crate) fn kinetic_energy(&self) -> f64 {
    //     let lin_mom = self.linear_momentum;
    //     let ang_mom = self.angular_momentum.imag();
    //     let ang_vel_q = self.angular_velocity();
    //     let ang_vel = ang_vel_q.vector();
    //
    //     lin_mom.dot(&lin_mom) / (2.0 * self.mass()) + 0.5 * ang_vel.dot(&ang_mom)
    // }

    pub fn energy_ratio(&self) -> f64 {
        let lin_energy = self.linear_energy();
        let rot_energy = self.rotational_energy();
        lin_energy / rot_energy
    }

    pub fn ic_generator(&self, direction: na::Vector3<f64>, ratio: f64) -> na::Vector3<f64> {
        let desired_ke = ratio * self.rotational_energy();
        let p_scalar = (2.0 * desired_ke * self.mass()).sqrt();
        p_scalar * direction.normalize()
    }

    pub fn inertia_tensor(&self, rho_f :f64) -> (Matrix3<f64>, Matrix3<f64>) {


        let m_s = na::Matrix3::from_diagonal(&self.shape);
        let v_s = 4.0 / 3.0 * PI * self.shape.iter().fold(1.0, |acc, x| acc * x);

        let lambda = 10000.0;

        let shape_fun = calc_shape_factor(lambda, m_s).unwrap();

        let alpha = shape_fun.alpha;
        let beta = shape_fun.beta;
        let gamma = shape_fun.gamma;
        let m_f = mf_calc(alpha, beta, gamma, v_s, rho_f);

        let m: Matrix3<f64> = m_f + m_s;

        let i_f = if_calc(alpha, beta, gamma, v_s, rho_f, m_s);
        let i_s = is_calc(m_s, self.density);
        let i: Matrix3<f64> = i_f + i_s;

        (m, i)
    }

    pub fn print_vel(&self) {
        println!("Linear velocity is {:?}", self.linear_velocity());
    }
    pub fn print_omega(&self) {
        println!("Angular velocity is {:?}", self.angular_velocity());
    }
    pub fn print_lin_mom(&self) {
        println!("Linear momentum is {:?}", self.linear_momentum);
    }
    pub fn print_ang_mom(&self) {
        println!("Angular momentum is {:?}", self.angular_momentum);
    }
    pub fn print_mass(&self) {
        println!("mass is {:?}", self.mass());

    }
}