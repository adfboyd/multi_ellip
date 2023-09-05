use std::f64::consts::PI;
// use csv::*;
use nalgebra as na;
use nalgebra::{Matrix3, Quaternion, Vector3};
use serde::Serialize;

use crate::ellipsoids::state::State;
use crate::ode::pcdm::{body_to_lab, lab_to_body};
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

    pub fn set_linear_velocity(&mut self, v :Vector3<f64>) {
        let linear_momentum = self.linear_momentum_from_vel(v);

        self.linear_momentum = linear_momentum;
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
    pub fn print_mass(&self) { println!("Mass is {:?}", self.mass()); }
    pub fn print_position(&self) { println!("Position is {:?}", self.position); }
    pub fn print_orientation(&self) { println!("Orientation is {:?}", self.orientation); }

    pub fn print_stats(&self) {

        self.print_position();
        self.print_orientation();
        self.print_vel();
        self.print_omega();

    }

    pub fn surface_area_estimate(&self) -> f64 {
        let (a, b, c) = (self.shape[0], self.shape[1], self.shape[2]);

        let p = 1.605;

        let numerator = (a * b).powf(p) + (a * c).powf(p) + (b * c).powf(p);
        let frac = (numerator / 3.0).powf(1.0/p);

        4.0 * PI * frac
    }

    pub fn lab_body_convert(&self, v :&Vector3<f64>) -> Vector3<f64>{

        let v_lin = v - self.position;
        let v_lin_quaternion = Quaternion::from_imag(v_lin);
        let v_body = lab_to_body(&v_lin_quaternion, &self.orientation);

        v_body.imag()

    }

    pub fn surface_splitter(&self, p0 :&Vector3<f64>) -> (usize, usize) {
        ///Decides for a point p0 (in lab coordinates) on the ellipsoid, which plane in the orientation of the body to split the surface by.
        /// outputs eg (1,0,0) for positive side of x-plane or (0, -1, 0) for negative side of y-plane.

        let p0_lin_transformed = p0 - self.position;
        let p0_quaternion = na::Quaternion::from_imag(p0_lin_transformed);
        let p0_body = lab_to_body(&p0_quaternion, &self.orientation);
        let shape = self.shape;

        // println!("Point in lab frame is {:?}", p0);
        // println!("Point in body frame is {:?}", p0_lin_transformed);
        // println!("Point rotated is {:?}", p0_body);

        let mut scaled_p0 = p0_body.imag();


        // println!("scaled p0 has real part {:?}, imag part {:?}", p0_body.w, p0_body.imag());
        // println!("scaled p0 = {:?}", scaled_p0);

        // let mut check = 0.0;
        // for i in 0..3 {
        //     check += (scaled_p0[i]/shape[i]).powi(2);
        // }
        // println!("Check = {:?}", check);

        let mut max_abs_val = scaled_p0[0];
        let mut max_abs_ind :usize = 0;
        for i in 1..3 {
            if max_abs_val.abs() < scaled_p0[i].abs() {
                max_abs_val = scaled_p0[i];
                max_abs_ind = i;
            }
        }

        // let mut result_vec = Vector3::new(0.0, 0.0, 0.0);
        // result_vec[max_abs_ind] = max_abs_val / max_abs_val.abs();

        let mut is_positive = 0_usize;
        if max_abs_val < 0.0 {
            is_positive = 1_usize
        }

        (max_abs_ind, is_positive)
    }

}