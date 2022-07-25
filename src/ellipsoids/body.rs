use csv::*;
use nalgebra as na;
use serde::Serialize;

use crate::ellipsoids::state::State;

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
        // let ang_vel = ang_vel_q.vector();
        let abs_ang_vel = ang_vel_q.i.abs();
        abs_ang_vel / (2.0 * std::f64::consts::PI)
    }


    pub(crate) fn kinetic_energy(&self) -> f64 {
        let lin_mom = self.linear_momentum;
        let ang_mom = self.angular_momentum.imag();
        let ang_vel_q = self.angular_velocity();
        let ang_vel = ang_vel_q.vector();

        lin_mom.dot(&lin_mom) / (2.0 * self.mass()) + 0.5 * ang_vel.dot(&ang_mom)
    }

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
}