//! Energy and impulse diagnostics for the integrator.
#![allow(unused_imports)]
use super::{Rk4PCDM, SolidEnergy};
use crate::bem::bem_for_ode::{AngularState, LinearState};
use crate::math::rotation;
use crate::ode::dop_shared::{System2, System4};
use nalgebra::{Matrix3, Quaternion, Vector3};

impl<F, G, I> Rk4PCDM<F, G, I>
where
    F: System2<LinearState>,
    G: System2<AngularState>,
    I: System4<LinearState>,
{
    pub(crate) fn impulse_diagnostics(
        &self,
        x: &LinearState,
        o: &AngularState,
    ) -> Option<Vec<(Vector3<f64>, Vector3<f64>, Vector3<f64>, Vector3<f64>)>> {
        let l_lin = self.fluid_impulse_lin_step_start.as_ref()?;
        let l_ang = self.fluid_impulse_ang_step_start.as_ref()?;
        let (pos, vel) = x;
        let (q, omega_lab) = o;

        let mut out = Vec::with_capacity(self.nbody);
        for b in 0..self.nbody {
            let x_b = Vector3::new(pos[3 * b], pos[3 * b + 1], pos[3 * b + 2]);
            let v_b = Vector3::new(vel[3 * b], vel[3 * b + 1], vel[3 * b + 2]);
            let l_fluid = Vector3::new(l_lin[3 * b], l_lin[3 * b + 1], l_lin[3 * b + 2]);
            let lambda_fluid = Vector3::new(l_ang[3 * b], l_ang[3 * b + 1], l_ang[3 * b + 2]);
            let omega_body = rotation::lab_to_body(&omega_lab[b], &q[b]).imag();
            let h_body = self.inertias[b] * omega_body;
            let h_solid = rotation::body_to_lab(&Quaternion::from_imag(h_body), &q[b])
                .imag();
            let p_con = self.masses[b] * v_b - l_fluid;
            let h_con = x_b.cross(&p_con) + h_solid - lambda_fluid;
            out.push((l_fluid, lambda_fluid, p_con, h_con));
        }
        Some(out)
    }

    pub(crate) fn solid_kinetic_energy(&self, x: &LinearState, o: &AngularState) -> SolidEnergy {
        let (_, vel) = x;
        let (q, omega_lab) = o;

        let mut total_lin = 0.0;
        let mut total_rot = 0.0;
        let mut per_body = Vec::with_capacity(self.nbody);

        for i in 0..self.nbody {
            let v = Vector3::new(vel[3 * i], vel[3 * i + 1], vel[3 * i + 2]);
            let mass = self.masses[i];

            let ke_lin = if mass > 0.0 {
                0.5 * mass * v.dot(&v)
            } else {
                0.0
            };

            let omega_b = rotation::lab_to_body(&omega_lab[i], &q[i]).imag();
            let ke_rot = if mass > 0.0 {
                0.5 * omega_b.dot(&(self.inertias[i] * omega_b))
            } else {
                0.0
            };

            total_lin += ke_lin;
            total_rot += ke_rot;
            per_body.push((ke_lin, ke_rot, ke_lin + ke_rot));
        }

        SolidEnergy {
            total_lin,
            total_rot,
            total: total_lin + total_rot,
            per_body,
        }
    }

    pub(crate) fn fluid_kinetic_energy(&self) -> f64 {
        self.fluid_ke_getter
            .as_ref()
            .map(|get| get())
            .unwrap_or(0.0)
    }

}
