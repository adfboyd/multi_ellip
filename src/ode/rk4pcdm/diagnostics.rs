//! Energy and impulse diagnostics for the integrator.
use super::{ImpulsePartitionState, Rk4PCDM, SolidEnergy};
use crate::bem::bem_for_ode::{AngularState, LinearState};
use crate::math::rotation;
use nalgebra::{Quaternion, Vector3};

impl Rk4PCDM {
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
            let h_solid = rotation::body_to_lab(&Quaternion::from_imag(h_body), &q[b]).imag();
            let p_con = self.masses[b] * v_b - l_fluid;
            let h_con = x_b.cross(&p_con) + h_solid - lambda_fluid;
            out.push((l_fluid, lambda_fluid, p_con, h_con));
        }
        Some(out)
    }

    pub(crate) fn record_impulse_partition_drift(
        &mut self,
        x: &LinearState,
        o: &AngularState,
        l_lin: &nalgebra::DVector<f64>,
        l_ang: &nalgebra::DVector<f64>,
    ) {
        let current = self.impulse_partition_state(x, o, l_lin, l_ang);
        let initial = self
            .impulse_partition_initial
            .get_or_insert_with(|| current.clone());

        self.impulse_global_p_drift_last =
            Self::relative_or_absolute_drift(&current.global_p, &initial.global_p);
        self.impulse_global_h_drift_last =
            Self::relative_or_absolute_drift(&current.global_h, &initial.global_h);

        let mut body_p_max = 0.0_f64;
        let mut body_h_max = 0.0_f64;
        for (now, start) in current.per_body_p.iter().zip(initial.per_body_p.iter()) {
            body_p_max = body_p_max.max(Self::relative_or_absolute_drift(now, start));
        }
        for (now, start) in current.per_body_h.iter().zip(initial.per_body_h.iter()) {
            body_h_max = body_h_max.max(Self::relative_or_absolute_drift(now, start));
        }

        self.impulse_body_p_drift_max_last = body_p_max;
        self.impulse_body_h_drift_max_last = body_h_max;
        if self.impulse_global_p_drift_last.is_finite() {
            self.impulse_global_p_drift_max = self
                .impulse_global_p_drift_max
                .max(self.impulse_global_p_drift_last);
        }
        if self.impulse_global_h_drift_last.is_finite() {
            self.impulse_global_h_drift_max = self
                .impulse_global_h_drift_max
                .max(self.impulse_global_h_drift_last);
        }
        if body_p_max.is_finite() {
            self.impulse_body_p_drift_max = self.impulse_body_p_drift_max.max(body_p_max);
        }
        if body_h_max.is_finite() {
            self.impulse_body_h_drift_max = self.impulse_body_h_drift_max.max(body_h_max);
        }
    }

    fn impulse_partition_state(
        &self,
        x: &LinearState,
        o: &AngularState,
        l_lin: &nalgebra::DVector<f64>,
        l_ang: &nalgebra::DVector<f64>,
    ) -> ImpulsePartitionState {
        let (pos, vel) = x;
        let (q, omega_lab) = o;
        let mut per_body_p = Vec::with_capacity(self.nbody);
        let mut per_body_h = Vec::with_capacity(self.nbody);
        let mut global_p = Vector3::zeros();
        let mut global_h = Vector3::zeros();

        for b in 0..self.nbody {
            let x_b = Vector3::new(pos[3 * b], pos[3 * b + 1], pos[3 * b + 2]);
            let v_b = Vector3::new(vel[3 * b], vel[3 * b + 1], vel[3 * b + 2]);
            let l_b = Vector3::new(l_lin[3 * b], l_lin[3 * b + 1], l_lin[3 * b + 2]);
            let lambda_b = Vector3::new(l_ang[3 * b], l_ang[3 * b + 1], l_ang[3 * b + 2]);
            let p_con = self.masses[b] * v_b - l_b;
            let omega_body = rotation::lab_to_body(&omega_lab[b], &q[b]).imag();
            let h_body = self.inertias[b] * omega_body;
            let h_solid = rotation::body_to_lab(&Quaternion::from_imag(h_body), &q[b]).imag();
            let h_con = x_b.cross(&p_con) + h_solid - lambda_b;

            global_p += p_con;
            global_h += h_con;
            per_body_p.push(p_con);
            per_body_h.push(h_con);
        }

        ImpulsePartitionState {
            per_body_p,
            per_body_h,
            global_p,
            global_h,
        }
    }

    fn relative_or_absolute_drift(current: &Vector3<f64>, initial: &Vector3<f64>) -> f64 {
        let absolute = (current - initial).norm();
        let scale = initial.norm();
        if scale > 1.0e-14 {
            absolute / scale
        } else {
            absolute
        }
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
}
