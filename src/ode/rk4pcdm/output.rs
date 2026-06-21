//! Console and CSV reporting for the impulse/PCDM integrator.
use super::{CouplingScheme, Rk4PCDM};
use crate::bem::bem_for_ode::{AngularState, LinearState};
use nalgebra::{DVector, Quaternion, Vector3};
use std::io::Write;
use std::time::Duration;

impl Rk4PCDM {
    pub(crate) fn print_progress(
        &self,
        i: usize,
        num_steps: usize,
        steady_elapsed: Duration,
        elapsed_dt: Duration,
    ) {
        let completion_percentage = 100.0 * (i as f64) / (num_steps as f64);
        let dt_sec = (elapsed_dt.as_millis() as f64) * 0.001;
        let ke_solid = self.solid_kinetic_energy(&self.state.lin, &self.state.ang);
        let ke_fluid = self.solver.fluid_kinetic_energy();
        let ke_total = ke_solid.total + ke_fluid;
        let ke_drift = self.format_ke_drift(ke_total);

        // Average over steady-state steps only: `steady_elapsed` excludes the
        // first (build-heavy) step, and `i - 1` steady steps have completed.
        let eta = if i < 2 {
            "calculating".to_string()
        } else {
            let ratio = ((num_steps - i) as f64) / ((i - 1) as f64);
            self.format_duration(self.multiply_duration(steady_elapsed, ratio))
        };

        println!(
            "Step {:>5}/{:<5} {:>5.1}% | t {:.3} | ETA {} | step {:.2}s | KE {:.4} | drift {} | fluid {:.4} | solid {:.4}",
            i,
            num_steps,
            completion_percentage,
            self.t,
            eta,
            dt_sec,
            ke_total,
            ke_drift,
            ke_fluid,
            ke_solid.total
        );
        if self.hamiltonian_coupled_solve {
            println!(
                "  Coupled residual last/max {:.3e}/{:.3e} | true dE max {:.3e} | corr rel {:.3e} ({:.3e} KE metric) | raw dP/dH max {:.3e}/{:.3e} | Jacobians {} | adaptive retries {} | max substeps {}",
                self.coupled_last_step_residual_norm,
                self.coupled_max_residual_norm,
                self.coupled_max_true_energy_err_rel,
                self.coupled_last_step_correction_rel,
                self.coupled_last_step_correction_kinetic_rel,
                self.coupled_max_raw_linear_impulse_resid,
                self.coupled_max_raw_angular_impulse_resid,
                self.coupled_jacobian_builds,
                self.hamiltonian_adaptive_retry_count,
                self.hamiltonian_max_substeps_used
            );
        }
        if self.scheme == CouplingScheme::Impulse && self.impulse_fp_steps > 0 {
            let mean_iters = self.impulse_fp_iter_sum as f64 / self.impulse_fp_steps as f64;
            println!(
                "  Impulse FP iterations last/mean/max {}/{:.2}/{}",
                self.impulse_fp_last_iter, mean_iters, self.impulse_fp_max_iter
            );
        }
        if self.scheme == CouplingScheme::Variational && self.variational_momentum_diagnostic {
            println!(
                "  Discrete momentum drift last/max {:.3e}/{:.3e}",
                self.variational_discrete_momentum_last_drift,
                self.variational_discrete_momentum_max_drift
            );
        }
        if self.impulse_variational_defect_probe {
            println!(
                "  Impulse variational defect {:.3e} | metric cos/scale {:.3e}/{:.3e} | pressure cos/scale {:.3e}/{:.3e} | pair cos/scale {:.3e}/{:.3e}",
                self.impulse_variational_defect_last_norm,
                self.impulse_variational_defect_last_metric_cos,
                self.impulse_variational_defect_last_metric_scale,
                self.impulse_variational_defect_last_pressure_cos,
                self.impulse_variational_defect_last_pressure_scale,
                self.impulse_variational_defect_last_pair_cos,
                self.impulse_variational_defect_last_pair_scale
            );
        }
    }

    pub(crate) fn print_projection_floor_event(&self, step: usize, time: f64) {
        if self.projection_last_step_floor_hits == 0
            && self.projection_last_step_floor_fallbacks == 0
        {
            return;
        }

        println!(
            "  Projection floor step {:>5} | t {:.6} | hits {} | fallbacks {} | max excess {:.6e} ({:.6e})",
            step,
            time,
            self.projection_last_step_floor_hits,
            self.projection_last_step_floor_fallbacks,
            self.projection_last_step_max_floor_rel,
            self.projection_last_step_max_floor_abs
        );
    }

    pub(crate) fn print_initial_state(&self) {
        let (pos, vel) = &self.state.lin;
        let (q, omega) = &self.state.ang;

        println!("Initial state:");
        for b in 0..self.nbody {
            let p = Vector3::new(pos[3 * b], pos[3 * b + 1], pos[3 * b + 2]);
            let v = Vector3::new(vel[3 * b], vel[3 * b + 1], vel[3 * b + 2]);
            let w = omega[b].imag();
            println!("  Body {:>2}", b + 1);
            if let Some(info) = self.body_info.get(b) {
                let shape_ratio = if info.shape[0].abs() > f64::EPSILON {
                    info.shape / info.shape[0]
                } else {
                    Vector3::repeat(f64::NAN)
                };
                println!("    density:          {:.6}", info.density);
                println!("    shape axes ratio: {}", self.format_vec3(&shape_ratio));
                println!("    initial KE ratio: {:.6}", info.initial_ke_ratio);
            }
            println!("    position:         {}", self.format_vec3(&p));
            println!("    velocity:         {}", self.format_vec3(&v));
            println!("    angular velocity: {}", self.format_vec3(&w));
            println!("    orientation:      {}", self.format_quat(&q[b]));
        }
        println!();
    }

    pub(crate) fn format_vec3(&self, v: &Vector3<f64>) -> String {
        format!("({:.6}, {:.6}, {:.6})", v[0], v[1], v[2])
    }

    pub(crate) fn format_quat(&self, q: &Quaternion<f64>) -> String {
        format!("(w={:.6}, i={:.6}, j={:.6}, k={:.6})", q.w, q.i, q.j, q.k)
    }

    pub(crate) fn format_duration(&self, duration: Duration) -> String {
        let secs = duration.as_secs();
        if secs >= 3600 {
            let hrs = secs / 3600;
            let mins = (secs - hrs * 3600) / 60;
            let secs = secs - hrs * 3600 - mins * 60;
            format!("{}h {}m {}s", hrs, mins, secs)
        } else if secs >= 60 {
            let mins = secs / 60;
            let secs = secs - mins * 60;
            format!("{}m {}s", mins, secs)
        } else {
            format!("{}s", secs)
        }
    }

    pub(crate) fn format_ke_drift(&self, ke_total: f64) -> String {
        match self.initial_total_ke {
            Some(initial_ke) if initial_ke.abs() > f64::EPSILON => {
                let drift_pct = 100.0 * (ke_total - initial_ke) / initial_ke;
                format!("{:+.2}%", drift_pct)
            }
            _ => "n/a".to_string(),
        }
    }

    pub(crate) fn capture_initial_total_ke(&mut self) {
        if self.initial_total_ke.is_none() {
            let ke_solid = self.solid_kinetic_energy(&self.state.lin, &self.state.ang);
            self.initial_total_ke = Some(ke_solid.total + self.fluid_ke_step_start);
        }
    }

    pub(crate) fn write_header<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        // Global columns first (energy-conservation monitoring), then a fixed
        // block per body.
        write!(
            writer,
            "time,ke_total,ke_fluid,ke_solid,ke_lin_solid,ke_rot_solid"
        )?;
        for b in 1..=self.nbody {
            write!(
                writer,
                ",px_{b},py_{b},pz_{b},vx_{b},vy_{b},vz_{b},q1_{b},q2_{b},q3_{b},q0_{b},w1_{b},w2_{b},w3_{b},w0_{b},ofix1_{b},ofix2_{b},ofix3_{b},ke_lin_{b},ke_rot_{b},ke_{b},lfluid_x_{b},lfluid_y_{b},lfluid_z_{b},lambdafluid_x_{b},lambdafluid_y_{b},lambdafluid_z_{b},pcon_x_{b},pcon_y_{b},pcon_z_{b},hcon_x_{b},hcon_y_{b},hcon_z_{b}",
                b = b
            )?;
        }
        write!(
            writer,
            ",impulse_global_p_drift,impulse_global_h_drift,impulse_body_p_drift_max,impulse_body_h_drift_max,impulse_pair_metric_pairs,impulse_pair_metric_load_norm,jdisc_px,jdisc_py,jdisc_pz,jdisc_hx,jdisc_hy,jdisc_hz,jdisc_drift,impulse_var_defect_norm,impulse_var_metric_cos,impulse_var_metric_scale,impulse_var_pressure_cos,impulse_var_pressure_scale,impulse_var_pair_cos,impulse_var_pair_scale"
        )?;
        writeln!(writer)
    }

    pub(crate) fn write_row<W: Write>(
        &self,
        writer: &mut W,
        time: f64,
        x: &LinearState,
        o: &AngularState,
        o_lab: &DVector<f64>,
        ke_fluid: f64,
    ) -> std::io::Result<()> {
        let ke = self.solid_kinetic_energy(x, o);
        let ke_total = ke.total + ke_fluid;
        let impulse_diagnostics = self.impulse_diagnostics(x, o);

        write!(
            writer,
            "{}, {}, {}, {}, {}, {}",
            time, ke_total, ke_fluid, ke.total, ke.total_lin, ke.total_rot
        )?;

        let (pos, vel) = x;
        let (q, omega) = o;
        for b in 0..self.nbody {
            // position, velocity
            for c in 0..3 {
                write!(writer, ", {}", pos[3 * b + c])?;
            }
            for c in 0..3 {
                write!(writer, ", {}", vel[3 * b + c])?;
            }
            // orientation quaternion (i, j, k, w)
            for val in q[b].as_vector().iter() {
                write!(writer, ", {}", val)?;
            }
            // angular velocity quaternion (i, j, k, w)
            for val in omega[b].as_vector().iter() {
                write!(writer, ", {}", val)?;
            }
            // marker point
            for c in 0..3 {
                write!(writer, ", {}", o_lab[3 * b + c])?;
            }
            // per-body energies
            let (ke_lin, ke_rot, ke_b) = ke.per_body[b];
            write!(writer, ", {}, {}, {}", ke_lin, ke_rot, ke_b)?;
            let (l_fluid, lambda_fluid, p_con, h_con) = impulse_diagnostics
                .as_ref()
                .map(|d| d[b])
                .unwrap_or_else(|| {
                    (
                        Vector3::repeat(f64::NAN),
                        Vector3::repeat(f64::NAN),
                        Vector3::repeat(f64::NAN),
                        Vector3::repeat(f64::NAN),
                    )
                });
            for val in l_fluid.iter() {
                write!(writer, ", {}", val)?;
            }
            for val in lambda_fluid.iter() {
                write!(writer, ", {}", val)?;
            }
            for val in p_con.iter() {
                write!(writer, ", {}", val)?;
            }
            for val in h_con.iter() {
                write!(writer, ", {}", val)?;
            }
        }
        write!(
            writer,
            ", {}, {}, {}, {}",
            self.impulse_global_p_drift_last,
            self.impulse_global_h_drift_last,
            self.impulse_body_p_drift_max_last,
            self.impulse_body_h_drift_max_last
        )?;
        write!(
            writer,
            ", {}, {}",
            self.impulse_pair_metric_last_pairs, self.impulse_pair_metric_last_norm
        )?;
        if let Some(momentum) = &self.variational_discrete_momentum_out {
            for c in 0..6 {
                write!(writer, ", {}", momentum[c])?;
            }
            write!(
                writer,
                ", {}",
                self.variational_discrete_momentum_last_drift
            )?;
        } else {
            write!(writer, ", NaN, NaN, NaN, NaN, NaN, NaN, NaN")?;
        }
        if self.impulse_variational_defect_out {
            write!(
                writer,
                ", {}, {}, {}, {}, {}, {}, {}",
                self.impulse_variational_defect_last_norm,
                self.impulse_variational_defect_last_metric_cos,
                self.impulse_variational_defect_last_metric_scale,
                self.impulse_variational_defect_last_pressure_cos,
                self.impulse_variational_defect_last_pressure_scale,
                self.impulse_variational_defect_last_pair_cos,
                self.impulse_variational_defect_last_pair_scale
            )?;
        } else {
            write!(writer, ", NaN, NaN, NaN, NaN, NaN, NaN, NaN")?;
        }
        writeln!(writer)
    }
}
