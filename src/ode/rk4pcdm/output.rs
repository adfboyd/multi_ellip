//! Console and CSV reporting for the impulse/PCDM integrator.
#![allow(unused_imports)]
use super::{Rk4PCDM, SolidEnergy};
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
        let ke_solid = self.solid_kinetic_energy(&self.x, &self.o);
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
    }

    pub(crate) fn print_initial_state(&self) {
        let (pos, vel) = &self.x;
        let (q, omega) = &self.o;

        println!("Initial state:");
        for b in 0..self.nbody {
            let p = Vector3::new(pos[3 * b], pos[3 * b + 1], pos[3 * b + 2]);
            let v = Vector3::new(vel[3 * b], vel[3 * b + 1], vel[3 * b + 2]);
            let w = omega[b].imag();
            println!("  Body {:>2}", b + 1);
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
            let ke_solid = self.solid_kinetic_energy(&self.x, &self.o);
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
        writeln!(writer)
    }

}
