use nalgebra::{DVector, Matrix3, Quaternion, Vector3};
use std::io::Write;
use std::time::{Duration, Instant};
use crate::ode::dop_shared::{IntegrationError, Stats, System2, System4};
use crate::ode::pcdm::accel_get;
use crate::bem::bem_for_ode::{AngularState, LinearState};
use crate::system::system::BOOTSTRAP_PASSES;

pub struct Rk4PCDM<F, G, I>
    where
        F: System2<LinearState>,
        G: System2<AngularState>,
        I: System4<LinearState>,
{
    f: F,
    g: G,
    i: I,
    t: f64,
    x: LinearState,
    o_lab: DVector<f64>,
    o: AngularState,
    nbody: usize,
    inertias: Vec<Matrix3<f64>>,
    masses: Vec<f64>,
    /// Per-body body-frame (diagonal) added-mass tensors, scaled by the safety
    /// factor: M_a_safe = SAFETY * M_a. Used by the semi-implicit stabiliser.
    added_mass_safe: Vec<Matrix3<f64>>,
    /// Enable the semi-implicit added-mass-partitioned (Robin) velocity update.
    added_mass_stab: bool,
    /// Previous accepted linear acceleration per body (lab frame), the `a_prev`
    /// of the single-step Robin form. Zero on the first step.
    lin_accel_prev: DVector<f64>,
    fluid_ke_getter: Option<Box<dyn Fn() -> f64 + Send + Sync>>,
    t_begin: f64,
    t_end: f64,
    step_size: f64,
    half_step: f64,
    quarter_step: f64,
    pub samp_rate: u32,
    pub print_rate: u32,
    pub t_out: Vec<f64>,
    pub x_out: Vec<LinearState>,
    pub o_out: Vec<AngularState>,
    pub o_lab_out: Vec<DVector<f64>>,
    stats: Stats,
    /// Timing of the most recent integration, for the end-of-run summary.
    pub run_wall_secs: f64,
    pub run_first_step_secs: f64,
    pub run_steady_per_step: f64,
    /// Fluid KE captured at the start of the current step (stage-A force call,
    /// evaluated at exactly z_n), used to write the row sampled at time t_n.
    fluid_ke_step_start: f64,
}

/// Solid-body kinetic energy breakdown for one timestep.
struct SolidEnergy {
    total_lin: f64,
    total_rot: f64,
    total: f64,
    /// (linear, rotational, total) kinetic energy for each body.
    per_body: Vec<(f64, f64, f64)>,
}

impl<F, G, I> Rk4PCDM<F, G, I>
    where
        F: System2<LinearState>,
        G: System2<AngularState>,
        I: System4<LinearState>,
{
    //Function for creating new solver
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        f: F,
        g: G,
        i: I,
        t_begin: f64,
        x: LinearState, // (positions, velocities) stacked over bodies
        orientations: Vec<(Quaternion<f64>, Quaternion<f64>)>, // (orientation, angular velocity) per body
        inertias: Vec<Matrix3<f64>>,
        masses: Vec<f64>,
        added_mass_tensors: Vec<Matrix3<f64>>,
        added_mass_stab: bool,
        fluid_ke_getter: Option<Box<dyn Fn() -> f64 + Send + Sync>>,
        t_end: f64,
        step_size: f64,
        samp_rate: u32,
        print_rate: u32,
    ) -> Self {
        let nbody = orientations.len();
        let q: Vec<Quaternion<f64>> = orientations.iter().map(|o| o.0).collect();
        let omega: Vec<Quaternion<f64>> = orientations.iter().map(|o| o.1).collect();

        // added_mass_tensors are already scaled by the safety factor in main
        // (input key added_mass_safety). M_a only needs to be >= the true
        // effective added mass for stability of the Robin update and does not
        // change the fixed point (a_stab = a_expl).
        let added_mass_safe: Vec<Matrix3<f64>> = added_mass_tensors;

        Rk4PCDM {
            f,
            g,
            i,
            t: t_begin,
            x,
            o: (q, omega),
            o_lab: DVector::zeros(3 * nbody),
            nbody,
            inertias,
            masses,
            added_mass_safe,
            added_mass_stab,
            lin_accel_prev: DVector::zeros(3 * nbody),
            fluid_ke_getter,
            t_begin,
            t_end,
            step_size,
            half_step: step_size * 0.5,
            quarter_step: step_size * 0.25,
            samp_rate,
            print_rate,
            t_out: Vec::new(),
            x_out: Vec::new(),
            o_out: Vec::new(),
            o_lab_out: vec![],
            stats: Stats::new(),
            run_wall_secs: 0.0,
            run_first_step_secs: 0.0,
            run_steady_per_step: 0.0,
            fluid_ke_step_start: 0.0,
        }
    }

    pub fn integrate(&mut self) -> Result<Stats, IntegrationError> {
        self.t_out.push(self.t);
        self.x_out.push(self.x.clone());
        println!("Initial positions and velocities = {:?}", &self.x);
        self.o_out.push(self.o.clone());
        println!("Initial orientations and angular velocities = {:?}", &self.o);
        self.o_lab = self.orientation_to_marker_point();
        self.o_lab_out.push(self.o_lab.clone());

        let num_steps = ((self.t_end - self.t_begin) / self.step_size).ceil() as usize;
        let samp_rate = self.samp_rate as usize;
        let print_rate = self.print_rate as usize;

        let start_t = Instant::now();
        // Timer for the ETA average, reset after the first (build-heavy) step.
        let mut steady_start = start_t;
        let mut start_dt = Instant::now();
        for i in 0..num_steps {
            if (i % print_rate == 0) & (i > 0) {
                self.print_progress(i, num_steps, steady_start.elapsed(), start_dt.elapsed());
            };
            start_dt = Instant::now();

            if i == 0 {
                self.bootstrap_first_step();
            }
            self.advance_one_step();
            if i == 0 {
                steady_start = Instant::now();
            }
            let t_new = self.t + self.step_size;

            if i % samp_rate == 0 {
                self.t_out.push(t_new);
                self.x_out.push(self.x.clone());
                self.o_out.push(self.o.clone());
                self.o_lab_out.push(self.o_lab.clone());
            }

            self.t = t_new;
            self.stats.accepted_steps += 1;
        }
        Ok(self.stats)
    }

    pub fn integrate_with_writer<W: Write>(&mut self, writer: &mut W) -> Result<Stats, IntegrationError> {
        self.t_out.push(self.t);
        self.x_out.push(self.x.clone());
        println!("Initial positions and velocities = {:?}", &self.x);
        self.o_out.push(self.o.clone());
        println!("Initial orientations and angular velocities = {:?}", &self.o);
        self.o_lab = self.orientation_to_marker_point();
        self.o_lab_out.push(self.o_lab.clone());

        self.write_header(writer)?;
        let x0 = self.x.clone();
        let o0 = self.o.clone();
        let olab0 = self.o_lab.clone();
        // Defer writing each sampled row until the next step has run, so its
        // fluid KE can be taken from that step's stage-A solve (evaluated at the
        // row's own state) rather than the dt/2-stale half-step value.
        let mut pending: Option<(f64, LinearState, AngularState, DVector<f64>)> =
            Some((self.t, x0, o0, olab0));

        let num_steps = ((self.t_end - self.t_begin) / self.step_size).ceil() as usize;
        let samp_rate = self.samp_rate as usize;
        let print_rate = self.print_rate as usize;

        let start_t = Instant::now();
        // Timer for the ETA average, reset after the first (build-heavy) step.
        let mut steady_start = start_t;
        let mut start_dt = Instant::now();
        for i in 0..num_steps {
            if (i % print_rate == 0) & (i > 0) {
                self.print_progress(i, num_steps, steady_start.elapsed(), start_dt.elapsed());
            };
            start_dt = Instant::now();

            if i == 0 {
                self.bootstrap_first_step();
            }
            self.advance_one_step();
            if i == 0 {
                steady_start = Instant::now();
            }
            let t_new = self.t + self.step_size;

            // Flush the pending row now: its state is the start state of the step
            // just run, so `fluid_ke_step_start` (captured in stage A) is the
            // fluid KE at exactly that row's state.
            if let Some((tp, xp, op, olp)) = pending.take() {
                self.write_row(writer, tp, &xp, &op, &olp, self.fluid_ke_step_start)?;
            }

            if i % samp_rate == 0 {
                self.t_out.push(t_new);
                self.x_out.push(self.x.clone());
                self.o_out.push(self.o.clone());
                self.o_lab_out.push(self.o_lab.clone());
                pending = Some((t_new, self.x.clone(), self.o.clone(), self.o_lab.clone()));
            }

            self.t = t_new;
            self.stats.accepted_steps += 1;
        }

        // Flush the final pending row. Its state is the end state, whose fluid KE
        // has not yet been computed, so run one extra BEM solve to obtain it (the
        // side effects — φ history push, fluid KE update — are harmless).
        if let Some((tp, xp, op, olp)) = pending.take() {
            let _ = self.force_get();
            let kef = self.fluid_kinetic_energy();
            self.write_row(writer, tp, &xp, &op, &olp, kef)?;
        }

        // Record timing for the end-of-run summary.
        self.run_wall_secs = start_t.elapsed().as_secs_f64();
        self.run_first_step_secs = steady_start.duration_since(start_t).as_secs_f64();
        let steady_steps = num_steps.saturating_sub(1).max(1) as f64;
        self.run_steady_per_step = steady_start.elapsed().as_secs_f64() / steady_steps;

        Ok(self.stats)
    }

    /// Bootstrap the ∂φ/∂t history at t = 0. The dphi force needs temporal φ
    /// history that doesn't exist at startup (initial acceleration and φ̇ are
    /// mutually dependent through the added mass), so run the first step
    /// provisionally [`BOOTSTRAP_PASSES`] times, rewinding the state after each
    /// pass. Each pass refines the seed φ̇ by a geometric (added-mass
    /// contraction) factor; the bookkeeping on the force side lives in
    /// `ForceCalculate` via `Simulation::bootstrap_redos`.
    fn bootstrap_first_step(&mut self) {
        let x0 = self.x.clone();
        let o0 = self.o.clone();
        for _ in 0..BOOTSTRAP_PASSES {
            self.advance_one_step();
            // Rewind to the initial state and push it back into the system.
            self.x = x0.clone();
            self.o = o0.clone();
            let _ = self.f.system(0.0, &self.x);
            let o_push = self.o.clone();
            let _ = self.g.system(0.0, &o_push);
        }
    }

    /// Predictor-corrector (Verlet-like translation + PCDM rotation) update of the
    /// full state `(self.x, self.o, self.o_lab)` by one timestep.
    fn advance_one_step(&mut self) {
        // Forces at the start of the step.
        let (linear_accel_expl, angular_force) = self.force_get();
        // Stage A is evaluated at exactly the step-start state z_n, so its fluid
        // KE is the correct value to write for the row sampled at time t_n.
        self.fluid_ke_step_start = self.fluid_kinetic_energy();

        // Optional semi-implicit added-mass stabilisation (single-step Robin):
        // a_stab = (M_s I + M_a_safe)^{-1}(M_s a_expl + M_a_safe a_prev), with
        // a_prev the accepted acceleration from the previous step. Answer-
        // preserving (fixed point a_stab = a_prev = a_expl) and unconditionally
        // stable for M_a_safe >= effective added mass. Applied to both stages.
        let linear_accel = if self.added_mass_stab {
            self.stabilise_lin_accel(&linear_accel_expl)
        } else {
            linear_accel_expl.clone()
        };

        // Half-step prediction.
        let (p_half, v_half) = self.lin_half_step(&linear_accel);
        let (q_half, o_half) = self.ang_half_step(&angular_force);

        let _ = self.f.system(0.0, &(p_half, v_half.clone()));
        let _ = self.g.system(0.0, &(q_half.clone(), o_half.clone()));

        // Forces at the half step.
        let (linear_force_half_expl, angular_force_half) = self.force_get();

        let linear_force_half = if self.added_mass_stab {
            self.stabilise_lin_accel(&linear_force_half_expl)
        } else {
            linear_force_half_expl
        };

        let x_new = self.lin_full_step(&linear_force_half, &v_half);
        let o_new = self.ang_full_step(&angular_force_half, &(q_half, o_half));

        let _ = self.f.system(0.0, &x_new);
        let _ = self.g.system(0.0, &o_new);

        let o_lab_new = self.orientation_to_marker_point();

        self.x = x_new;
        self.o = o_new;
        self.o_lab = o_lab_new;

        // Carry the stage-A explicit acceleration as a_prev for the next step's
        // Robin update. At steady state a_prev -> a_expl, so the stabiliser is
        // answer-preserving.
        if self.added_mass_stab {
            self.lin_accel_prev = linear_accel_expl;
        }
    }

    /// Single-step semi-implicit (Robin) added-mass stabilisation of the
    /// lab-frame linear acceleration. Per body: rotate a_expl and a_prev into
    /// the body frame (where M_a is diagonal/constant), solve
    /// a_stab = (M_s I + M_a_safe)^{-1}(M_s a_expl + M_a_safe a_prev), and
    /// rotate back to the lab frame.
    fn stabilise_lin_accel(&self, accel_expl: &DVector<f64>) -> DVector<f64> {
        let (q, _) = &self.o;
        let mut out = DVector::zeros(3 * self.nbody);
        for i in 0..self.nbody {
            let ms = self.masses[i];
            // Bodies with zero mass (fixed) are left untouched.
            if ms <= 0.0 {
                for c in 0..3 {
                    out[3 * i + c] = accel_expl[3 * i + c];
                }
                continue;
            }
            let a_expl_lab = Vector3::new(
                accel_expl[3 * i],
                accel_expl[3 * i + 1],
                accel_expl[3 * i + 2],
            );
            let a_prev_lab = Vector3::new(
                self.lin_accel_prev[3 * i],
                self.lin_accel_prev[3 * i + 1],
                self.lin_accel_prev[3 * i + 2],
            );

            // Rotate lab -> body using the body orientation quaternion.
            let a_expl_b = self.lab_to_body(&Quaternion::from_imag(a_expl_lab), &q[i]).imag();
            let a_prev_b = self.lab_to_body(&Quaternion::from_imag(a_prev_lab), &q[i]).imag();

            let ms_i = Matrix3::identity() * ms;
            let m_a = self.added_mass_safe[i];
            let lhs = ms_i + m_a;
            let rhs = ms_i * a_expl_b + m_a * a_prev_b;
            let a_stab_b = lhs
                .try_inverse()
                .map(|inv| inv * rhs)
                .unwrap_or(a_expl_b);

            // Rotate body -> lab.
            let a_stab_lab = self.body_to_lab(&Quaternion::from_imag(a_stab_b), &q[i]).imag();
            for c in 0..3 {
                out[3 * i + c] = a_stab_lab[c];
            }
        }
        out
    }

    fn print_progress(&self, i: usize, num_steps: usize, steady_elapsed: Duration, elapsed_dt: Duration) {
        let completion_percentage = 100.0 * (i as f64) / (num_steps as f64);
        let dt_sec = (elapsed_dt.as_millis() as f64) * 0.001;

        // Average over steady-state steps only: `steady_elapsed` excludes the
        // first (build-heavy) step, and `i - 1` steady steps have completed.
        if i < 2 {
            println!("Time = {:.7}. Timestep {:?}/{:?}. Estimating time remaining... - {:.3}% complete. Time for this timestep = {:.3}s.", self.t, i, num_steps, completion_percentage, dt_sec);
            return;
        }
        let ratio = ((num_steps - i) as f64) / ((i - 1) as f64);
        let remain_est = self.multiply_duration(steady_elapsed, ratio);
        if remain_est.as_secs() >= 3600 {
            let hrs = remain_est.as_secs() / 3600;
            let mins = (remain_est.as_secs() - hrs * 3600) / 60;
            let secs = remain_est.as_secs() - hrs * 3600 - mins * 60;
            println!("Time = {:.7}. Timestep {:?}/{:?}. Estimated time remaining = {:?}hrs {:?}min {:?}sec - {:.3}% complete. Time for this timestep = {:.3}s.", self.t, i, num_steps, hrs, mins, secs, completion_percentage, dt_sec);
        } else if remain_est.as_secs() >= 60 {
            let mins = remain_est.as_secs() / 60;
            let secs = remain_est.as_secs() - mins * 60;
            println!("Time = {:.7}. Timestep {:?}/{:?}. Estimated time remaining = {:?}min {:?}sec - {:.3}% complete. Time for this timestep = {:.3}s.", self.t, i, num_steps, mins, secs, completion_percentage, dt_sec);
        } else {
            let secs = remain_est.as_secs();
            println!("Time = {:.7}. Timestep {:?}/{:?}. Estimated time remaining = {:?}sec - {:.3}% complete. Time for this timestep = {:.3}s.", self.t, i, num_steps, secs, completion_percentage, dt_sec);
        }
    }

    fn write_header<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        // Global columns first (energy-conservation monitoring), then a fixed
        // 20-column block per body.
        write!(writer, "time,ke_total,ke_fluid,ke_solid,ke_lin_solid,ke_rot_solid")?;
        for b in 1..=self.nbody {
            write!(
                writer,
                ",px_{b},py_{b},pz_{b},vx_{b},vy_{b},vz_{b},q1_{b},q2_{b},q3_{b},q0_{b},w1_{b},w2_{b},w3_{b},w0_{b},ofix1_{b},ofix2_{b},ofix3_{b},ke_lin_{b},ke_rot_{b},ke_{b}",
                b = b
            )?;
        }
        writeln!(writer)
    }

    fn write_row<W: Write>(
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
        }
        writeln!(writer)
    }

    fn solid_kinetic_energy(&self, x: &LinearState, o: &AngularState) -> SolidEnergy {
        let (_, vel) = x;
        let (q, omega_lab) = o;

        let mut total_lin = 0.0;
        let mut total_rot = 0.0;
        let mut per_body = Vec::with_capacity(self.nbody);

        for i in 0..self.nbody {
            let v = Vector3::new(vel[3 * i], vel[3 * i + 1], vel[3 * i + 2]);
            let mass = self.masses[i];

            let ke_lin = if mass > 0.0 { 0.5 * mass * v.dot(&v) } else { 0.0 };

            let omega_b = self.lab_to_body(&omega_lab[i], &q[i]).imag();
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

    fn fluid_kinetic_energy(&self) -> f64 {
        self.fluid_ke_getter
            .as_ref()
            .map(|get| get())
            .unwrap_or(0.0)
    }

    fn force_get(&mut self) -> LinearState {
        self.i.system()
    }

    fn lin_half_step(&mut self, lin_force: &DVector<f64>) -> LinearState {
        let (p, v) = self.x.clone();
        let p_new = &p + &v * self.half_step;
        let v_new = &v + lin_force * self.half_step;
        (p_new, v_new)
    }

    fn lin_full_step(&mut self, lin_force: &DVector<f64>, v_half: &DVector<f64>) -> LinearState {
        let (p, v) = self.x.clone();
        // Position uses the midpoint velocity v_{n+1/2} (previously v_n — explicit
        // Euler for position, the source of a small O(dt) global position error).
        let p_new = &p + v_half * self.step_size;
        let v_new = &v + lin_force * self.step_size;
        (p_new, v_new)
    }

    fn ang_half_step(&mut self, ang: &DVector<f64>) -> AngularState {
        let (q, omega_lab) = self.o.clone();

        let mut q_new = Vec::with_capacity(self.nbody);
        let mut o_new = Vec::with_capacity(self.nbody);

        for i in 0..self.nbody {
            let qi = q[i];
            let inertia = self.inertias[i];

            let omega_b = self.lab_to_body(&omega_lab[i], &qi);

            let torque_lab = Quaternion::new(0.0, ang[3 * i], ang[3 * i + 1], ang[3 * i + 2]);
            let torque = self.lab_to_body(&torque_lab, &qi);

            let ang_accel_b = accel_get(&omega_b, &inertia, &torque);

            let omega_n_quarter_b = self.omega_stepper(&omega_b, &ang_accel_b, self.quarter_step);
            let omega_n_half_b = self.omega_stepper(&omega_b, &ang_accel_b, self.half_step);

            let omega_n_quarter = self.body_to_lab(&omega_n_quarter_b, &qi);
            let qi_half_predict = self.orientation_stepper(&qi, &omega_n_quarter, self.half_step);

            let omega_n_half_lab = self.body_to_lab(&omega_n_half_b, &qi_half_predict);

            q_new.push(qi_half_predict);
            o_new.push(omega_n_half_lab);
        }

        (q_new, o_new)
    }

    fn ang_full_step(&mut self, ang: &DVector<f64>, half_qo: &AngularState) -> AngularState {
        let (q, omega_lab) = self.o.clone();
        let (q_half, o_half) = half_qo;

        let mut q_full_v = Vec::with_capacity(self.nbody);
        let mut omega_v = Vec::with_capacity(self.nbody);

        for i in 0..self.nbody {
            let qi = q[i];
            let qi_half = q_half[i];
            let inertia = self.inertias[i];

            let omega_b = self.lab_to_body(&omega_lab[i], &qi);
            let omega_n_half_b = self.lab_to_body(&o_half[i], &qi_half);

            let torque_lab = Quaternion::new(0.0, ang[3 * i], ang[3 * i + 1], ang[3 * i + 2]);
            let torque = self.lab_to_body(&torque_lab, &qi_half);

            let ang_accel_half_b = accel_get(&omega_n_half_b, &inertia, &torque);
            let omega_n_half = self.body_to_lab(&omega_n_half_b, &qi_half);

            let qi_full = self.orientation_stepper(&qi, &omega_n_half, self.step_size);

            let omega_b_full = self.omega_stepper(&omega_b, &ang_accel_half_b, self.step_size);
            let omega_i = self.body_to_lab(&omega_b_full, &qi_full);

            q_full_v.push(qi_full);
            omega_v.push(omega_i);
        }

        self.stats.num_eval += self.nbody as u32;

        (q_full_v, omega_v)
    }

    fn multiply_duration(&self, duration: Duration, factor: f64) -> Duration {
        let seconds = duration.as_secs() as f64 + f64::from(duration.subsec_nanos()) / 1_000_000_000.0;
        let result_seconds = seconds * factor;
        let whole_seconds = result_seconds as u64;
        let fractional_seconds = ((result_seconds - whole_seconds as f64) * 1_000_000_000.0) as u32;
        Duration::new(whole_seconds, fractional_seconds)
    }

    /// Body-frame x-axis marker direction in the lab frame, for each body (3*N vector).
    fn orientation_to_marker_point(&self) -> DVector<f64> {
        let (q, _) = &self.o;
        let mut out = DVector::zeros(3 * self.nbody);
        for i in 0..self.nbody {
            let marker = self.body_to_lab(&Quaternion::from_imag(Vector3::new(1.0, 0.0, 0.0)), &q[i]);
            let v = marker.vector();
            out[3 * i] = v[0];
            out[3 * i + 1] = v[1];
            out[3 * i + 2] = v[2];
        }
        out
    }

    //Converts a (pure) quaternion p_space from lab space to body space for a body of orientation q.
    fn lab_to_body(&self, p_space: &Quaternion<f64>, q: &Quaternion<f64>) -> Quaternion<f64> {
        let q_inv = q.try_inverse().unwrap();
        q_inv * (p_space * q)
    }

    //Converts a (pure) quaternion p_body from body space to lab space for a body of orientation q.
    fn body_to_lab(&self, p_body: &Quaternion<f64>, q: &Quaternion<f64>) -> Quaternion<f64> {
        let q_inv = if q.norm() > 0.00001 {
            q.try_inverse().unwrap()
        } else {
            Quaternion::from_real(0.0)
        };
        q * (p_body * q_inv)
    }

    //Steps forward the rotational velocity omega_n according to a given angular acceleration/torque.
    fn omega_stepper(&self, omega_n: &Quaternion<f64>, ang_accel: &Quaternion<f64>, dt: f64) -> Quaternion<f64> {
        omega_n + ang_accel * dt
    }

    //Steps forward the orientation of a body given initial orientation q1 and rotational velocity omega.
    fn orientation_stepper(&self, q1: &Quaternion<f64>, omega: &Quaternion<f64>, dt: f64) -> Quaternion<f64> {
        let mag = omega.norm();
        let real_part = (mag * dt * 0.5).cos();
        let imag_scalar = if mag > 0.0000001 {
            (mag * dt * 0.5).sin() / mag
        } else {
            0.0
        };
        let imag_part = imag_scalar * omega.vector();
        let omega_n1 = Quaternion::from_parts(real_part, imag_part);
        omega_n1 * q1
    }
}