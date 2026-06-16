use crate::bem::bem_for_ode::{AngularState, BemSolver, BodyState, LinearState};
use crate::math::anderson::anderson_next;
use crate::math::rotation;
use crate::ode::dop_shared::{IntegrationError, Stats};
use crate::system::system::BOOTSTRAP_PASSES;
use nalgebra::{DVector, Matrix3, Quaternion, Vector3};
use std::io::Write;
use std::time::{Duration, Instant};

mod diagnostics;
mod output;

/// Max Newton iterations for the strong-coupling implicit-midpoint velocity solve.
const STRONG_MAXITER: usize = 30;

/// History depth for Anderson acceleration of the impulse fixed-point coupling.
const ANDERSON_DEPTH: usize = 5;

/// Max iterations for the per-body implicit-midpoint angular solve.
const ANG_MID_MAXITER: usize = 20;

/// Fluid--structure coupling scheme the integrator advances with. Replaces the
/// former `strong_couple`/`impulse_scheme` boolean pair and their precedence.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CouplingScheme {
    /// Explicit predictor-corrector with the unsteady-pressure force (and the
    /// optional added-mass Robin stabiliser).
    Explicit,
    /// Strong (implicit-midpoint) coupling of the linear DOF.
    Strong,
    /// Implicit impulse-difference scheme (momentum/energy-consistent).
    Impulse,
}

pub struct Rk4PCDM {
    /// BEM fluid coupling: pushes body state, solves, returns impulse/force/KE.
    solver: BemSolver,
    t: f64,
    /// The single owned rigid-body state (positions/velocities + orientations/
    /// angular velocities) handed to the solver each step.
    state: BodyState,
    o_lab: DVector<f64>,
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
    /// Fluid impulse captured at the start of the current step in impulse mode.
    fluid_impulse_lin_step_start: Option<DVector<f64>>,
    fluid_impulse_ang_step_start: Option<DVector<f64>>,
    /// Selected fluid--structure coupling scheme.
    scheme: CouplingScheme,
    initial_total_ke: Option<f64>,
}

/// Solid-body kinetic energy breakdown for one timestep.
pub(crate) struct SolidEnergy {
    total_lin: f64,
    total_rot: f64,
    total: f64,
    /// (linear, rotational, total) kinetic energy for each body.
    per_body: Vec<(f64, f64, f64)>,
}

impl Rk4PCDM {
    //Function for creating new solver
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        solver: BemSolver,
        t_begin: f64,
        x: LinearState, // (positions, velocities) stacked over bodies
        orientations: Vec<(Quaternion<f64>, Quaternion<f64>)>, // (orientation, angular velocity) per body
        inertias: Vec<Matrix3<f64>>,
        masses: Vec<f64>,
        added_mass_tensors: Vec<Matrix3<f64>>,
        added_mass_stab: bool,
        t_end: f64,
        step_size: f64,
        samp_rate: u32,
        print_rate: u32,
        scheme: CouplingScheme,
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
            solver,
            t: t_begin,
            state: BodyState {
                lin: x,
                ang: (q, omega),
            },
            o_lab: DVector::zeros(3 * nbody),
            nbody,
            inertias,
            masses,
            added_mass_safe,
            added_mass_stab,
            lin_accel_prev: DVector::zeros(3 * nbody),
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
            fluid_impulse_lin_step_start: None,
            fluid_impulse_ang_step_start: None,
            scheme,
            initial_total_ke: None,
        }
    }

    pub fn integrate(&mut self) -> Result<Stats, IntegrationError> {
        self.t_out.push(self.t);
        self.x_out.push(self.state.lin.clone());
        self.o_out.push(self.state.ang.clone());
        self.o_lab = self.orientation_to_marker_point();
        self.o_lab_out.push(self.o_lab.clone());
        self.print_initial_state();

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
                self.run_first_step_secs = steady_start.duration_since(start_t).as_secs_f64();
                println!("First step completed in {:.3} s.", self.run_first_step_secs);
                // println!("Progress timing starts now.");
                println!();
            }
            let t_new = self.t + self.step_size;

            if i % samp_rate == 0 {
                self.t_out.push(t_new);
                self.x_out.push(self.state.lin.clone());
                self.o_out.push(self.state.ang.clone());
                self.o_lab_out.push(self.o_lab.clone());
            }

            self.t = t_new;
            self.stats.accepted_steps += 1;
        }
        Ok(self.stats)
    }

    pub fn integrate_with_writer<W: Write>(
        &mut self,
        writer: &mut W,
    ) -> Result<Stats, IntegrationError> {
        self.t_out.push(self.t);
        self.x_out.push(self.state.lin.clone());
        self.o_out.push(self.state.ang.clone());
        self.o_lab = self.orientation_to_marker_point();
        self.o_lab_out.push(self.o_lab.clone());
        self.print_initial_state();

        self.write_header(writer)?;
        let x0 = self.state.lin.clone();
        let o0 = self.state.ang.clone();
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
                self.run_first_step_secs = steady_start.duration_since(start_t).as_secs_f64();
                println!("First step completed in {:.3} s.", self.run_first_step_secs);
                // println!("Progress timing starts now.");
                println!();
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
                self.x_out.push(self.state.lin.clone());
                self.o_out.push(self.state.ang.clone());
                self.o_lab_out.push(self.o_lab.clone());
                pending = Some((t_new, self.state.lin.clone(), self.state.ang.clone(), self.o_lab.clone()));
            }

            self.t = t_new;
            self.stats.accepted_steps += 1;
        }

        // Flush the final pending row. Its state is the end state, whose fluid KE
        // has not yet been computed, so run one extra BEM solve to obtain it (the
        // side effects — φ history push, fluid KE update — are harmless).
        if let Some((tp, xp, op, olp)) = pending.take() {
            if self.scheme == CouplingScheme::Impulse {
                let (l_lin, l_ang) = self.solver.impulse();
                self.fluid_impulse_lin_step_start = Some(l_lin);
                self.fluid_impulse_ang_step_start = Some(l_ang);
            } else {
                let _ = self.solver.force();
            }
            let kef = self.solver.fluid_kinetic_energy();
            self.write_row(writer, tp, &xp, &op, &olp, kef)?;
        }

        // Record timing for the end-of-run summary.
        self.run_wall_secs = start_t.elapsed().as_secs_f64();
        if self.run_first_step_secs == 0.0 {
            self.run_first_step_secs = steady_start.duration_since(start_t).as_secs_f64();
        }
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
        // The implicit schemes iterate each step to consistency, so they do not
        // need (and are corrupted by) the explicit scheme's repeated provisional
        // first step. Skip it.
        if self.scheme != CouplingScheme::Explicit {
            println!("Preparing first step: implicit scheme, no bootstrap passes required.");
            println!("First step may take longer than later steps.");
            println!();
            return;
        }
        println!(
            "Preparing first step: {} bootstrap pass(es); this may take longer than later steps.",
            BOOTSTRAP_PASSES
        );
        let x0 = self.state.lin.clone();
        let o0 = self.state.ang.clone();
        for _ in 0..BOOTSTRAP_PASSES {
            self.advance_one_step();
            // Rewind to the initial state and push it back into the system.
            self.state.lin = x0.clone();
            self.state.ang = o0.clone();
            let x_push = self.state.lin.clone();
            let o_push = self.state.ang.clone();
            self.solver.set_state(&BodyState { lin: x_push, ang: o_push });
        }
    }

    /// Advance the full state `(self.state.lin, self.state.ang, self.o_lab)` by one timestep with
    /// the configured coupling scheme.
    fn advance_one_step(&mut self) {
        match self.scheme {
            CouplingScheme::Impulse => self.advance_one_step_impulse(),
            CouplingScheme::Strong => self.advance_one_step_strong(),
            CouplingScheme::Explicit => self.advance_one_step_explicit(),
        }
    }

    /// Explicit predictor-corrector (Verlet-like translation + PCDM rotation) step
    /// with the unsteady-pressure force and optional added-mass Robin stabilisation.
    fn advance_one_step_explicit(&mut self) {
        // Forces at the start of the step.
        let (linear_accel_expl, angular_force) = self.solver.force();
        // Stage A is evaluated at exactly the step-start state z_n, so its fluid
        // KE is the correct value to write for the row sampled at time t_n.
        self.fluid_ke_step_start = self.solver.fluid_kinetic_energy();
        self.capture_initial_total_ke();

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

        self.solver.set_state(&BodyState { lin: (p_half, v_half.clone()), ang: (q_half.clone(), o_half.clone()) });

        // Forces at the half step.
        let (linear_force_half_expl, angular_force_half) = self.solver.force();

        let linear_force_half = if self.added_mass_stab {
            self.stabilise_lin_accel(&linear_force_half_expl)
        } else {
            linear_force_half_expl
        };

        let x_new = self.lin_full_step(&linear_force_half, &v_half);
        let o_new = self.ang_full_step(&angular_force_half, &(q_half, o_half));

        self.solver.set_state(&BodyState { lin: x_new.clone(), ang: o_new.clone() });

        let o_lab_new = self.orientation_to_marker_point();

        self.state.lin = x_new;
        self.state.ang = o_new;
        self.o_lab = o_lab_new;

        // Carry the stage-A explicit acceleration as a_prev for the next step's
        // Robin update. At steady state a_prev -> a_expl, so the stabiliser is
        // answer-preserving.
        if self.added_mass_stab {
            self.lin_accel_prev = linear_accel_expl;
        }
    }

    /// Approach A: one timestep with the implicit impulse-difference scheme.
    /// Force/torque are F = -(L_{n+1}-L_n)/dt, τ = -(Λ_{n+1}-Λ_n)/dt using the
    /// BEM state-function impulse. The linear velocity is solved implicitly so
    /// that m_s u + L_lin is conserved; the angular DOF are integrated by the
    /// existing Euler/PCDM machinery driven by the impulse-difference torque.
    /// Coupled end-state fixed point (linear + torque) with the added-mass
    /// preconditioner. No φ-history, no bootstrap.
    fn advance_one_step_impulse(&mut self) {
        let (l_lin_n, l_ang_n) = self.solver.impulse();
        self.fluid_ke_step_start = self.solver.fluid_kinetic_energy();
        self.fluid_impulse_lin_step_start = Some(l_lin_n.clone());
        self.fluid_impulse_ang_step_start = Some(l_ang_n.clone());
        self.capture_initial_total_ke();

        let (p_n, v_n) = self.state.lin.clone();
        let mut v_new = v_n.clone();
        let mut torque = DVector::zeros(3 * self.nbody);
        let mut o_new = self.state.ang.clone();

        // Anderson-acceleration history for the coupled (v_new, torque) iterate.
        let nb3 = 3 * self.nbody;
        let mut w_hist: Vec<DVector<f64>> = Vec::new();
        let mut f_hist: Vec<DVector<f64>> = Vec::new();

        let tol = 1e-9 * (v_n.norm() + 1.0);
        for _it in 0..STRONG_MAXITER {
            // Angular: implicit-midpoint Lie-group update driven by the current
            // step-averaged torque. Symplectic-flavoured (energy drift no longer
            // grows with the added-mass stiffness that the mesh resolves), unlike
            // the explicit predictor-corrector PCDM used by the other schemes.
            o_new = self.ang_step_implicit_midpoint(&torque);
            let q_end = o_new.0.clone();

            // Position uses the midpoint velocity; sync the end state.
            let v_mid = 0.5 * (&v_n + &v_new);
            let p_new = &p_n + &v_mid * self.step_size;
            self.solver.set_state(&BodyState { lin: (p_new.clone(), v_new.clone()), ang: o_new.clone() });

            let (l_lin_np1, l_ang_np1) = self.solver.impulse();

            // The fluid force/torque on the body is F = +dL/dt (validated sign;
            // L = ρ∮φn̂dA is negative for positive velocity, so the body conserves
            // m_s u - L, i.e. effective mass m_s + M_a). Residual of the implicit
            // midpoint m_s(v_{n+1}-v_n) = (L_{n+1}-L_n); torque N = dΛ/dt.
            let mut r_lin = DVector::zeros(3 * self.nbody);
            let mut new_torque = DVector::zeros(3 * self.nbody);
            for b in 0..self.nbody {
                let ms = self.masses[b];
                let v_old = Vector3::new(v_n[3 * b], v_n[3 * b + 1], v_n[3 * b + 2]);
                let v_end = Vector3::new(v_new[3 * b], v_new[3 * b + 1], v_new[3 * b + 2]);
                let l_old = Vector3::new(l_lin_n[3 * b], l_lin_n[3 * b + 1], l_lin_n[3 * b + 2]);
                let l_end =
                    Vector3::new(l_lin_np1[3 * b], l_lin_np1[3 * b + 1], l_lin_np1[3 * b + 2]);
                let lambda_old =
                    Vector3::new(l_ang_n[3 * b], l_ang_n[3 * b + 1], l_ang_n[3 * b + 2]);
                let lambda_end =
                    Vector3::new(l_ang_np1[3 * b], l_ang_np1[3 * b + 1], l_ang_np1[3 * b + 2]);

                for c in 0..3 {
                    r_lin[3 * b + c] = ms * (v_new[3 * b + c] - v_n[3 * b + c])
                        - (l_lin_np1[3 * b + c] - l_lin_n[3 * b + c]);
                }

                // Λ_b is taken about body b's translating centre, so the
                // consistent torque carries the reference-point transport term
                // −v×p_con. p_con_b = m v − L_lin is conserved per body by the
                // linear update above, so this holds for any nbody — the same
                // correction that fixed the single-body case.
                let v_mid = 0.5 * (v_old + v_end);
                let p_total_mid = 0.5 * (ms * v_old - l_old + ms * v_end - l_end);
                let torque_vec =
                    (lambda_end - lambda_old) / self.step_size - v_mid.cross(&p_total_mid);
                for c in 0..3 {
                    new_torque[3 * b + c] = torque_vec[c];
                }
            }
            // Base-map corrections. Linear: preconditioned Newton step
            // g_v = -P^{-1} r_lin with P = m_s I + M_a (body frame). Angular:
            // the torque functional-iteration step g_t = new_torque - torque.
            let mut g_v = DVector::zeros(nb3);
            for b in 0..self.nbody {
                let ms = self.masses[b];
                if ms <= 0.0 {
                    continue;
                }
                let r_lab = Vector3::new(r_lin[3 * b], r_lin[3 * b + 1], r_lin[3 * b + 2]);
                let r_b = rotation::lab_to_body(&Quaternion::from_imag(r_lab), &q_end[b])
                    .imag();
                let p_mat = Matrix3::identity() * ms + self.added_mass_safe[b];
                let d_b = p_mat.try_inverse().map(|inv| inv * r_b).unwrap_or(r_b / ms);
                let d_lab = rotation::body_to_lab(&Quaternion::from_imag(d_b), &q_end[b])
                    .imag();
                for c in 0..3 {
                    g_v[3 * b + c] = -d_lab[c];
                }
            }
            let g_t = &new_torque - &torque;

            // Same fixed-point test as before: linear residual and torque change.
            if r_lin.norm() < tol && g_t.norm() < tol {
                break;
            }

            // Anderson-accelerated update of the stacked iterate w = (v_new, torque)
            // with correction f = (g_v, g_t). Same fixed point, fewer iterations.
            let mut w_k = DVector::zeros(2 * nb3);
            let mut f_k = DVector::zeros(2 * nb3);
            for i in 0..nb3 {
                w_k[i] = v_new[i];
                w_k[nb3 + i] = torque[i];
                f_k[i] = g_v[i];
                f_k[nb3 + i] = g_t[i];
            }
            let w_next = anderson_next(&mut w_hist, &mut f_hist, &w_k, &f_k, ANDERSON_DEPTH);
            for i in 0..nb3 {
                v_new[i] = w_next[i];
                torque[i] = w_next[nb3 + i];
            }
        }

        let v_mid = 0.5 * (&v_n + &v_new);
        let p_new = &p_n + &v_mid * self.step_size;
        let x_new = (p_new, v_new);

        self.solver.set_state(&BodyState { lin: x_new.clone(), ang: o_new.clone() });
        let o_lab_new = self.orientation_to_marker_point();

        self.state.lin = x_new;
        self.state.ang = o_new;
        self.o_lab = o_lab_new;
    }

    /// Prototype B: one timestep with strong (implicit-midpoint) coupling of the
    /// linear velocity. Solves v_half = v_n + half_step * a(v_half) by Newton
    /// iteration with the analytic added-mass preconditioner P = I + M_a/(2 M_s),
    /// re-solving the BEM force at each trial v_half (history frozen). The
    /// added-mass reaction is then implicit, removing the explicit oscillation
    /// while preserving 2nd order. The angular DOF use the existing explicit
    /// midpoint (the added-mass instability is translational).
    fn advance_one_step_strong(&mut self) {
        // Stage A at z_n (commits φ(v_n) to the history).
        let (a_n, torque_n) = self.solver.force();
        self.fluid_ke_step_start = self.solver.fluid_kinetic_energy();
        self.capture_initial_total_ke();

        let (q_half, o_half) = self.ang_half_step(&torque_n);

        let (p_n, v_n) = self.state.lin.clone();
        let mut v_half = &v_n + &a_n * self.half_step; // explicit initial guess

        self.solver.set_freeze(true);
        let tol = 1e-9 * (v_n.norm() + 1.0);
        for _it in 0..STRONG_MAXITER {
            let p_half = &p_n + &v_half * self.half_step;
            self.solver.set_state(&BodyState { lin: (p_half, v_half.clone()), ang: (q_half.clone(), o_half.clone()) });
            let (a_h, _torque_h) = self.solver.force(); // frozen: no history push
            let resid = &v_half - &v_n - &a_h * self.half_step; // g(v_half)
            if resid.norm() < tol {
                break;
            }
            for b in 0..self.nbody {
                let ms = self.masses[b];
                if ms <= 0.0 {
                    continue;
                }
                let r_lab = Vector3::new(resid[3 * b], resid[3 * b + 1], resid[3 * b + 2]);
                let r_b = rotation::lab_to_body(&Quaternion::from_imag(r_lab), &q_half[b])
                    .imag();
                let p_mat = Matrix3::identity() + self.added_mass_safe[b] / (2.0 * ms);
                let d_b = p_mat.try_inverse().map(|inv| inv * r_b).unwrap_or(r_b);
                let d_lab = rotation::body_to_lab(&Quaternion::from_imag(d_b), &q_half[b])
                    .imag();
                for c in 0..3 {
                    v_half[3 * b + c] -= d_lab[c];
                }
            }
        }
        self.solver.set_freeze(false);

        // Commit stage B: evaluate at the converged v_half to push φ(v_half).
        let p_half = &p_n + &v_half * self.half_step;
        self.solver.set_state(&BodyState { lin: (p_half, v_half.clone()), ang: (q_half.clone(), o_half.clone()) });
        let (_a_final, torque_half) = self.solver.force();

        // Implicit-midpoint update: v_{n+1} = 2 v_half - v_n, x uses v_half.
        let v_new = 2.0 * &v_half - &v_n;
        let p_new = &p_n + &v_half * self.step_size;
        let x_new = (p_new, v_new);
        let o_new = self.ang_full_step(&torque_half, &(q_half, o_half));

        self.solver.set_state(&BodyState { lin: x_new.clone(), ang: o_new.clone() });
        let o_lab_new = self.orientation_to_marker_point();

        self.state.lin = x_new;
        self.state.ang = o_new;
        self.o_lab = o_lab_new;
    }

    /// Single-step semi-implicit (Robin) added-mass stabilisation of the
    /// lab-frame linear acceleration. Per body: rotate a_expl and a_prev into
    /// the body frame (where M_a is diagonal/constant), solve
    /// a_stab = (M_s I + M_a_safe)^{-1}(M_s a_expl + M_a_safe a_prev), and
    /// rotate back to the lab frame.
    fn stabilise_lin_accel(&self, accel_expl: &DVector<f64>) -> DVector<f64> {
        let (q, _) = &self.state.ang;
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
            let a_expl_b = rotation::lab_to_body(&Quaternion::from_imag(a_expl_lab), &q[i])
                .imag();
            let a_prev_b = rotation::lab_to_body(&Quaternion::from_imag(a_prev_lab), &q[i])
                .imag();

            let ms_i = Matrix3::identity() * ms;
            let m_a = self.added_mass_safe[i];
            let lhs = ms_i + m_a;
            let rhs = ms_i * a_expl_b + m_a * a_prev_b;
            let a_stab_b = lhs.try_inverse().map(|inv| inv * rhs).unwrap_or(a_expl_b);

            // Rotate body -> lab.
            let a_stab_lab = rotation::body_to_lab(&Quaternion::from_imag(a_stab_b), &q[i])
                .imag();
            for c in 0..3 {
                out[3 * i + c] = a_stab_lab[c];
            }
        }
        out
    }

    fn lin_half_step(&mut self, lin_force: &DVector<f64>) -> LinearState {
        let (p, v) = self.state.lin.clone();
        let p_new = &p + &v * self.half_step;
        let v_new = &v + lin_force * self.half_step;
        (p_new, v_new)
    }

    fn lin_full_step(&mut self, lin_force: &DVector<f64>, v_half: &DVector<f64>) -> LinearState {
        let (p, v) = self.state.lin.clone();
        // Position uses the midpoint velocity v_{n+1/2} (previously v_n — explicit
        // Euler for position, the source of a small O(dt) global position error).
        let p_new = &p + v_half * self.step_size;
        let v_new = &v + lin_force * self.step_size;
        (p_new, v_new)
    }

    fn ang_half_step(&mut self, ang: &DVector<f64>) -> AngularState {
        let (q, omega_lab) = self.state.ang.clone();

        let mut q_new = Vec::with_capacity(self.nbody);
        let mut o_new = Vec::with_capacity(self.nbody);

        for i in 0..self.nbody {
            let qi = q[i];
            let inertia = self.inertias[i];

            let omega_b = rotation::lab_to_body(&omega_lab[i], &qi);

            let torque_lab = Quaternion::new(0.0, ang[3 * i], ang[3 * i + 1], ang[3 * i + 2]);
            let torque = rotation::lab_to_body(&torque_lab, &qi);

            let ang_accel_b = rotation::accel_get(&omega_b, &inertia, &torque);

            let omega_n_quarter_b = self.omega_stepper(&omega_b, &ang_accel_b, self.quarter_step);
            let omega_n_half_b = self.omega_stepper(&omega_b, &ang_accel_b, self.half_step);

            let omega_n_quarter = rotation::body_to_lab(&omega_n_quarter_b, &qi);
            let qi_half_predict = self.orientation_stepper(&qi, &omega_n_quarter, self.half_step);

            let omega_n_half_lab = rotation::body_to_lab(&omega_n_half_b, &qi_half_predict);

            q_new.push(qi_half_predict);
            o_new.push(omega_n_half_lab);
        }

        (q_new, o_new)
    }

    fn ang_full_step(&mut self, ang: &DVector<f64>, half_qo: &AngularState) -> AngularState {
        let (q, omega_lab) = self.state.ang.clone();
        let (q_half, o_half) = half_qo;

        let mut q_full_v = Vec::with_capacity(self.nbody);
        let mut omega_v = Vec::with_capacity(self.nbody);

        for i in 0..self.nbody {
            let qi = q[i];
            let qi_half = q_half[i];
            let inertia = self.inertias[i];

            let omega_b = rotation::lab_to_body(&omega_lab[i], &qi);
            let omega_n_half_b = rotation::lab_to_body(&o_half[i], &qi_half);

            let torque_lab = Quaternion::new(0.0, ang[3 * i], ang[3 * i + 1], ang[3 * i + 2]);
            let torque = rotation::lab_to_body(&torque_lab, &qi_half);

            let ang_accel_half_b = rotation::accel_get(&omega_n_half_b, &inertia, &torque);
            let omega_n_half = rotation::body_to_lab(&omega_n_half_b, &qi_half);

            let qi_full = self.orientation_stepper(&qi, &omega_n_half, self.step_size);

            let omega_b_full = self.omega_stepper(&omega_b, &ang_accel_half_b, self.step_size);
            let omega_i = rotation::body_to_lab(&omega_b_full, &qi_full);

            q_full_v.push(qi_full);
            omega_v.push(omega_i);
        }

        self.stats.num_eval += self.nbody as u32;

        (q_full_v, omega_v)
    }

    /// Implicit-midpoint (symplectic Lie-group) angular update for the impulse
    /// scheme. Solves the body-frame Euler equation with the gyroscopic term and
    /// the orientation-dependent torque evaluated at the self-consistent midpoint,
    /// then reconstructs the orientation by the exponential map of the midpoint
    /// angular velocity. Unlike the explicit predictor-corrector PCDM, the energy
    /// error does not grow secularly with the added-mass stiffness the mesh
    /// resolves. `ang` is the lab-frame step-averaged torque per body.
    fn ang_step_implicit_midpoint(&mut self, ang: &DVector<f64>) -> AngularState {
        let (q, omega_lab) = self.state.ang.clone();
        let mut q_new = Vec::with_capacity(self.nbody);
        let mut o_new = Vec::with_capacity(self.nbody);

        for i in 0..self.nbody {
            let qi = q[i];
            let inertia = self.inertias[i];
            let n_lab = Quaternion::new(0.0, ang[3 * i], ang[3 * i + 1], ang[3 * i + 2]);
            let omega_n_b = rotation::lab_to_body(&omega_lab[i], &qi).imag();

            // Fixed point on the body-frame midpoint angular velocity
            // omega_mid = omega_n + (dt/2) I^{-1}(N(q_mid) - omega_mid x I omega_mid).
            let mut omega_mid_b = omega_n_b;
            for _ in 0..ANG_MID_MAXITER {
                // Midpoint orientation: rotate q_n by omega_mid over dt/2.
                let omega_mid_lab = rotation::body_to_lab(&Quaternion::from_imag(omega_mid_b), &qi);
                let q_mid = self.orientation_stepper(&qi, &omega_mid_lab, self.half_step);
                let n_mid_b = rotation::lab_to_body(&n_lab, &q_mid);
                let accel_mid_b =
                    rotation::accel_get(&Quaternion::from_imag(omega_mid_b), &inertia, &n_mid_b).imag();
                let omega_mid_new = omega_n_b + 0.5 * self.step_size * accel_mid_b;
                let delta = (omega_mid_new - omega_mid_b).norm();
                omega_mid_b = omega_mid_new;
                if delta < 1e-12 * (omega_n_b.norm() + 1.0) {
                    break;
                }
            }

            // Full step from the converged midpoint: orientation rotates by
            // omega_mid over dt; omega_{n+1} = 2 omega_mid - omega_n.
            let omega_mid_lab = rotation::body_to_lab(&Quaternion::from_imag(omega_mid_b), &qi);
            let q_full = self.orientation_stepper(&qi, &omega_mid_lab, self.step_size);
            let omega_full_b = 2.0 * omega_mid_b - omega_n_b;
            let omega_full_lab = rotation::body_to_lab(&Quaternion::from_imag(omega_full_b), &q_full);

            q_new.push(q_full);
            o_new.push(omega_full_lab);
        }

        self.stats.num_eval += self.nbody as u32;
        (q_new, o_new)
    }

    fn multiply_duration(&self, duration: Duration, factor: f64) -> Duration {
        let seconds =
            duration.as_secs() as f64 + f64::from(duration.subsec_nanos()) / 1_000_000_000.0;
        let result_seconds = seconds * factor;
        let whole_seconds = result_seconds as u64;
        let fractional_seconds = ((result_seconds - whole_seconds as f64) * 1_000_000_000.0) as u32;
        Duration::new(whole_seconds, fractional_seconds)
    }

    /// Body-frame x-axis marker direction in the lab frame, for each body (3*N vector).
    fn orientation_to_marker_point(&self) -> DVector<f64> {
        let (q, _) = &self.state.ang;
        let mut out = DVector::zeros(3 * self.nbody);
        for i in 0..self.nbody {
            let marker =
                rotation::body_to_lab(&Quaternion::from_imag(Vector3::new(1.0, 0.0, 0.0)), &q[i]);
            let v = marker.vector();
            out[3 * i] = v[0];
            out[3 * i + 1] = v[1];
            out[3 * i + 2] = v[2];
        }
        out
    }

    //Steps forward the rotational velocity omega_n according to a given angular acceleration/torque.
    fn omega_stepper(
        &self,
        omega_n: &Quaternion<f64>,
        ang_accel: &Quaternion<f64>,
        dt: f64,
    ) -> Quaternion<f64> {
        omega_n + ang_accel * dt
    }

    //Steps forward the orientation of a body given initial orientation q1 and rotational velocity omega.
    fn orientation_stepper(
        &self,
        q1: &Quaternion<f64>,
        omega: &Quaternion<f64>,
        dt: f64,
    ) -> Quaternion<f64> {
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
