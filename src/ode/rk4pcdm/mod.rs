use crate::bem::bem_for_ode::{AngularState, BemSolver, BodyState, LinearState};
use crate::math::anderson::anderson_next;
use crate::math::rotation;
use crate::ode::dop_shared::{IntegrationError, Stats};
use crate::system::system::BOOTSTRAP_PASSES;
use nalgebra::{DMatrix, DVector, Matrix3, Quaternion, Unit, UnitQuaternion, Vector3};
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

/// Fixed-point iterations for the experimental Hamiltonian/discrete-gradient
/// endpoint solve.
const HAMILTONIAN_MAXITER: usize = 12;

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
    /// Experimental endpoint scheme: iterate a midpoint kinematic update with a
    /// final-state projection that preserves global conserved impulse and total
    /// body-fluid kinetic energy.
    Hamiltonian,
    /// Experimental midpoint Hamiltonian scheme: solve endpoint configuration
    /// from a BEM/projection evaluated at the midpoint configuration.
    HamiltonianMidpoint,
    /// Experimental discrete-Lagrangian midpoint variational integrator. Solves
    /// the fully determined discrete Euler-Lagrange equations for q_{n+1}.
    Variational,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PairMetricMode {
    PointGradient,
    TranslationalDiscreteGradient,
}

pub struct Rk4PCDM {
    /// BEM fluid coupling: pushes body state, solves, returns impulse/force/KE.
    solver: BemSolver,
    t: f64,
    /// The single owned rigid-body state (positions/velocities + orientations/
    /// angular velocities) handed to the solver each step.
    state: BodyState,
    /// Previous accepted configuration for the two-point discrete Lagrangian
    /// update. Velocities stored in this state are ignored.
    prev_variational_state: Option<BodyState>,
    o_lab: DVector<f64>,
    nbody: usize,
    inertias: Vec<Matrix3<f64>>,
    masses: Vec<f64>,
    body_info: Vec<BodyInfo>,
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
    /// Aggregate diagnostics for the experimental energy projection.
    pub projection_max_corr_rel: f64,
    pub projection_max_energy_err_rel: f64,
    pub projection_max_energy_floor_rel: f64,
    pub projection_max_energy_floor_abs: f64,
    pub projection_max_constraint_resid: f64,
    pub projection_floor_hit_count: usize,
    pub projection_floor_fallback_count: usize,
    pub projection_last_step_floor_hits: usize,
    pub projection_last_step_floor_fallbacks: usize,
    pub projection_last_step_max_floor_rel: f64,
    pub projection_last_step_max_floor_abs: f64,
    pub coupled_max_residual_norm: f64,
    pub coupled_max_impulse_resid: f64,
    pub coupled_max_energy_err_rel: f64,
    pub coupled_max_raw_linear_impulse_resid: f64,
    pub coupled_max_raw_angular_impulse_resid: f64,
    pub coupled_max_true_energy_err_rel: f64,
    pub coupled_last_step_residual_norm: f64,
    pub coupled_last_step_impulse_resid: f64,
    pub coupled_last_step_energy_err_rel: f64,
    pub coupled_last_step_raw_linear_impulse_resid: f64,
    pub coupled_last_step_raw_angular_impulse_resid: f64,
    pub coupled_last_step_true_energy_err_rel: f64,
    pub coupled_max_correction_rel: f64,
    pub coupled_last_step_correction_rel: f64,
    pub coupled_max_correction_kinetic_rel: f64,
    pub coupled_last_step_correction_kinetic_rel: f64,
    pub coupled_min_jacobian_rank: usize,
    pub coupled_max_jacobian_nullity: usize,
    pub coupled_min_jacobian_sigma: f64,
    pub coupled_jacobian_builds: usize,
    pub hamiltonian_adaptive_retry_count: usize,
    pub hamiltonian_max_substeps_used: usize,
    pub impulse_fp_steps: usize,
    pub impulse_fp_iter_sum: usize,
    pub impulse_fp_last_iter: usize,
    pub impulse_fp_max_iter: usize,
    pub impulse_start_cache_hits: usize,
    pub impulse_start_direct_solves: usize,
    pub variational_discrete_momentum_last_drift: f64,
    pub variational_discrete_momentum_max_drift: f64,
    pub impulse_variational_defect_probe_count: usize,
    pub impulse_variational_defect_last_norm: f64,
    pub impulse_variational_defect_max_norm: f64,
    pub impulse_variational_defect_last_metric_cos: f64,
    pub impulse_variational_defect_last_metric_scale: f64,
    pub impulse_variational_defect_last_pressure_cos: f64,
    pub impulse_variational_defect_last_pressure_scale: f64,
    pub impulse_global_p_drift_last: f64,
    pub impulse_global_h_drift_last: f64,
    pub impulse_body_p_drift_max_last: f64,
    pub impulse_body_h_drift_max_last: f64,
    pub impulse_global_p_drift_max: f64,
    pub impulse_global_h_drift_max: f64,
    pub impulse_body_p_drift_max: f64,
    pub impulse_body_h_drift_max: f64,
    impulse_variational_defect_out: bool,
    impulse_partition_initial: Option<ImpulsePartitionState>,
    impulse_next_start_cache: Option<ImpulseStartCache>,
    variational_discrete_momentum_out: Option<DVector<f64>>,
    variational_discrete_momentum_initial: Option<DVector<f64>>,
    /// Fluid KE captured at the start of the current step (stage-A force call,
    /// evaluated at exactly z_n), used to write the row sampled at time t_n.
    fluid_ke_step_start: f64,
    /// Fluid impulse captured at the start of the current step in impulse mode.
    fluid_impulse_lin_step_start: Option<DVector<f64>>,
    fluid_impulse_ang_step_start: Option<DVector<f64>>,
    /// Selected fluid--structure coupling scheme.
    scheme: CouplingScheme,
    /// Experimental impulse-mode post-step projection: adjusts generalized
    /// velocities to restore step-start KE while preserving total impulse.
    energy_projection: bool,
    /// Use the full kinetic-energy metric when choosing the particular
    /// impulse-conserving velocity in the energy projection. This is much more
    /// expensive because the metric is assembled from BEM energy evaluations,
    /// so the default production projection uses the Euclidean minimum-norm
    /// particular solution and leaves this as a diagnostic option.
    projection_kinetic_metric: bool,
    /// Experimental impulse-mode configuration-gradient correction. This is a
    /// deliberately slow finite-difference diagnostic for the multibody
    /// added-mass metric force.
    fluid_energy_gradient: bool,
    /// Experimental impulse-mode coordinate discrete-gradient correction. This
    /// estimates the configuration work over the proposed step at fixed
    /// midpoint velocity so the impulse update can carry the fluid KE metric
    /// force without a post-step energy projection.
    fluid_energy_discrete_gradient: bool,
    fluid_energy_gradient_eps: f64,
    fluid_energy_gradient_linear_scale: f64,
    fluid_energy_gradient_angular_scale: f64,
    impulse_pair_metric_correction: bool,
    impulse_pair_metric_mode: PairMetricMode,
    impulse_pair_metric_cutoff: f64,
    impulse_pair_metric_inner_cutoff: f64,
    impulse_pair_metric_outer_cutoff: f64,
    impulse_pair_metric_eps: f64,
    impulse_pair_metric_linear_scale: f64,
    impulse_pair_metric_angular_scale: f64,
    pub impulse_pair_metric_last_pairs: usize,
    pub impulse_pair_metric_max_pairs: usize,
    pub impulse_pair_metric_last_norm: f64,
    pub impulse_pair_metric_max_norm: f64,
    impulse_quadratic_pressure: bool,
    impulse_quadratic_pressure_scale: f64,
    impulse_internal_load_constraint: bool,
    impulse_variational_defect_probe: bool,
    variational_momentum_diagnostic: bool,
    variational_reuse_step_jacobian: bool,
    variational_energy_only_lagrangian: bool,
    variational_step_jacobian: Option<DMatrix<f64>>,
    hamiltonian_substeps: usize,
    hamiltonian_adaptive_substeps: bool,
    hamiltonian_max_substeps: usize,
    hamiltonian_floor_tol: f64,
    hamiltonian_coupled_solve: bool,
    hamiltonian_coupled_iters: usize,
    hamiltonian_coupled_eps: f64,
    hamiltonian_coupled_max_shift: f64,
    hamiltonian_coupled_jacobian_interval: usize,
    hamiltonian_coupled_broyden_update: bool,
    hamiltonian_coupled_endpoint_velocity: bool,
    hamiltonian_coupled_kinetic_metric: bool,
    initial_total_ke: Option<f64>,
}

#[derive(Clone)]
pub struct BodyInfo {
    pub density: f64,
    pub shape: Vector3<f64>,
    pub initial_ke_ratio: f64,
}

#[derive(Clone)]
struct ImpulsePartitionState {
    per_body_p: Vec<Vector3<f64>>,
    per_body_h: Vec<Vector3<f64>>,
    global_p: Vector3<f64>,
    global_h: Vector3<f64>,
}

struct ImpulseStartCache {
    l_lin: DVector<f64>,
    l_ang: DVector<f64>,
    fluid_ke: f64,
}

/// Solid-body kinetic energy breakdown for one timestep.
pub(crate) struct SolidEnergy {
    total_lin: f64,
    total_rot: f64,
    total: f64,
    /// (linear, rotational, total) kinetic energy for each body.
    per_body: Vec<(f64, f64, f64)>,
}

#[derive(Clone, Copy)]
struct ProjectionDiagnosticsSnapshot {
    max_corr_rel: f64,
    max_energy_err_rel: f64,
    max_energy_floor_rel: f64,
    max_energy_floor_abs: f64,
    max_constraint_resid: f64,
    floor_hit_count: usize,
    floor_fallback_count: usize,
}

#[derive(Clone, Copy)]
struct CoupledDiagnosticsSnapshot {
    max_residual_norm: f64,
    max_impulse_resid: f64,
    max_energy_err_rel: f64,
    max_raw_linear_impulse_resid: f64,
    max_raw_angular_impulse_resid: f64,
    max_true_energy_err_rel: f64,
    max_correction_rel: f64,
    max_correction_kinetic_rel: f64,
    min_jacobian_rank: usize,
    max_jacobian_nullity: usize,
    min_jacobian_sigma: f64,
}

struct CoupledResidualEval {
    scaled: DVector<f64>,
    raw_linear_impulse_resid: f64,
    raw_angular_impulse_resid: f64,
    true_energy_err_rel: f64,
}

impl Rk4PCDM {
    //Function for creating new solver
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mut solver: BemSolver,
        t_begin: f64,
        x: LinearState, // (positions, velocities) stacked over bodies
        orientations: Vec<(Quaternion<f64>, Quaternion<f64>)>, // (orientation, angular velocity) per body
        inertias: Vec<Matrix3<f64>>,
        masses: Vec<f64>,
        body_info: Vec<BodyInfo>,
        added_mass_tensors: Vec<Matrix3<f64>>,
        added_mass_stab: bool,
        t_end: f64,
        step_size: f64,
        samp_rate: u32,
        print_rate: u32,
        scheme: CouplingScheme,
        energy_projection: bool,
        projection_kinetic_metric: bool,
        fluid_energy_gradient: bool,
        fluid_energy_discrete_gradient: bool,
        fluid_energy_gradient_eps: f64,
        fluid_energy_gradient_linear_scale: f64,
        fluid_energy_gradient_angular_scale: f64,
        impulse_pair_metric_correction: bool,
        impulse_pair_metric_mode: usize,
        impulse_pair_metric_cutoff: f64,
        impulse_pair_metric_inner_cutoff: f64,
        impulse_pair_metric_outer_cutoff: f64,
        impulse_pair_metric_eps: f64,
        impulse_pair_metric_linear_scale: f64,
        impulse_pair_metric_angular_scale: f64,
        impulse_quadratic_pressure: bool,
        impulse_quadratic_pressure_scale: f64,
        impulse_internal_load_constraint: bool,
        impulse_variational_defect_probe: bool,
        variational_momentum_diagnostic: bool,
        variational_reuse_step_jacobian: bool,
        variational_energy_only_lagrangian: bool,
        hamiltonian_substeps: usize,
        hamiltonian_adaptive_substeps: bool,
        hamiltonian_max_substeps: usize,
        hamiltonian_floor_tol: f64,
        hamiltonian_coupled_solve: bool,
        hamiltonian_coupled_iters: usize,
        hamiltonian_coupled_eps: f64,
        hamiltonian_coupled_max_shift: f64,
        hamiltonian_coupled_jacobian_interval: usize,
        hamiltonian_coupled_broyden_update: bool,
        hamiltonian_coupled_endpoint_velocity: bool,
        hamiltonian_coupled_kinetic_metric: bool,
    ) -> Self {
        let nbody = orientations.len();
        let q: Vec<Quaternion<f64>> = orientations.iter().map(|o| o.0).collect();
        let omega: Vec<Quaternion<f64>> = orientations.iter().map(|o| o.1).collect();

        // added_mass_tensors are already scaled by the safety factor in main
        // (input key added_mass_safety). M_a only needs to be >= the true
        // effective added mass for stability of the Robin update and does not
        // change the fixed point (a_stab = a_expl).
        let added_mass_safe: Vec<Matrix3<f64>> = added_mass_tensors;
        solver.set_compute_quadratic_pressure(
            impulse_quadratic_pressure || impulse_variational_defect_probe,
        );

        Rk4PCDM {
            solver,
            t: t_begin,
            state: BodyState {
                lin: x,
                ang: (q, omega),
            },
            prev_variational_state: None,
            o_lab: DVector::zeros(3 * nbody),
            nbody,
            inertias,
            masses,
            body_info,
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
            projection_max_corr_rel: 0.0,
            projection_max_energy_err_rel: 0.0,
            projection_max_energy_floor_rel: 0.0,
            projection_max_energy_floor_abs: 0.0,
            projection_max_constraint_resid: 0.0,
            projection_floor_hit_count: 0,
            projection_floor_fallback_count: 0,
            projection_last_step_floor_hits: 0,
            projection_last_step_floor_fallbacks: 0,
            projection_last_step_max_floor_rel: 0.0,
            projection_last_step_max_floor_abs: 0.0,
            coupled_max_residual_norm: 0.0,
            coupled_max_impulse_resid: 0.0,
            coupled_max_energy_err_rel: 0.0,
            coupled_max_raw_linear_impulse_resid: 0.0,
            coupled_max_raw_angular_impulse_resid: 0.0,
            coupled_max_true_energy_err_rel: 0.0,
            coupled_last_step_residual_norm: 0.0,
            coupled_last_step_impulse_resid: 0.0,
            coupled_last_step_energy_err_rel: 0.0,
            coupled_last_step_raw_linear_impulse_resid: 0.0,
            coupled_last_step_raw_angular_impulse_resid: 0.0,
            coupled_last_step_true_energy_err_rel: 0.0,
            coupled_max_correction_rel: 0.0,
            coupled_last_step_correction_rel: 0.0,
            coupled_max_correction_kinetic_rel: 0.0,
            coupled_last_step_correction_kinetic_rel: 0.0,
            coupled_min_jacobian_rank: usize::MAX,
            coupled_max_jacobian_nullity: 0,
            coupled_min_jacobian_sigma: f64::INFINITY,
            coupled_jacobian_builds: 0,
            hamiltonian_adaptive_retry_count: 0,
            hamiltonian_max_substeps_used: hamiltonian_substeps.max(1),
            impulse_fp_steps: 0,
            impulse_fp_iter_sum: 0,
            impulse_fp_last_iter: 0,
            impulse_fp_max_iter: 0,
            impulse_start_cache_hits: 0,
            impulse_start_direct_solves: 0,
            variational_discrete_momentum_last_drift: f64::NAN,
            variational_discrete_momentum_max_drift: 0.0,
            impulse_variational_defect_probe_count: 0,
            impulse_variational_defect_last_norm: f64::NAN,
            impulse_variational_defect_max_norm: 0.0,
            impulse_variational_defect_last_metric_cos: f64::NAN,
            impulse_variational_defect_last_metric_scale: f64::NAN,
            impulse_variational_defect_last_pressure_cos: f64::NAN,
            impulse_variational_defect_last_pressure_scale: f64::NAN,
            impulse_global_p_drift_last: f64::NAN,
            impulse_global_h_drift_last: f64::NAN,
            impulse_body_p_drift_max_last: f64::NAN,
            impulse_body_h_drift_max_last: f64::NAN,
            impulse_global_p_drift_max: 0.0,
            impulse_global_h_drift_max: 0.0,
            impulse_body_p_drift_max: 0.0,
            impulse_body_h_drift_max: 0.0,
            impulse_variational_defect_out: false,
            impulse_partition_initial: None,
            impulse_next_start_cache: None,
            variational_discrete_momentum_out: None,
            variational_discrete_momentum_initial: None,
            fluid_ke_step_start: 0.0,
            fluid_impulse_lin_step_start: None,
            fluid_impulse_ang_step_start: None,
            scheme,
            energy_projection,
            projection_kinetic_metric,
            fluid_energy_gradient,
            fluid_energy_discrete_gradient,
            fluid_energy_gradient_eps,
            fluid_energy_gradient_linear_scale,
            fluid_energy_gradient_angular_scale,
            impulse_pair_metric_correction,
            impulse_pair_metric_mode: if impulse_pair_metric_mode == 1 {
                PairMetricMode::TranslationalDiscreteGradient
            } else {
                PairMetricMode::PointGradient
            },
            impulse_pair_metric_cutoff,
            impulse_pair_metric_inner_cutoff,
            impulse_pair_metric_outer_cutoff,
            impulse_pair_metric_eps,
            impulse_pair_metric_linear_scale,
            impulse_pair_metric_angular_scale,
            impulse_pair_metric_last_pairs: 0,
            impulse_pair_metric_max_pairs: 0,
            impulse_pair_metric_last_norm: f64::NAN,
            impulse_pair_metric_max_norm: 0.0,
            impulse_quadratic_pressure,
            impulse_quadratic_pressure_scale,
            impulse_internal_load_constraint,
            impulse_variational_defect_probe,
            variational_momentum_diagnostic,
            variational_reuse_step_jacobian,
            variational_energy_only_lagrangian,
            variational_step_jacobian: None,
            hamiltonian_substeps: hamiltonian_substeps.max(1),
            hamiltonian_adaptive_substeps,
            hamiltonian_max_substeps: hamiltonian_max_substeps.max(hamiltonian_substeps.max(1)),
            hamiltonian_floor_tol: hamiltonian_floor_tol.max(0.0),
            hamiltonian_coupled_solve,
            hamiltonian_coupled_iters: hamiltonian_coupled_iters.max(1),
            hamiltonian_coupled_eps: hamiltonian_coupled_eps.max(1.0e-8),
            hamiltonian_coupled_max_shift: hamiltonian_coupled_max_shift.max(0.0),
            hamiltonian_coupled_jacobian_interval: hamiltonian_coupled_jacobian_interval.max(1),
            hamiltonian_coupled_broyden_update,
            hamiltonian_coupled_endpoint_velocity,
            hamiltonian_coupled_kinetic_metric,
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
            self.print_projection_floor_event(i + 1, t_new);
            self.ensure_finite_state(t_new)?;

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
            self.print_projection_floor_event(i + 1, t_new);

            // Flush the pending row now: its state is the start state of the step
            // just run, so `fluid_ke_step_start` (captured in stage A) is the
            // fluid KE at exactly that row's state.
            if let Some((tp, xp, op, olp)) = pending.take() {
                self.write_row(writer, tp, &xp, &op, &olp, self.fluid_ke_step_start)?;
            }

            self.ensure_finite_state(t_new)?;

            if i % samp_rate == 0 {
                self.t_out.push(t_new);
                self.x_out.push(self.state.lin.clone());
                self.o_out.push(self.state.ang.clone());
                self.o_lab_out.push(self.o_lab.clone());
                pending = Some((
                    t_new,
                    self.state.lin.clone(),
                    self.state.ang.clone(),
                    self.o_lab.clone(),
                ));
            }

            self.t = t_new;
            self.stats.accepted_steps += 1;
        }

        // Flush the final pending row. Its state is the end state, whose fluid KE
        // has not yet been written, so run one extra BEM solve to obtain it.
        if let Some((tp, xp, op, olp)) = pending.take() {
            if matches!(
                self.scheme,
                CouplingScheme::Impulse
                    | CouplingScheme::Hamiltonian
                    | CouplingScheme::HamiltonianMidpoint
                    | CouplingScheme::Variational
            ) {
                let (l_lin, l_ang) = self.solver.impulse();
                self.record_impulse_partition_drift(&xp, &op, &l_lin, &l_ang);
                self.fluid_impulse_lin_step_start = Some(l_lin);
                self.fluid_impulse_ang_step_start = Some(l_ang);
            } else {
                let _ = self.solver.force();
            }
            let kef = self.solver.fluid_kinetic_energy();
            if self.scheme == CouplingScheme::Variational {
                self.variational_discrete_momentum_out = None;
            }
            self.impulse_variational_defect_out = false;
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

    fn ensure_finite_state(&self, t: f64) -> Result<(), IntegrationError> {
        let (pos, vel) = &self.state.lin;
        if let Some((idx, value)) = pos.iter().enumerate().find(|(_, v)| !v.is_finite()) {
            return Err(IntegrationError::NonFiniteState {
                x: t,
                reason: format!("position[{idx}] = {value}"),
            });
        }
        if let Some((idx, value)) = vel.iter().enumerate().find(|(_, v)| !v.is_finite()) {
            return Err(IntegrationError::NonFiniteState {
                x: t,
                reason: format!("velocity[{idx}] = {value}"),
            });
        }

        let (q, omega) = &self.state.ang;
        for (idx, qi) in q.iter().enumerate() {
            for (name, value) in [("w", qi.w), ("i", qi.i), ("j", qi.j), ("k", qi.k)] {
                if !value.is_finite() {
                    return Err(IntegrationError::NonFiniteState {
                        x: t,
                        reason: format!("orientation[{idx}].{name} = {value}"),
                    });
                }
            }
        }
        for (idx, wi) in omega.iter().enumerate() {
            for (name, value) in [("w", wi.w), ("i", wi.i), ("j", wi.j), ("k", wi.k)] {
                if !value.is_finite() {
                    return Err(IntegrationError::NonFiniteState {
                        x: t,
                        reason: format!("angular_velocity[{idx}].{name} = {value}"),
                    });
                }
            }
        }

        let ke_solid = self.solid_kinetic_energy(&self.state.lin, &self.state.ang);
        if !ke_solid.total.is_finite() {
            return Err(IntegrationError::NonFiniteState {
                x: t,
                reason: format!("solid kinetic energy = {}", ke_solid.total),
            });
        }
        let ke_fluid = self.solver.fluid_kinetic_energy();
        if !ke_fluid.is_finite() {
            return Err(IntegrationError::NonFiniteState {
                x: t,
                reason: format!("fluid kinetic energy = {ke_fluid}"),
            });
        }

        Ok(())
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
            self.solver.set_state(&BodyState {
                lin: x_push,
                ang: o_push,
            });
        }
    }

    /// Advance the full state `(self.state.lin, self.state.ang, self.o_lab)` by one timestep with
    /// the configured coupling scheme.
    fn advance_one_step(&mut self) {
        self.reset_projection_step_diagnostics();
        match self.scheme {
            CouplingScheme::Variational => self.advance_one_step_variational(),
            CouplingScheme::HamiltonianMidpoint => self.advance_one_step_hamiltonian_midpoint(),
            CouplingScheme::Hamiltonian => self.advance_one_step_hamiltonian(),
            CouplingScheme::Impulse => self.advance_one_step_impulse(),
            CouplingScheme::Strong => self.advance_one_step_strong(),
            CouplingScheme::Explicit => self.advance_one_step_explicit(),
        }
    }

    fn reset_projection_step_diagnostics(&mut self) {
        self.projection_last_step_floor_hits = 0;
        self.projection_last_step_floor_fallbacks = 0;
        self.projection_last_step_max_floor_rel = 0.0;
        self.projection_last_step_max_floor_abs = 0.0;
    }

    fn projection_diagnostics_snapshot(&self) -> ProjectionDiagnosticsSnapshot {
        ProjectionDiagnosticsSnapshot {
            max_corr_rel: self.projection_max_corr_rel,
            max_energy_err_rel: self.projection_max_energy_err_rel,
            max_energy_floor_rel: self.projection_max_energy_floor_rel,
            max_energy_floor_abs: self.projection_max_energy_floor_abs,
            max_constraint_resid: self.projection_max_constraint_resid,
            floor_hit_count: self.projection_floor_hit_count,
            floor_fallback_count: self.projection_floor_fallback_count,
        }
    }

    fn restore_projection_diagnostics(&mut self, snapshot: ProjectionDiagnosticsSnapshot) {
        self.projection_max_corr_rel = snapshot.max_corr_rel;
        self.projection_max_energy_err_rel = snapshot.max_energy_err_rel;
        self.projection_max_energy_floor_rel = snapshot.max_energy_floor_rel;
        self.projection_max_energy_floor_abs = snapshot.max_energy_floor_abs;
        self.projection_max_constraint_resid = snapshot.max_constraint_resid;
        self.projection_floor_hit_count = snapshot.floor_hit_count;
        self.projection_floor_fallback_count = snapshot.floor_fallback_count;
        self.reset_projection_step_diagnostics();
    }

    fn coupled_diagnostics_snapshot(&self) -> CoupledDiagnosticsSnapshot {
        CoupledDiagnosticsSnapshot {
            max_residual_norm: self.coupled_max_residual_norm,
            max_impulse_resid: self.coupled_max_impulse_resid,
            max_energy_err_rel: self.coupled_max_energy_err_rel,
            max_raw_linear_impulse_resid: self.coupled_max_raw_linear_impulse_resid,
            max_raw_angular_impulse_resid: self.coupled_max_raw_angular_impulse_resid,
            max_true_energy_err_rel: self.coupled_max_true_energy_err_rel,
            max_correction_rel: self.coupled_max_correction_rel,
            max_correction_kinetic_rel: self.coupled_max_correction_kinetic_rel,
            min_jacobian_rank: self.coupled_min_jacobian_rank,
            max_jacobian_nullity: self.coupled_max_jacobian_nullity,
            min_jacobian_sigma: self.coupled_min_jacobian_sigma,
        }
    }

    fn restore_coupled_diagnostics(&mut self, snapshot: CoupledDiagnosticsSnapshot) {
        self.coupled_max_residual_norm = snapshot.max_residual_norm;
        self.coupled_max_impulse_resid = snapshot.max_impulse_resid;
        self.coupled_max_energy_err_rel = snapshot.max_energy_err_rel;
        self.coupled_max_raw_linear_impulse_resid = snapshot.max_raw_linear_impulse_resid;
        self.coupled_max_raw_angular_impulse_resid = snapshot.max_raw_angular_impulse_resid;
        self.coupled_max_true_energy_err_rel = snapshot.max_true_energy_err_rel;
        self.coupled_max_correction_rel = snapshot.max_correction_rel;
        self.coupled_max_correction_kinetic_rel = snapshot.max_correction_kinetic_rel;
        self.coupled_min_jacobian_rank = snapshot.min_jacobian_rank;
        self.coupled_max_jacobian_nullity = snapshot.max_jacobian_nullity;
        self.coupled_min_jacobian_sigma = snapshot.min_jacobian_sigma;
        self.coupled_last_step_residual_norm = 0.0;
        self.coupled_last_step_impulse_resid = 0.0;
        self.coupled_last_step_energy_err_rel = 0.0;
        self.coupled_last_step_raw_linear_impulse_resid = 0.0;
        self.coupled_last_step_raw_angular_impulse_resid = 0.0;
        self.coupled_last_step_true_energy_err_rel = 0.0;
        self.coupled_last_step_correction_rel = 0.0;
        self.coupled_last_step_correction_kinetic_rel = 0.0;
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

        self.solver.set_state(&BodyState {
            lin: (p_half, v_half.clone()),
            ang: (q_half.clone(), o_half.clone()),
        });

        // Forces at the half step.
        let (linear_force_half_expl, angular_force_half) = self.solver.force();

        let linear_force_half = if self.added_mass_stab {
            self.stabilise_lin_accel(&linear_force_half_expl)
        } else {
            linear_force_half_expl
        };

        let x_new = self.lin_full_step(&linear_force_half, &v_half);
        let o_new = self.ang_full_step(&angular_force_half, &(q_half, o_half));

        self.solver.set_state(&BodyState {
            lin: x_new.clone(),
            ang: o_new.clone(),
        });

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
        let (l_lin_n, l_ang_n) = if let Some(cache) = self.impulse_next_start_cache.take() {
            self.impulse_start_cache_hits += 1;
            self.fluid_ke_step_start = cache.fluid_ke;
            (cache.l_lin, cache.l_ang)
        } else {
            self.impulse_start_direct_solves += 1;
            let impulse = self.solver.impulse();
            self.fluid_ke_step_start = self.solver.fluid_kinetic_energy();
            impulse
        };
        self.record_impulse_partition_drift(
            &self.state.lin.clone(),
            &self.state.ang.clone(),
            &l_lin_n,
            &l_ang_n,
        );
        self.fluid_impulse_lin_step_start = Some(l_lin_n.clone());
        self.fluid_impulse_ang_step_start = Some(l_ang_n.clone());
        self.capture_initial_total_ke();
        let ke_step_start = self
            .solid_kinetic_energy(&self.state.lin, &self.state.ang)
            .total
            + self.fluid_ke_step_start;

        let start_state = self.state.clone();
        let (p_n, v_n) = self.state.lin.clone();
        let conserved_step_start =
            self.total_conserved_impulse(&p_n, &v_n, &self.state.ang, &l_lin_n, &l_ang_n);
        let mut v_new = v_n.clone();
        let mut torque = DVector::zeros(3 * self.nbody);
        let mut o_new = self.state.ang.clone();
        let mut l_lin_step_end = l_lin_n.clone();
        let mut l_ang_step_end = l_ang_n.clone();
        let mut pair_metric_force = DVector::zeros(3 * self.nbody);
        let mut pair_metric_torque = DVector::zeros(3 * self.nbody);
        let mut pair_metric_ready = !self.impulse_pair_metric_correction;

        // Anderson-acceleration history for the coupled (v_new, torque) iterate.
        let nb3 = 3 * self.nbody;
        let mut w_hist: Vec<DVector<f64>> = Vec::new();
        let mut f_hist: Vec<DVector<f64>> = Vec::new();

        let tol = 1e-9 * (v_n.norm() + 1.0);
        let mut fp_iters = 0usize;
        for it in 0..STRONG_MAXITER {
            fp_iters = it + 1;
            // Angular: implicit-midpoint Lie-group update driven by the current
            // step-averaged torque. Symplectic-flavoured (energy drift no longer
            // grows with the added-mass stiffness that the mesh resolves), unlike
            // the explicit predictor-corrector PCDM used by the other schemes.
            o_new = self.ang_step_implicit_midpoint(&torque);
            let q_end = o_new.0.clone();

            // Position uses the midpoint velocity; sync the end state.
            let v_mid = 0.5 * (&v_n + &v_new);
            let p_new = &p_n + &v_mid * self.step_size;
            self.solver.set_state(&BodyState {
                lin: (p_new.clone(), v_new.clone()),
                ang: o_new.clone(),
            });

            let (l_lin_np1, l_ang_np1) = self.solver.impulse();
            l_lin_step_end = l_lin_np1.clone();
            l_ang_step_end = l_ang_np1.clone();
            let (mut pressure_force, mut pressure_torque) = if self.impulse_quadratic_pressure {
                self.solver.quadratic_pressure_load()
            } else {
                (
                    DVector::zeros(3 * self.nbody),
                    DVector::zeros(3 * self.nbody),
                )
            };
            let end_state = BodyState {
                lin: (p_new.clone(), v_new.clone()),
                ang: o_new.clone(),
            };
            if !pair_metric_ready {
                let (force, torque_load, active_pairs) =
                    self.impulse_pair_metric_gradient(&start_state, &end_state);
                pair_metric_force = force;
                pair_metric_torque = torque_load;
                self.impulse_pair_metric_last_pairs = active_pairs;
                self.impulse_pair_metric_max_pairs =
                    self.impulse_pair_metric_max_pairs.max(active_pairs);
                pair_metric_ready = true;
            }
            let (mut gradient_force, mut gradient_torque) = if self.fluid_energy_discrete_gradient {
                self.fluid_energy_configuration_discrete_gradient(&start_state, &end_state)
            } else if self.fluid_energy_gradient {
                self.fluid_energy_configuration_gradient(&BodyState {
                    lin: (p_new.clone(), v_new.clone()),
                    ang: o_new.clone(),
                })
            } else {
                (
                    DVector::zeros(3 * self.nbody),
                    DVector::zeros(3 * self.nbody),
                )
            };
            if self.impulse_pair_metric_correction {
                gradient_force += self.impulse_pair_metric_linear_scale * &pair_metric_force;
                gradient_torque += self.impulse_pair_metric_angular_scale * &pair_metric_torque;
                let correction_norm = pair_metric_force.norm() + pair_metric_torque.norm();
                self.impulse_pair_metric_last_norm = correction_norm;
                if correction_norm.is_finite() {
                    self.impulse_pair_metric_max_norm =
                        self.impulse_pair_metric_max_norm.max(correction_norm);
                }
            }
            if self.impulse_internal_load_constraint {
                let p_mid = 0.5 * (&p_n + &p_new);
                self.enforce_internal_load_constraint(
                    &p_mid,
                    &mut gradient_force,
                    &mut gradient_torque,
                );
                self.enforce_internal_load_constraint(
                    &p_mid,
                    &mut pressure_force,
                    &mut pressure_torque,
                );
            }

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
                        - (l_lin_np1[3 * b + c] - l_lin_n[3 * b + c])
                        - self.step_size
                            * self.fluid_energy_gradient_linear_scale
                            * gradient_force[3 * b + c]
                        - self.step_size
                            * self.impulse_quadratic_pressure_scale
                            * pressure_force[3 * b + c];
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
                    new_torque[3 * b + c] = torque_vec[c]
                        + self.fluid_energy_gradient_angular_scale * gradient_torque[3 * b + c]
                        + self.impulse_quadratic_pressure_scale * pressure_torque[3 * b + c];
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
                let r_b = rotation::lab_to_body(&Quaternion::from_imag(r_lab), &q_end[b]).imag();
                let p_mat = Matrix3::identity() * ms + self.added_mass_safe[b];
                let d_b = p_mat.try_inverse().map(|inv| inv * r_b).unwrap_or(r_b / ms);
                let d_lab = rotation::body_to_lab(&Quaternion::from_imag(d_b), &q_end[b]).imag();
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
        self.impulse_fp_steps += 1;
        self.impulse_fp_iter_sum += fp_iters;
        self.impulse_fp_last_iter = fp_iters;
        self.impulse_fp_max_iter = self.impulse_fp_max_iter.max(fp_iters);

        let v_mid = 0.5 * (&v_n + &v_new);
        let p_new = &p_n + &v_mid * self.step_size;
        let x_new = (p_new, v_new);

        self.solver.set_state(&BodyState {
            lin: x_new.clone(),
            ang: o_new.clone(),
        });
        let o_lab_new = self.orientation_to_marker_point();

        self.state.lin = x_new;
        self.state.ang = o_new;
        self.o_lab = o_lab_new;

        if self.impulse_variational_defect_probe {
            let end_state = self.state.clone();
            self.record_impulse_variational_defect_probe(&start_state, &end_state);
            self.prev_variational_state = Some(start_state);
        }

        if self.energy_projection {
            self.project_step_energy_preserving_impulse(ke_step_start, &conserved_step_start);
            self.impulse_next_start_cache = None;
        } else {
            self.impulse_next_start_cache = Some(ImpulseStartCache {
                l_lin: l_lin_step_end,
                l_ang: l_ang_step_end,
                fluid_ke: self.solver.fluid_kinetic_energy(),
            });
        }
    }

    fn advance_one_step_variational(&mut self) {
        self.solver.set_state(&self.state);
        let (l_lin_n, l_ang_n) = self.solver.impulse();
        self.fluid_ke_step_start = self.solver.fluid_kinetic_energy();
        self.record_impulse_partition_drift(
            &self.state.lin.clone(),
            &self.state.ang.clone(),
            &l_lin_n,
            &l_ang_n,
        );
        self.fluid_impulse_lin_step_start = Some(l_lin_n);
        self.fluid_impulse_ang_step_start = Some(l_ang_n);
        self.capture_initial_total_ke();

        let current = self.state.clone();
        let previous = self
            .prev_variational_state
            .clone()
            .unwrap_or_else(|| self.backward_state_from_current_velocity(&current));
        let z0 = self.generalized_velocity_vector();
        let z_mid = self.solve_variational_mid_velocity(&previous, &current, &z0);
        let end_state = self.endpoint_state_from_mid_velocity(&current, &z_mid);
        if self.variational_momentum_diagnostic {
            if let Some(momentum) = self.discrete_noether_momentum(&current, &end_state) {
                self.record_variational_discrete_momentum(momentum);
            } else {
                self.variational_discrete_momentum_out = None;
                self.variational_discrete_momentum_last_drift = f64::NAN;
            }
        } else {
            self.variational_discrete_momentum_out = None;
            self.variational_discrete_momentum_last_drift = f64::NAN;
        }

        self.prev_variational_state = Some(current);
        self.state = end_state;
        self.solver.set_state(&self.state);
        let _ = self.solver.impulse();
        self.o_lab = self.orientation_to_marker_point();
    }

    fn backward_state_from_current_velocity(&self, current: &BodyState) -> BodyState {
        let (pos, vel) = &current.lin;
        let (q, omega) = &current.ang;

        let mut pos_prev = DVector::zeros(3 * self.nbody);
        for i in 0..3 * self.nbody {
            pos_prev[i] = pos[i] - self.step_size * vel[i];
        }

        let mut q_prev = Vec::with_capacity(self.nbody);
        for b in 0..self.nbody {
            q_prev.push(
                self.orientation_stepper(&q[b], &(-omega[b]), self.step_size)
                    .normalize(),
            );
        }

        BodyState {
            lin: (pos_prev, vel.clone()),
            ang: (q_prev, omega.clone()),
        }
    }

    fn solve_variational_mid_velocity(
        &mut self,
        previous: &BodyState,
        current: &BodyState,
        z0: &DVector<f64>,
    ) -> DVector<f64> {
        let mut z = z0.clone();
        let Some(first_leg_gradient) = self.variational_first_leg_gradient(previous, current)
        else {
            return z;
        };
        let Some(mut residual) =
            self.variational_residual_from_first_leg(current, &z, &first_leg_gradient)
        else {
            return z;
        };
        let mut residual_norm = residual.norm();
        let residual_tol = self.hamiltonian_floor_tol.max(1.0e-10);
        let max_dz = if self.hamiltonian_coupled_max_shift > 0.0 {
            self.hamiltonian_coupled_max_shift / self.step_size.max(1.0e-12)
        } else {
            f64::INFINITY
        };
        let dof = z.len();
        let mut reusable_jacobian: Option<DMatrix<f64>> = if self.variational_reuse_step_jacobian {
            self.variational_step_jacobian
                .take()
                .filter(|jacobian| jacobian.nrows() == residual.len() && jacobian.ncols() == dof)
        } else {
            None
        };
        let mut used_full_fallback = false;

        for iter in 0..self.hamiltonian_coupled_iters {
            if residual_norm <= residual_tol {
                break;
            }

            let interval = self.hamiltonian_coupled_jacobian_interval.max(1);
            let has_step_seed =
                self.variational_reuse_step_jacobian && iter == 0 && reusable_jacobian.is_some();
            if reusable_jacobian.is_none() || (!has_step_seed && iter % interval == 0) {
                reusable_jacobian =
                    self.variational_jacobian(current, &z, &residual, &first_leg_gradient);
            }

            let Some(jacobian) = reusable_jacobian.as_ref() else {
                break;
            };
            let rhs = -&residual;
            let mut dz = if let Some(step) = jacobian.clone().lu().solve(&rhs) {
                step
            } else if let Some(step) = jacobian.clone().qr().solve(&rhs) {
                step
            } else {
                break;
            };

            let dz_norm = dz.norm();
            if dz_norm.is_finite() && dz_norm > max_dz {
                dz *= max_dz / dz_norm;
            }

            let mut accepted = false;
            let mut alpha = 1.0;
            let z_before = z.clone();
            let residual_before = residual.clone();
            for _ in 0..8 {
                let z_trial = &z + alpha * &dz;
                let Some(r_trial) = self.variational_residual_from_first_leg(
                    current,
                    &z_trial,
                    &first_leg_gradient,
                ) else {
                    alpha *= 0.5;
                    continue;
                };
                let trial_norm = r_trial.norm();
                if trial_norm.is_finite() && trial_norm < residual_norm {
                    z = z_trial;
                    residual = r_trial;
                    residual_norm = trial_norm;
                    accepted = true;
                    break;
                }
                alpha *= 0.5;
            }

            if !accepted {
                break;
            }

            if self.hamiltonian_coupled_broyden_update {
                if let Some(jacobian) = reusable_jacobian.as_mut() {
                    let step = &z - &z_before;
                    let residual_delta = &residual - &residual_before;
                    Self::broyden_update_jacobian(jacobian, &step, &residual_delta);
                }
            }
        }

        if residual_norm > residual_tol
            && (self.hamiltonian_coupled_broyden_update
                || self.hamiltonian_coupled_jacobian_interval > 1)
        {
            used_full_fallback = true;
            reusable_jacobian = None;
            for _ in 0..self.hamiltonian_coupled_iters {
                if residual_norm <= residual_tol {
                    break;
                }

                let Some(mut jacobian) =
                    self.variational_jacobian(current, &z, &residual, &first_leg_gradient)
                else {
                    break;
                };
                let rhs = -&residual;
                let mut dz = if let Some(step) = jacobian.clone().lu().solve(&rhs) {
                    step
                } else if let Some(step) = jacobian.clone().qr().solve(&rhs) {
                    step
                } else {
                    break;
                };

                let dz_norm = dz.norm();
                if dz_norm.is_finite() && dz_norm > max_dz {
                    dz *= max_dz / dz_norm;
                }

                let mut accepted = false;
                let mut alpha = 1.0;
                for _ in 0..8 {
                    let z_trial = &z + alpha * &dz;
                    let Some(r_trial) = self.variational_residual_from_first_leg(
                        current,
                        &z_trial,
                        &first_leg_gradient,
                    ) else {
                        alpha *= 0.5;
                        continue;
                    };
                    let trial_norm = r_trial.norm();
                    if trial_norm.is_finite() && trial_norm < residual_norm {
                        let z_before = z.clone();
                        let residual_before = residual.clone();
                        z = z_trial;
                        residual = r_trial;
                        residual_norm = trial_norm;
                        if self.hamiltonian_coupled_broyden_update {
                            let step = &z - &z_before;
                            let residual_delta = &residual - &residual_before;
                            Self::broyden_update_jacobian(&mut jacobian, &step, &residual_delta);
                        }
                        reusable_jacobian = Some(jacobian);
                        accepted = true;
                        break;
                    }
                    alpha *= 0.5;
                }

                if !accepted {
                    break;
                }
            }
        }

        if self.variational_reuse_step_jacobian
            && residual_norm <= residual_tol
            && !used_full_fallback
        {
            self.variational_step_jacobian = reusable_jacobian;
        } else if !self.variational_reuse_step_jacobian || residual_norm > residual_tol {
            self.variational_step_jacobian = None;
        }

        self.coupled_max_residual_norm = self.coupled_max_residual_norm.max(residual_norm);
        self.coupled_last_step_residual_norm =
            self.coupled_last_step_residual_norm.max(residual_norm);
        self.record_coupled_correction_diagnostics(current, z0, &z);
        z
    }

    fn variational_jacobian(
        &mut self,
        current: &BodyState,
        z: &DVector<f64>,
        residual: &DVector<f64>,
        first_leg_gradient: &DVector<f64>,
    ) -> Option<DMatrix<f64>> {
        let dof = z.len();
        let mut jac = DMatrix::zeros(residual.len(), dof);
        let eps_base = self.hamiltonian_coupled_eps.abs().max(1.0e-8);
        for j in 0..dof {
            let eps = eps_base * (z[j].abs() + 1.0);
            let mut zp = z.clone();
            zp[j] += eps;
            let rp = self.variational_residual_from_first_leg(current, &zp, first_leg_gradient)?;
            for row in 0..residual.len() {
                jac[(row, j)] = (rp[row] - residual[row]) / eps;
            }
        }
        self.coupled_jacobian_builds += 1;
        self.record_coupled_jacobian_diagnostics(&jac);
        Some(jac)
    }

    fn variational_residual(
        &mut self,
        previous: &BodyState,
        current: &BodyState,
        z_mid: &DVector<f64>,
    ) -> Option<DVector<f64>> {
        let first_leg_gradient = self.variational_first_leg_gradient(previous, current)?;
        self.variational_residual_from_first_leg(current, z_mid, &first_leg_gradient)
    }

    fn variational_first_leg_gradient(
        &mut self,
        previous: &BodyState,
        current: &BodyState,
    ) -> Option<DVector<f64>> {
        let saved_state = self.state.clone();
        let dof = 6 * self.nbody;
        let mut gradient = DVector::zeros(dof);

        for j in 0..dof {
            let eps = self.variational_configuration_eps(current, j);
            let current_plus = self.perturb_configuration(current, j, eps);
            let current_minus = self.perturb_configuration(current, j, -eps);
            let action_plus = self.discrete_lagrangian(previous, &current_plus);
            let action_minus = self.discrete_lagrangian(previous, &current_minus);
            gradient[j] = (action_plus - action_minus) / (2.0 * eps);
            if !gradient[j].is_finite() {
                self.restore_variational_probe_state(&saved_state);
                return None;
            }
        }

        self.restore_variational_probe_state(&saved_state);
        Some(gradient)
    }

    fn variational_residual_from_first_leg(
        &mut self,
        current: &BodyState,
        z_mid: &DVector<f64>,
        first_leg_gradient: &DVector<f64>,
    ) -> Option<DVector<f64>> {
        let saved_state = self.state.clone();
        let endpoint = self.endpoint_state_from_mid_velocity(current, z_mid);
        let dof = 6 * self.nbody;
        let mut residual = first_leg_gradient.clone();

        for j in 0..dof {
            let eps = self.variational_configuration_eps(current, j);
            let current_plus = self.perturb_configuration(current, j, eps);
            let current_minus = self.perturb_configuration(current, j, -eps);
            let action_plus = self.discrete_lagrangian(&current_plus, &endpoint);
            let action_minus = self.discrete_lagrangian(&current_minus, &endpoint);
            residual[j] += (action_plus - action_minus) / (2.0 * eps);
            if !residual[j].is_finite() {
                self.restore_variational_probe_state(&saved_state);
                return None;
            }
        }

        self.restore_variational_probe_state(&saved_state);
        Some(residual)
    }

    fn variational_configuration_eps(&self, state: &BodyState, dof: usize) -> f64 {
        let base = self.hamiltonian_coupled_eps.abs().max(1.0e-6);
        if dof < 3 * self.nbody {
            base * (state.lin.0[dof].abs() + 1.0)
        } else {
            base
        }
    }

    fn perturb_configuration(&self, state: &BodyState, dof: usize, eps: f64) -> BodyState {
        let mut out = state.clone();
        if dof < 3 * self.nbody {
            out.lin.0[dof] += eps;
            return out;
        }

        let local = dof - 3 * self.nbody;
        let b = local / 3;
        let c = local % 3;
        let mut axis = Vector3::zeros();
        axis[c] = eps;
        let dq = UnitQuaternion::from_scaled_axis(axis);
        let q = UnitQuaternion::from_quaternion(out.ang.0[b].normalize());
        out.ang.0[b] = (dq * q).into_inner().normalize();
        out
    }

    fn discrete_lagrangian(&mut self, a: &BodyState, b: &BodyState) -> f64 {
        let midpoint = self.midpoint_state_from_endpoint(a, b);
        self.solver.set_state(&midpoint);
        let fluid_ke = if self.variational_energy_only_lagrangian {
            self.solver.kinetic_energy_only()
        } else {
            let _ = self.solver.impulse();
            self.solver.fluid_kinetic_energy()
        };
        let ke = self
            .solid_kinetic_energy(&midpoint.lin, &midpoint.ang)
            .total
            + fluid_ke;
        self.step_size * ke
    }

    fn restore_variational_probe_state(&mut self, saved_state: &BodyState) {
        self.state = saved_state.clone();
        self.solver.set_state(&self.state);
        let _ = self.solver.impulse();
    }

    fn record_variational_discrete_momentum(&mut self, momentum: DVector<f64>) {
        if self.variational_discrete_momentum_initial.is_none() {
            self.variational_discrete_momentum_initial = Some(momentum.clone());
        }
        let drift = self
            .variational_discrete_momentum_initial
            .as_ref()
            .map(|initial| (&momentum - initial).norm())
            .unwrap_or(f64::NAN);
        if drift.is_finite() {
            self.variational_discrete_momentum_max_drift =
                self.variational_discrete_momentum_max_drift.max(drift);
        }
        self.variational_discrete_momentum_last_drift = drift;
        self.variational_discrete_momentum_out = Some(momentum);
    }

    fn discrete_noether_momentum(&mut self, a: &BodyState, b: &BodyState) -> Option<DVector<f64>> {
        let saved_state = self.state.clone();
        let eps = self.hamiltonian_coupled_eps.abs().max(1.0e-6);
        let mut out = DVector::zeros(6);

        for axis in 0..3 {
            let plus = self.discrete_lagrangian(
                a,
                &self.perturb_second_endpoint_by_translation(b, axis, eps),
            );
            let minus = self.discrete_lagrangian(
                a,
                &self.perturb_second_endpoint_by_translation(b, axis, -eps),
            );
            out[axis] = (plus - minus) / (2.0 * eps);
        }

        for axis in 0..3 {
            let plus = self
                .discrete_lagrangian(a, &self.perturb_second_endpoint_by_rotation(b, axis, eps));
            let minus = self
                .discrete_lagrangian(a, &self.perturb_second_endpoint_by_rotation(b, axis, -eps));
            out[3 + axis] = (plus - minus) / (2.0 * eps);
        }

        self.restore_variational_probe_state(&saved_state);

        if out.iter().all(|v| v.is_finite()) {
            Some(out)
        } else {
            None
        }
    }

    fn perturb_second_endpoint_by_translation(
        &self,
        state: &BodyState,
        axis: usize,
        eps: f64,
    ) -> BodyState {
        let mut out = state.clone();
        for b in 0..self.nbody {
            out.lin.0[3 * b + axis] += eps;
        }
        out
    }

    fn perturb_second_endpoint_by_rotation(
        &self,
        state: &BodyState,
        axis: usize,
        eps: f64,
    ) -> BodyState {
        let mut out = state.clone();
        let mut scaled_axis = Vector3::zeros();
        scaled_axis[axis] = eps;
        let dq = UnitQuaternion::from_scaled_axis(scaled_axis);
        let rot = dq.to_rotation_matrix();

        for b in 0..self.nbody {
            let p = Vector3::new(out.lin.0[3 * b], out.lin.0[3 * b + 1], out.lin.0[3 * b + 2]);
            let p_rot = rot * p;
            for c in 0..3 {
                out.lin.0[3 * b + c] = p_rot[c];
            }

            let q = UnitQuaternion::from_quaternion(out.ang.0[b].normalize());
            out.ang.0[b] = (dq * q).into_inner().normalize();
        }

        out
    }

    fn record_impulse_variational_defect_probe(&mut self, start: &BodyState, end: &BodyState) {
        let previous = self
            .prev_variational_state
            .clone()
            .unwrap_or_else(|| self.backward_state_from_current_velocity(start));
        let midpoint = self.midpoint_state_from_endpoint(start, end);
        let z_mid = self.generalized_velocity_vector_from_state(&midpoint);
        let Some(residual) = self.variational_residual(&previous, start, &z_mid) else {
            self.impulse_variational_defect_out = false;
            self.impulse_variational_defect_last_norm = f64::NAN;
            self.impulse_variational_defect_last_metric_cos = f64::NAN;
            self.impulse_variational_defect_last_metric_scale = f64::NAN;
            self.impulse_variational_defect_last_pressure_cos = f64::NAN;
            self.impulse_variational_defect_last_pressure_scale = f64::NAN;
            return;
        };

        // Forced DEL would be R + dt Q = 0, so this is the generalized
        // configuration force that would make the accepted impulse endpoint
        // variationally balanced for this step.
        let target = -residual / self.step_size.max(1.0e-12);
        let target_norm = target.norm();

        let (mut metric_force, mut metric_torque) =
            self.fluid_energy_configuration_discrete_gradient(start, end);
        self.solver.set_state(end);
        let _ = self.solver.impulse();
        let (mut pressure_force, mut pressure_torque) = self.solver.quadratic_pressure_load();

        if self.impulse_internal_load_constraint {
            self.enforce_internal_load_constraint(
                &midpoint.lin.0,
                &mut metric_force,
                &mut metric_torque,
            );
            self.enforce_internal_load_constraint(
                &midpoint.lin.0,
                &mut pressure_force,
                &mut pressure_torque,
            );
        }

        let metric = self.generalized_load_vector(&metric_force, &metric_torque);
        let pressure = self.generalized_load_vector(&pressure_force, &pressure_torque);
        let (metric_cos, metric_scale) = Self::vector_alignment(&target, &metric);
        let (pressure_cos, pressure_scale) = Self::vector_alignment(&target, &pressure);

        self.impulse_variational_defect_probe_count += 1;
        self.impulse_variational_defect_last_norm = target_norm;
        if target_norm.is_finite() {
            self.impulse_variational_defect_max_norm =
                self.impulse_variational_defect_max_norm.max(target_norm);
        }
        self.impulse_variational_defect_last_metric_cos = metric_cos;
        self.impulse_variational_defect_last_metric_scale = metric_scale;
        self.impulse_variational_defect_last_pressure_cos = pressure_cos;
        self.impulse_variational_defect_last_pressure_scale = pressure_scale;
        self.impulse_variational_defect_out = true;

        self.solver.set_state(end);
        let _ = self.solver.impulse();
    }

    fn generalized_load_vector(&self, force: &DVector<f64>, torque: &DVector<f64>) -> DVector<f64> {
        let mut out = DVector::zeros(6 * self.nbody);
        for i in 0..(3 * self.nbody) {
            out[i] = force[i];
            out[3 * self.nbody + i] = torque[i];
        }
        out
    }

    fn vector_alignment(target: &DVector<f64>, candidate: &DVector<f64>) -> (f64, f64) {
        let target_norm = target.norm();
        let candidate_norm = candidate.norm();
        if target_norm <= 1.0e-14 || candidate_norm <= 1.0e-14 {
            return (f64::NAN, f64::NAN);
        }
        let dot = target.dot(candidate);
        let cos = dot / (target_norm * candidate_norm);
        let scale = dot / candidate.norm_squared();
        (cos, scale)
    }

    fn enforce_internal_load_constraint(
        &self,
        pos: &DVector<f64>,
        force: &mut DVector<f64>,
        torque: &mut DVector<f64>,
    ) {
        if self.nbody == 0 {
            return;
        }

        // Enforce Noether consistency on a candidate internal fluid load:
        // global translations and rotations of the whole system must do no
        // generalized work. This is a constrained least-change correction to a
        // force covector, not a post-step energy projection.
        let dof = 6 * self.nbody;
        let mut load = self.generalized_load_vector(force, torque);
        let mut constraint = DMatrix::zeros(6, dof);
        for b in 0..self.nbody {
            let x = Vector3::new(pos[3 * b], pos[3 * b + 1], pos[3 * b + 2]);
            for c in 0..3 {
                constraint[(c, 3 * b + c)] = 1.0;
                constraint[(3 + c, 3 * self.nbody + 3 * b + c)] = 1.0;
            }
            constraint[(3, 3 * b + 1)] = -x[2];
            constraint[(3, 3 * b + 2)] = x[1];
            constraint[(4, 3 * b)] = x[2];
            constraint[(4, 3 * b + 2)] = -x[0];
            constraint[(5, 3 * b)] = -x[1];
            constraint[(5, 3 * b + 1)] = x[0];
        }

        let rhs = &constraint * &load;
        let gram = &constraint * constraint.transpose();
        let Some(lambda) = gram.lu().solve(&rhs) else {
            return;
        };
        load -= constraint.transpose() * lambda;

        for i in 0..(3 * self.nbody) {
            force[i] = load[i];
            torque[i] = load[3 * self.nbody + i];
        }
    }

    /// Experimental Hamiltonian/discrete-gradient endpoint solve.
    ///
    /// This treats the BEM as a black-box energy/momentum oracle at the endpoint
    /// configuration. At each iteration it:
    /// 1. projects endpoint generalized velocities to the step-start total
    ///    conserved impulse and the run's initial kinetic energy at the current
    ///    endpoint q,
    /// 2. rebuilds the endpoint positions/orientations from midpoint
    ///    kinematics using those projected velocities.
    ///
    /// It is intentionally slower than the impulse scheme but avoids injecting a
    /// raw finite-difference force into the equations.
    fn advance_one_step_hamiltonian(&mut self) {
        let substeps = self.hamiltonian_substeps.max(1);
        if substeps == 1 {
            self.advance_one_hamiltonian_substep();
            return;
        }

        let (l_lin_start, l_ang_start) = self.solver.impulse();
        let fluid_ke_start = self.solver.fluid_kinetic_energy();
        let step_save = self.step_size;
        let half_save = self.half_step;
        let quarter_save = self.quarter_step;
        let h = step_save / substeps as f64;
        self.step_size = h;
        self.half_step = 0.5 * h;
        self.quarter_step = 0.25 * h;

        for _ in 0..substeps {
            self.advance_one_hamiltonian_substep();
        }

        self.step_size = step_save;
        self.half_step = half_save;
        self.quarter_step = quarter_save;
        self.fluid_ke_step_start = fluid_ke_start;
        self.record_impulse_partition_drift(
            &self.state.lin.clone(),
            &self.state.ang.clone(),
            &l_lin_start,
            &l_ang_start,
        );
        self.fluid_impulse_lin_step_start = Some(l_lin_start);
        self.fluid_impulse_ang_step_start = Some(l_ang_start);
    }

    fn advance_one_hamiltonian_substep(&mut self) {
        let (l_lin_n, l_ang_n) = self.solver.impulse();
        self.fluid_ke_step_start = self.solver.fluid_kinetic_energy();
        self.record_impulse_partition_drift(
            &self.state.lin.clone(),
            &self.state.ang.clone(),
            &l_lin_n,
            &l_ang_n,
        );
        self.fluid_impulse_lin_step_start = Some(l_lin_n.clone());
        self.fluid_impulse_ang_step_start = Some(l_ang_n.clone());
        self.capture_initial_total_ke();

        let current_ke = self
            .solid_kinetic_energy(&self.state.lin, &self.state.ang)
            .total
            + self.fluid_ke_step_start;
        let target_ke = self.initial_total_ke.unwrap_or(current_ke);
        let (p_n, v_n) = self.state.lin.clone();
        let start_state = self.state.clone();
        let target_impulse =
            self.total_conserved_impulse(&p_n, &v_n, &self.state.ang, &l_lin_n, &l_ang_n);

        let mut z = self.generalized_velocity_vector();
        self.state = self.endpoint_state_from_velocity(&start_state, &z);
        self.solver.set_state(&self.state);
        let _ = self.solver.impulse();

        for _ in 0..HAMILTONIAN_MAXITER {
            let z_before = self.generalized_velocity_vector();
            self.project_step_energy_preserving_impulse(target_ke, &target_impulse);
            let z_projected = self.generalized_velocity_vector();
            let next_state = self.endpoint_state_from_velocity(&start_state, &z_projected);
            let dz = (&z_projected - &z_before).norm();

            self.state = next_state;
            self.solver.set_state(&self.state);
            let _ = self.solver.impulse();
            z = z_projected;

            if dz < 1e-10 * (z.norm() + 1.0) {
                break;
            }
        }

        // One last projection at the final endpoint configuration. Leave that
        // projected endpoint in place; rebuilding q once more from the newly
        // projected velocities would move the configuration after enforcing the
        // energy/momentum constraints.
        self.project_step_energy_preserving_impulse(target_ke, &target_impulse);
        self.solver.set_state(&self.state);
        let _ = self.solver.impulse();
        self.o_lab = self.orientation_to_marker_point();
    }

    fn endpoint_state_from_velocity(&self, start: &BodyState, z: &DVector<f64>) -> BodyState {
        let (p_n, v_n) = &start.lin;
        let (q_n, omega_n) = &start.ang;

        let mut v_end = DVector::zeros(3 * self.nbody);
        for i in 0..3 * self.nbody {
            v_end[i] = z[i];
        }

        let mut p_end = DVector::zeros(3 * self.nbody);
        for i in 0..3 * self.nbody {
            p_end[i] = p_n[i] + 0.5 * (v_n[i] + v_end[i]) * self.step_size;
        }

        let mut q_end = Vec::with_capacity(self.nbody);
        let mut omega_end = Vec::with_capacity(self.nbody);
        let omega_offset = 3 * self.nbody;
        for b in 0..self.nbody {
            let w_end = Vector3::new(
                z[omega_offset + 3 * b],
                z[omega_offset + 3 * b + 1],
                z[omega_offset + 3 * b + 2],
            );
            let omega_end_q = Quaternion::from_imag(w_end);
            let omega_mid = Quaternion::from_imag(0.5 * (omega_n[b].imag() + w_end));
            let q_full = self.orientation_stepper(&q_n[b], &omega_mid, self.step_size);
            q_end.push(q_full.normalize());
            omega_end.push(omega_end_q);
        }

        BodyState {
            lin: (p_end, v_end),
            ang: (q_end, omega_end),
        }
    }

    fn advance_one_step_hamiltonian_midpoint(&mut self) {
        if self.hamiltonian_adaptive_substeps {
            self.advance_one_step_hamiltonian_midpoint_adaptive();
        } else {
            self.advance_one_step_hamiltonian_midpoint_fixed(self.hamiltonian_substeps.max(1));
        }
    }

    fn advance_one_step_hamiltonian_midpoint_adaptive(&mut self) {
        let base_substeps = self.hamiltonian_substeps.max(1);
        let max_substeps = self.hamiltonian_max_substeps.max(base_substeps);
        let start_state = self.state.clone();
        let start_marker = self.o_lab.clone();
        let start_projection_diag = self.projection_diagnostics_snapshot();
        let start_coupled_diag = self.coupled_diagnostics_snapshot();
        let mut substeps = base_substeps;

        loop {
            self.state = start_state.clone();
            self.o_lab = start_marker.clone();
            self.solver.set_state(&self.state);
            let _ = self.solver.impulse();
            self.restore_projection_diagnostics(start_projection_diag);
            self.restore_coupled_diagnostics(start_coupled_diag);

            self.advance_one_step_hamiltonian_midpoint_fixed(substeps);

            let coupled_retry =
                self.coupled_last_step_residual_norm > self.hamiltonian_floor_tol.max(1.0e-12);
            let retry = self.projection_last_step_max_floor_rel > self.hamiltonian_floor_tol
                || self.projection_last_step_floor_fallbacks > 0
                || coupled_retry;
            if !retry || substeps >= max_substeps {
                break;
            }

            let next_substeps = (2 * substeps).min(max_substeps);
            println!(
                "  Hamiltonian adaptive retry | t {:.6} | substeps {} -> {} | floor {:.6e} | fallbacks {} | coupled residual {:.6e}",
                self.t + self.step_size,
                substeps,
                next_substeps,
                self.projection_last_step_max_floor_rel,
                self.projection_last_step_floor_fallbacks,
                self.coupled_last_step_residual_norm
            );
            self.hamiltonian_adaptive_retry_count += 1;
            substeps = next_substeps;
        }
    }

    fn advance_one_step_hamiltonian_midpoint_fixed(&mut self, substeps: usize) {
        let substeps = substeps.max(1);
        self.hamiltonian_max_substeps_used = self.hamiltonian_max_substeps_used.max(substeps);
        self.coupled_last_step_residual_norm = 0.0;
        self.coupled_last_step_impulse_resid = 0.0;
        self.coupled_last_step_energy_err_rel = 0.0;
        self.coupled_last_step_raw_linear_impulse_resid = 0.0;
        self.coupled_last_step_raw_angular_impulse_resid = 0.0;
        self.coupled_last_step_true_energy_err_rel = 0.0;
        self.coupled_last_step_correction_rel = 0.0;
        self.coupled_last_step_correction_kinetic_rel = 0.0;
        if substeps == 1 {
            self.advance_one_hamiltonian_midpoint_substep();
            return;
        }

        let (l_lin_start, l_ang_start) = self.solver.impulse();
        let fluid_ke_start = self.solver.fluid_kinetic_energy();
        let step_save = self.step_size;
        let half_save = self.half_step;
        let quarter_save = self.quarter_step;
        let h = step_save / substeps as f64;
        self.step_size = h;
        self.half_step = 0.5 * h;
        self.quarter_step = 0.25 * h;

        for _ in 0..substeps {
            self.advance_one_hamiltonian_midpoint_substep();
        }

        self.step_size = step_save;
        self.half_step = half_save;
        self.quarter_step = quarter_save;
        self.fluid_ke_step_start = fluid_ke_start;
        self.record_impulse_partition_drift(
            &self.state.lin.clone(),
            &self.state.ang.clone(),
            &l_lin_start,
            &l_ang_start,
        );
        self.fluid_impulse_lin_step_start = Some(l_lin_start);
        self.fluid_impulse_ang_step_start = Some(l_ang_start);
    }

    fn advance_one_hamiltonian_midpoint_substep(&mut self) {
        let (l_lin_n, l_ang_n) = self.solver.impulse();
        self.fluid_ke_step_start = self.solver.fluid_kinetic_energy();
        self.record_impulse_partition_drift(
            &self.state.lin.clone(),
            &self.state.ang.clone(),
            &l_lin_n,
            &l_ang_n,
        );
        self.fluid_impulse_lin_step_start = Some(l_lin_n.clone());
        self.fluid_impulse_ang_step_start = Some(l_ang_n.clone());
        self.capture_initial_total_ke();

        let current_ke = self
            .solid_kinetic_energy(&self.state.lin, &self.state.ang)
            .total
            + self.fluid_ke_step_start;
        let target_ke = self.initial_total_ke.unwrap_or(current_ke);
        let (p_n, v_n) = self.state.lin.clone();
        let start_state = self.state.clone();
        let target_impulse =
            self.total_conserved_impulse(&p_n, &v_n, &self.state.ang, &l_lin_n, &l_ang_n);

        if self.hamiltonian_coupled_solve {
            let z0 = self.generalized_velocity_vector();
            let z_mid = self.solve_coupled_endpoint_mid_velocity(
                &start_state,
                &z0,
                target_ke,
                &target_impulse,
            );
            self.state = self.coupled_endpoint_state(&start_state, &z_mid);
            self.solver.set_state(&self.state);
            let _ = self.solver.impulse();
            self.o_lab = self.orientation_to_marker_point();
            return;
        }

        let mut z_mid = self.generalized_velocity_vector();
        let mut end_state = self.endpoint_state_from_mid_velocity(&start_state, &z_mid);

        for _ in 0..HAMILTONIAN_MAXITER {
            let mid_state = self.midpoint_state_from_endpoint(&start_state, &end_state);
            self.state = mid_state;
            self.solver.set_state(&self.state);
            let _ = self.solver.impulse();

            self.project_step_energy_preserving_impulse(target_ke, &target_impulse);
            z_mid = self.generalized_velocity_vector();
            let next_end = self.endpoint_state_from_mid_velocity(&start_state, &z_mid);
            let residual = self.configuration_distance(&end_state, &next_end);
            end_state = next_end;

            if residual < 1e-10 * (z_mid.norm() + 1.0) {
                break;
            }
        }

        self.state = end_state;
        self.solver.set_state(&self.state);
        let _ = self.solver.impulse();
        // Store endpoint velocities that satisfy the endpoint global invariants
        // when possible, so diagnostics/output report the same energy convention
        // as the other schemes.
        self.project_step_energy_preserving_impulse(target_ke, &target_impulse);
        self.solver.set_state(&self.state);
        let _ = self.solver.impulse();
        self.o_lab = self.orientation_to_marker_point();
    }

    fn solve_coupled_endpoint_mid_velocity(
        &mut self,
        start: &BodyState,
        z0: &DVector<f64>,
        target_ke: f64,
        target_impulse: &DVector<f64>,
    ) -> DVector<f64> {
        let mut z = z0.clone();
        let Some(mut residual) =
            self.coupled_endpoint_residual(start, &z, target_ke, target_impulse)
        else {
            return z;
        };
        let mut residual_norm = residual.norm();
        let residual_tol = self.hamiltonian_floor_tol.max(1.0e-12);
        let max_dz = if self.hamiltonian_coupled_max_shift > 0.0 {
            self.hamiltonian_coupled_max_shift / self.step_size.max(1.0e-12)
        } else {
            f64::INFINITY
        };
        let mut reusable_jacobian: Option<DMatrix<f64>> = None;

        for iter in 0..self.hamiltonian_coupled_iters {
            if residual_norm <= residual_tol {
                break;
            }

            let interval = self.hamiltonian_coupled_jacobian_interval.max(1);
            if reusable_jacobian.is_none() || iter % interval == 0 {
                reusable_jacobian =
                    self.coupled_endpoint_jacobian(start, &z, &residual, target_ke, target_impulse);
            }

            let Some(jacobian) = reusable_jacobian.as_ref() else {
                break;
            };
            let Some(dz) = self.coupled_newton_step(start, jacobian, &residual, max_dz) else {
                break;
            };

            let mut accepted = false;
            let mut alpha = 1.0;
            let z_before = z.clone();
            let residual_before = residual.clone();
            for _ in 0..8 {
                let z_trial = &z + alpha * &dz;
                let Some(r_trial) =
                    self.coupled_endpoint_residual(start, &z_trial, target_ke, target_impulse)
                else {
                    alpha *= 0.5;
                    continue;
                };
                let trial_norm = r_trial.norm();
                if trial_norm.is_finite() && trial_norm < residual_norm {
                    z = z_trial;
                    residual = r_trial;
                    residual_norm = trial_norm;
                    accepted = true;
                    break;
                }
                alpha *= 0.5;
            }

            if !accepted {
                break;
            }

            if self.hamiltonian_coupled_broyden_update {
                if let Some(jacobian) = reusable_jacobian.as_mut() {
                    let step = &z - &z_before;
                    let residual_delta = &residual - &residual_before;
                    Self::broyden_update_jacobian(jacobian, &step, &residual_delta);
                }
            }
        }

        if residual_norm > residual_tol
            && (self.hamiltonian_coupled_broyden_update
                || self.hamiltonian_coupled_jacobian_interval > 1)
        {
            for _ in 0..self.hamiltonian_coupled_iters {
                if residual_norm <= residual_tol {
                    break;
                }

                let Some(jacobian) =
                    self.coupled_endpoint_jacobian(start, &z, &residual, target_ke, target_impulse)
                else {
                    break;
                };
                let Some(dz) = self.coupled_newton_step(start, &jacobian, &residual, max_dz) else {
                    break;
                };

                let mut accepted = false;
                let mut alpha = 1.0;
                for _ in 0..8 {
                    let z_trial = &z + alpha * &dz;
                    let Some(r_trial) =
                        self.coupled_endpoint_residual(start, &z_trial, target_ke, target_impulse)
                    else {
                        alpha *= 0.5;
                        continue;
                    };
                    let trial_norm = r_trial.norm();
                    if trial_norm.is_finite() && trial_norm < residual_norm {
                        z = z_trial;
                        residual = r_trial;
                        residual_norm = trial_norm;
                        accepted = true;
                        break;
                    }
                    alpha *= 0.5;
                }

                if !accepted {
                    break;
                }
            }
        }

        if let Some(eval) =
            self.coupled_endpoint_residual_eval(start, &z, target_ke, target_impulse)
        {
            residual = eval.scaled;
            residual_norm = residual.norm();
            self.coupled_max_raw_linear_impulse_resid = self
                .coupled_max_raw_linear_impulse_resid
                .max(eval.raw_linear_impulse_resid);
            self.coupled_max_raw_angular_impulse_resid = self
                .coupled_max_raw_angular_impulse_resid
                .max(eval.raw_angular_impulse_resid);
            self.coupled_max_true_energy_err_rel = self
                .coupled_max_true_energy_err_rel
                .max(eval.true_energy_err_rel);
            self.coupled_last_step_raw_linear_impulse_resid = self
                .coupled_last_step_raw_linear_impulse_resid
                .max(eval.raw_linear_impulse_resid);
            self.coupled_last_step_raw_angular_impulse_resid = self
                .coupled_last_step_raw_angular_impulse_resid
                .max(eval.raw_angular_impulse_resid);
            self.coupled_last_step_true_energy_err_rel = self
                .coupled_last_step_true_energy_err_rel
                .max(eval.true_energy_err_rel);
        }

        let impulse_resid = residual.rows(0, 6).norm();
        let energy_err_rel = residual[6].abs();
        self.coupled_max_residual_norm = self.coupled_max_residual_norm.max(residual_norm);
        self.coupled_max_impulse_resid = self.coupled_max_impulse_resid.max(impulse_resid);
        self.coupled_max_energy_err_rel = self.coupled_max_energy_err_rel.max(energy_err_rel);
        self.coupled_last_step_residual_norm =
            self.coupled_last_step_residual_norm.max(residual_norm);
        self.coupled_last_step_impulse_resid =
            self.coupled_last_step_impulse_resid.max(impulse_resid);
        self.coupled_last_step_energy_err_rel =
            self.coupled_last_step_energy_err_rel.max(energy_err_rel);
        self.record_coupled_correction_diagnostics(start, z0, &z);

        z
    }

    fn coupled_endpoint_jacobian(
        &mut self,
        start: &BodyState,
        z: &DVector<f64>,
        residual: &DVector<f64>,
        target_ke: f64,
        target_impulse: &DVector<f64>,
    ) -> Option<DMatrix<f64>> {
        let dof = z.len();
        let mut jac = DMatrix::zeros(residual.len(), dof);
        for j in 0..dof {
            let eps = self.hamiltonian_coupled_eps * (z[j].abs() + 1.0);
            let mut zp = z.clone();
            zp[j] += eps;
            let rp = self.coupled_endpoint_residual(start, &zp, target_ke, target_impulse)?;
            for row in 0..residual.len() {
                jac[(row, j)] = (rp[row] - residual[row]) / eps;
            }
        }
        self.coupled_jacobian_builds += 1;
        self.record_coupled_jacobian_diagnostics(&jac);
        Some(jac)
    }

    fn coupled_newton_step(
        &self,
        start: &BodyState,
        jacobian: &DMatrix<f64>,
        residual: &DVector<f64>,
        max_dz: f64,
    ) -> Option<DVector<f64>> {
        let metric_inv = self.coupled_metric_inverse(start);
        let mut gram = if let Some(metric_inv) = metric_inv.as_ref() {
            jacobian * metric_inv * jacobian.transpose()
        } else {
            jacobian * jacobian.transpose()
        };
        for i in 0..gram.nrows() {
            gram[(i, i)] += 1.0e-10;
        }
        let gram_inv = gram.try_inverse()?;
        let y = gram_inv * residual;
        let mut dz = if let Some(metric_inv) = metric_inv.as_ref() {
            -(metric_inv * jacobian.transpose() * y)
        } else {
            -(jacobian.transpose() * y)
        };
        let dz_norm = dz.norm();
        if dz_norm.is_finite() && dz_norm > max_dz {
            dz *= max_dz / dz_norm;
        }
        Some(dz)
    }

    fn coupled_metric_inverse(&self, state: &BodyState) -> Option<DMatrix<f64>> {
        if !self.hamiltonian_coupled_kinetic_metric {
            return None;
        }

        let dof = 6 * self.nbody;
        let mut metric_inv = DMatrix::zeros(dof, dof);
        let (q, _) = &state.ang;
        let omega_offset = 3 * self.nbody;

        for b in 0..self.nbody {
            let mass = self.masses[b];
            let inv_mass = if mass > 1.0e-14 { 1.0 / mass } else { 1.0 };
            for c in 0..3 {
                metric_inv[(3 * b + c, 3 * b + c)] = inv_mass;
            }

            let q_unit = UnitQuaternion::from_quaternion(q[b].normalize());
            let rot = q_unit.to_rotation_matrix();
            let r = rot.matrix();
            let inertia_lab = r * self.inertias[b] * r.transpose();
            let inertia_inv = inertia_lab.try_inverse().unwrap_or_else(Matrix3::identity);
            for row in 0..3 {
                for col in 0..3 {
                    metric_inv[(omega_offset + 3 * b + row, omega_offset + 3 * b + col)] =
                        inertia_inv[(row, col)];
                }
            }
        }

        Some(metric_inv)
    }

    fn velocity_metric_norm(&self, state: &BodyState, z: &DVector<f64>) -> f64 {
        let (q, _) = &state.ang;
        let mut qform = 0.0;
        let omega_offset = 3 * self.nbody;
        for b in 0..self.nbody {
            let v = Vector3::new(z[3 * b], z[3 * b + 1], z[3 * b + 2]);
            qform += self.masses[b].max(0.0) * v.dot(&v);

            let omega_lab = Quaternion::from_imag(Vector3::new(
                z[omega_offset + 3 * b],
                z[omega_offset + 3 * b + 1],
                z[omega_offset + 3 * b + 2],
            ));
            let omega_body = rotation::lab_to_body(&omega_lab, &q[b]).imag();
            qform += omega_body.dot(&(self.inertias[b] * omega_body));
        }
        qform.max(0.0).sqrt()
    }

    fn record_coupled_correction_diagnostics(
        &mut self,
        start: &BodyState,
        z0: &DVector<f64>,
        z: &DVector<f64>,
    ) {
        let dz = z - z0;
        let rel = dz.norm() / z0.norm().max(1.0);
        let kinetic_rel =
            self.velocity_metric_norm(start, &dz) / self.velocity_metric_norm(start, z0).max(1.0);

        self.coupled_max_correction_rel = self.coupled_max_correction_rel.max(rel);
        self.coupled_last_step_correction_rel = self.coupled_last_step_correction_rel.max(rel);
        self.coupled_max_correction_kinetic_rel =
            self.coupled_max_correction_kinetic_rel.max(kinetic_rel);
        self.coupled_last_step_correction_kinetic_rel = self
            .coupled_last_step_correction_kinetic_rel
            .max(kinetic_rel);
    }

    fn record_coupled_jacobian_diagnostics(&mut self, jacobian: &DMatrix<f64>) {
        let svd = jacobian.clone().svd(false, false);
        let mut sigma_max: f64 = 0.0;
        let mut sigma_min = f64::INFINITY;
        for sigma in svd.singular_values.iter().copied() {
            if sigma.is_finite() {
                sigma_max = sigma_max.max(sigma);
                sigma_min = sigma_min.min(sigma);
            }
        }
        if !sigma_min.is_finite() {
            return;
        }

        let tol = (1.0e-10 * sigma_max).max(1.0e-12);
        let rank = svd
            .singular_values
            .iter()
            .filter(|sigma| sigma.is_finite() && **sigma > tol)
            .count();
        let nullity = jacobian.ncols().saturating_sub(rank);

        self.coupled_min_jacobian_rank = self.coupled_min_jacobian_rank.min(rank);
        self.coupled_max_jacobian_nullity = self.coupled_max_jacobian_nullity.max(nullity);
        self.coupled_min_jacobian_sigma = self.coupled_min_jacobian_sigma.min(sigma_min);
    }

    fn broyden_update_jacobian(
        jacobian: &mut DMatrix<f64>,
        step: &DVector<f64>,
        residual_delta: &DVector<f64>,
    ) {
        let denom = step.dot(step);
        if !denom.is_finite() || denom <= 1.0e-24 {
            return;
        }
        let predicted = &*jacobian * step;
        let correction = residual_delta - predicted;
        for row in 0..jacobian.nrows() {
            for col in 0..jacobian.ncols() {
                jacobian[(row, col)] += correction[row] * step[col] / denom;
            }
        }
    }

    fn coupled_endpoint_residual(
        &mut self,
        start: &BodyState,
        z: &DVector<f64>,
        target_ke: f64,
        target_impulse: &DVector<f64>,
    ) -> Option<DVector<f64>> {
        Some(
            self.coupled_endpoint_residual_eval(start, z, target_ke, target_impulse)?
                .scaled,
        )
    }

    fn coupled_endpoint_residual_eval(
        &mut self,
        start: &BodyState,
        z: &DVector<f64>,
        target_ke: f64,
        target_impulse: &DVector<f64>,
    ) -> Option<CoupledResidualEval> {
        let lin_scale = target_impulse.rows(0, 3).norm().max(1.0);
        let ang_scale = target_impulse.rows(3, 3).norm().max(1.0);
        let energy_scale = target_ke.abs().max(1.0);
        let saved_state = self.state.clone();
        self.state = self.coupled_endpoint_state(start, z);
        self.solver.set_state(&self.state);
        let (l_lin, l_ang) = self.solver.impulse();
        let conserved = self.total_conserved_impulse(
            &self.state.lin.0,
            &self.state.lin.1,
            &self.state.ang,
            &l_lin,
            &l_ang,
        );
        let ke = self
            .solid_kinetic_energy(&self.state.lin, &self.state.ang)
            .total
            + self.solver.fluid_kinetic_energy();

        self.state = saved_state;
        self.solver.set_state(&self.state);
        let _ = self.solver.impulse();

        if !ke.is_finite() {
            return None;
        }
        let mut residual = DVector::zeros(7);
        let mut lin_resid = [0.0; 3];
        let mut ang_resid = [0.0; 3];
        for c in 0..3 {
            lin_resid[c] = conserved[c] - target_impulse[c];
            ang_resid[c] = conserved[3 + c] - target_impulse[3 + c];
            residual[c] = lin_resid[c] / lin_scale;
            residual[3 + c] = ang_resid[c] / ang_scale;
        }
        residual[6] = (ke - target_ke) / energy_scale;
        Some(CoupledResidualEval {
            scaled: residual,
            raw_linear_impulse_resid: norm3(lin_resid),
            raw_angular_impulse_resid: norm3(ang_resid),
            true_energy_err_rel: (ke - target_ke).abs() / target_ke.abs().max(f64::EPSILON),
        })
    }

    fn coupled_endpoint_state(&self, start: &BodyState, z: &DVector<f64>) -> BodyState {
        if self.hamiltonian_coupled_endpoint_velocity {
            self.endpoint_state_from_velocity(start, z)
        } else {
            self.endpoint_state_from_mid_velocity(start, z)
        }
    }

    fn endpoint_state_from_mid_velocity(
        &self,
        start: &BodyState,
        z_mid: &DVector<f64>,
    ) -> BodyState {
        let (p_n, _) = &start.lin;
        let (q_n, _) = &start.ang;
        let mut v_mid = DVector::zeros(3 * self.nbody);
        for i in 0..3 * self.nbody {
            v_mid[i] = z_mid[i];
        }

        let mut p_end = DVector::zeros(3 * self.nbody);
        for i in 0..3 * self.nbody {
            p_end[i] = p_n[i] + v_mid[i] * self.step_size;
        }

        let omega_offset = 3 * self.nbody;
        let mut q_end = Vec::with_capacity(self.nbody);
        let mut omega_mid = Vec::with_capacity(self.nbody);
        for b in 0..self.nbody {
            let w_mid = Vector3::new(
                z_mid[omega_offset + 3 * b],
                z_mid[omega_offset + 3 * b + 1],
                z_mid[omega_offset + 3 * b + 2],
            );
            let w_mid_q = Quaternion::from_imag(w_mid);
            q_end.push(
                self.orientation_stepper(&q_n[b], &w_mid_q, self.step_size)
                    .normalize(),
            );
            omega_mid.push(w_mid_q);
        }

        BodyState {
            lin: (p_end, v_mid),
            ang: (q_end, omega_mid),
        }
    }

    fn midpoint_state_from_endpoint(&self, start: &BodyState, end: &BodyState) -> BodyState {
        let (p_n, _) = &start.lin;
        let (p_end, _) = &end.lin;
        let (q_n, _) = &start.ang;
        let (q_end, _) = &end.ang;

        let mut p_mid = DVector::zeros(3 * self.nbody);
        let mut v_mid = DVector::zeros(3 * self.nbody);
        for i in 0..3 * self.nbody {
            p_mid[i] = 0.5 * (p_n[i] + p_end[i]);
            v_mid[i] = (p_end[i] - p_n[i]) / self.step_size;
        }

        let mut q_mid = Vec::with_capacity(self.nbody);
        let mut omega_mid = Vec::with_capacity(self.nbody);
        for b in 0..self.nbody {
            let w_mid = self.midpoint_omega_from_endpoint(&q_n[b], &q_end[b]);
            let w_mid_q = Quaternion::from_imag(w_mid);
            q_mid.push(
                self.orientation_stepper(&q_n[b], &w_mid_q, self.half_step)
                    .normalize(),
            );
            omega_mid.push(w_mid_q);
        }

        BodyState {
            lin: (p_mid, v_mid),
            ang: (q_mid, omega_mid),
        }
    }

    fn midpoint_omega_from_endpoint(
        &self,
        q_start: &Quaternion<f64>,
        q_end: &Quaternion<f64>,
    ) -> Vector3<f64> {
        let Some(q_inv) = q_start.try_inverse() else {
            return Vector3::repeat(f64::NAN);
        };
        let mut dq = q_end * q_inv;
        if dq.w < 0.0 {
            dq = -dq;
        }
        let v = dq.vector();
        let s = v.norm();
        if s <= 1.0e-14 {
            return Vector3::zeros();
        }
        let angle = 2.0 * s.atan2(dq.w);
        v * (angle / (s * self.step_size))
    }

    fn configuration_distance(&self, a: &BodyState, b: &BodyState) -> f64 {
        let pos_dist = (&a.lin.0 - &b.lin.0).norm();
        let (qa, _) = &a.ang;
        let (qb, _) = &b.ang;
        let mut rot_dist2 = 0.0;
        for i in 0..self.nbody {
            let dot = qa[i].coords.dot(&qb[i].coords).abs().min(1.0);
            let angle = 2.0 * dot.acos();
            rot_dist2 += angle * angle;
        }
        pos_dist + rot_dist2.sqrt()
    }

    fn fluid_ke_for_state(&mut self, state: &BodyState) -> f64 {
        self.solver.set_state(state);
        let _ = self.solver.impulse();
        self.solver.fluid_kinetic_energy()
    }

    fn fluid_ke_only_for_state(&mut self, state: &BodyState) -> f64 {
        self.solver.set_state(state);
        self.solver.kinetic_energy_only()
    }

    fn state_on_config_path(
        &self,
        start: &BodyState,
        pos: &DVector<f64>,
        eta: &[Vector3<f64>],
        vel_mid: &DVector<f64>,
        omega_mid: &[Quaternion<f64>],
    ) -> BodyState {
        let q_start = &start.ang.0;
        let mut q = Vec::with_capacity(self.nbody);
        for b in 0..self.nbody {
            let q0 = UnitQuaternion::from_quaternion(q_start[b].normalize());
            let dq = UnitQuaternion::from_scaled_axis(eta[b]);
            q.push((dq * q0).into_inner());
        }

        BodyState {
            lin: (pos.clone(), vel_mid.clone()),
            ang: (q, omega_mid.to_vec()),
        }
    }

    fn fluid_energy_configuration_discrete_gradient(
        &mut self,
        start: &BodyState,
        end: &BodyState,
    ) -> (DVector<f64>, DVector<f64>) {
        let (pos_start, vel_start) = &start.lin;
        let (pos_end, vel_end) = &end.lin;
        let (q_start, omega_start) = &start.ang;
        let (q_end, omega_end) = &end.ang;

        let vel_mid = 0.5 * (vel_start + vel_end);
        let mut omega_mid = Vec::with_capacity(self.nbody);
        for b in 0..self.nbody {
            omega_mid.push(Quaternion::from_imag(
                0.5 * (omega_start[b].imag() + omega_end[b].imag()),
            ));
        }

        let mut eta_total = Vec::with_capacity(self.nbody);
        for b in 0..self.nbody {
            let q0 = UnitQuaternion::from_quaternion(q_start[b].normalize());
            let q1 = UnitQuaternion::from_quaternion(q_end[b].normalize());
            eta_total.push((q1 * q0.inverse()).scaled_axis());
        }

        let eps = self.fluid_energy_gradient_eps.abs().max(1.0e-6);
        let ndof = 6 * self.nbody;
        let mut delta = DVector::zeros(ndof);
        let mut pos_mid = DVector::zeros(3 * self.nbody);
        let mut eta_mid = vec![Vector3::zeros(); self.nbody];
        for i in 0..(3 * self.nbody) {
            delta[i] = pos_end[i] - pos_start[i];
            pos_mid[i] = 0.5 * (pos_start[i] + pos_end[i]);
        }
        for b in 0..self.nbody {
            eta_mid[b] = 0.5 * eta_total[b];
            for c in 0..3 {
                delta[3 * self.nbody + 3 * b + c] = eta_total[b][c];
            }
        }

        let eta_zero = vec![Vector3::zeros(); self.nbody];
        let state_a = self.state_on_config_path(start, pos_start, &eta_zero, &vel_mid, &omega_mid);
        let state_b = self.state_on_config_path(start, pos_end, &eta_total, &vel_mid, &omega_mid);
        let ke_a = self.fluid_ke_for_state(&state_a);
        let ke_b = self.fluid_ke_for_state(&state_b);

        let mut grad_mid = DVector::zeros(ndof);
        for dof in 0..(3 * self.nbody) {
            let mut p_plus = pos_mid.clone();
            p_plus[dof] += eps;
            let state_plus =
                self.state_on_config_path(start, &p_plus, &eta_mid, &vel_mid, &omega_mid);
            let ke_plus = self.fluid_ke_for_state(&state_plus);

            let mut p_minus = pos_mid.clone();
            p_minus[dof] -= eps;
            let state_minus =
                self.state_on_config_path(start, &p_minus, &eta_mid, &vel_mid, &omega_mid);
            let ke_minus = self.fluid_ke_for_state(&state_minus);
            grad_mid[dof] = (ke_plus - ke_minus) / (2.0 * eps);
        }

        for b in 0..self.nbody {
            for c in 0..3 {
                let idx = 3 * self.nbody + 3 * b + c;
                let mut eta_plus = eta_mid.clone();
                eta_plus[b][c] += eps;
                let state_plus =
                    self.state_on_config_path(start, &pos_mid, &eta_plus, &vel_mid, &omega_mid);
                let ke_plus = self.fluid_ke_for_state(&state_plus);

                let mut eta_minus = eta_mid.clone();
                eta_minus[b][c] -= eps;
                let state_minus =
                    self.state_on_config_path(start, &pos_mid, &eta_minus, &vel_mid, &omega_mid);
                let ke_minus = self.fluid_ke_for_state(&state_minus);
                grad_mid[idx] = (ke_plus - ke_minus) / (2.0 * eps);
            }
        }

        let denom = delta.dot(&delta);
        let grad = if denom > 1.0e-24 {
            let defect = (ke_b - ke_a) - grad_mid.dot(&delta);
            grad_mid + delta * (defect / denom)
        } else {
            grad_mid
        };

        let mut force = DVector::zeros(3 * self.nbody);
        let mut torque = DVector::zeros(3 * self.nbody);
        for i in 0..(3 * self.nbody) {
            force[i] = grad[i];
            torque[i] = grad[3 * self.nbody + i];
        }

        self.solver.set_state(end);
        let _ = self.solver.impulse();
        (force, torque)
    }

    fn fluid_energy_configuration_gradient(
        &mut self,
        state: &BodyState,
    ) -> (DVector<f64>, DVector<f64>) {
        let eps = self.fluid_energy_gradient_eps.abs().max(1.0e-6);
        let (pos, vel) = &state.lin;
        let (q, omega) = &state.ang;
        let mut force = DVector::zeros(3 * self.nbody);
        let mut torque = DVector::zeros(3 * self.nbody);

        for dof in 0..(3 * self.nbody) {
            let mut p_plus = pos.clone();
            p_plus[dof] += eps;
            self.solver.set_state(&BodyState {
                lin: (p_plus, vel.clone()),
                ang: state.ang.clone(),
            });
            let _ = self.solver.impulse();
            let ke_plus = self.solver.fluid_kinetic_energy();

            let mut p_minus = pos.clone();
            p_minus[dof] -= eps;
            self.solver.set_state(&BodyState {
                lin: (p_minus, vel.clone()),
                ang: state.ang.clone(),
            });
            let _ = self.solver.impulse();
            let ke_minus = self.solver.fluid_kinetic_energy();

            force[dof] = (ke_plus - ke_minus) / (2.0 * eps);
        }

        let axes = [
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ];
        for b in 0..self.nbody {
            for c in 0..3 {
                let axis = Unit::new_normalize(axes[c]);
                let dq_plus = UnitQuaternion::from_axis_angle(&axis, eps);
                let dq_minus = UnitQuaternion::from_axis_angle(&axis, -eps);

                let mut q_plus = q.clone();
                q_plus[b] = (dq_plus * UnitQuaternion::from_quaternion(q[b])).into_inner();
                self.solver.set_state(&BodyState {
                    lin: (pos.clone(), vel.clone()),
                    ang: (q_plus, omega.clone()),
                });
                let _ = self.solver.impulse();
                let ke_plus = self.solver.fluid_kinetic_energy();

                let mut q_minus = q.clone();
                q_minus[b] = (dq_minus * UnitQuaternion::from_quaternion(q[b])).into_inner();
                self.solver.set_state(&BodyState {
                    lin: (pos.clone(), vel.clone()),
                    ang: (q_minus, omega.clone()),
                });
                let _ = self.solver.impulse();
                let ke_minus = self.solver.fluid_kinetic_energy();

                torque[3 * b + c] = (ke_plus - ke_minus) / (2.0 * eps);
            }
        }

        self.solver.set_state(state);
        let _ = self.solver.impulse();
        (force, torque)
    }

    fn impulse_pair_metric_gradient(
        &mut self,
        start: &BodyState,
        end: &BodyState,
    ) -> (DVector<f64>, DVector<f64>, usize) {
        let mut force = DVector::zeros(3 * self.nbody);
        let mut torque = DVector::zeros(3 * self.nbody);
        if self.nbody < 2 {
            return (force, torque, 0);
        }

        let midpoint = self.midpoint_state_from_endpoint(start, end);
        let (pos_start, _) = &start.lin;
        let (pos_end, _) = &end.lin;
        let (pos_mid, vel_mid) = &midpoint.lin;
        let (q_mid, omega_mid) = &midpoint.ang;
        let eps = self.impulse_pair_metric_eps.abs().max(1.0e-6);
        let axes = [
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ];
        let mut active_pairs = 0usize;

        for a in 0..self.nbody {
            let xa = Vector3::new(pos_mid[3 * a], pos_mid[3 * a + 1], pos_mid[3 * a + 2]);
            for b in (a + 1)..self.nbody {
                let xb = Vector3::new(pos_mid[3 * b], pos_mid[3 * b + 1], pos_mid[3 * b + 2]);
                let sep = (xb - xa).norm();
                let weight = self.impulse_pair_metric_weight(sep);
                if weight <= 0.0 {
                    continue;
                }
                active_pairs += 1;

                match self.impulse_pair_metric_mode {
                    PairMetricMode::PointGradient => {
                        let h_pos = eps * sep.max(1.0);
                        for c in 0..3 {
                            let mut p_plus = pos_mid.clone();
                            p_plus[3 * a + c] -= 0.5 * h_pos;
                            p_plus[3 * b + c] += 0.5 * h_pos;
                            let state_plus = BodyState {
                                lin: (p_plus, vel_mid.clone()),
                                ang: (q_mid.clone(), omega_mid.clone()),
                            };
                            let ke_plus = self.fluid_ke_only_for_state(&state_plus);

                            let mut p_minus = pos_mid.clone();
                            p_minus[3 * a + c] += 0.5 * h_pos;
                            p_minus[3 * b + c] -= 0.5 * h_pos;
                            let state_minus = BodyState {
                                lin: (p_minus, vel_mid.clone()),
                                ang: (q_mid.clone(), omega_mid.clone()),
                            };
                            let ke_minus = self.fluid_ke_only_for_state(&state_minus);
                            let grad = weight * (ke_plus - ke_minus) / (2.0 * h_pos);
                            force[3 * a + c] -= grad;
                            force[3 * b + c] += grad;
                        }
                    }
                    PairMetricMode::TranslationalDiscreteGradient => {
                        let r_start = Vector3::new(
                            pos_start[3 * b] - pos_start[3 * a],
                            pos_start[3 * b + 1] - pos_start[3 * a + 1],
                            pos_start[3 * b + 2] - pos_start[3 * a + 2],
                        );
                        let r_end = Vector3::new(
                            pos_end[3 * b] - pos_end[3 * a],
                            pos_end[3 * b + 1] - pos_end[3 * a + 1],
                            pos_end[3 * b + 2] - pos_end[3 * a + 2],
                        );
                        let delta_r = r_end - r_start;
                        let denom = delta_r.dot(&delta_r);
                        if denom > 1.0e-24 {
                            let state_start_pair = self.pair_relative_translation_state(
                                &midpoint,
                                a,
                                b,
                                r_start - (xb - xa),
                            );
                            let ke_start = self.fluid_ke_only_for_state(&state_start_pair);
                            let state_end_pair = self.pair_relative_translation_state(
                                &midpoint,
                                a,
                                b,
                                r_end - (xb - xa),
                            );
                            let ke_end = self.fluid_ke_only_for_state(&state_end_pair);
                            let grad_vec = weight * (ke_end - ke_start) / denom * delta_r;
                            for c in 0..3 {
                                force[3 * a + c] -= grad_vec[c];
                                force[3 * b + c] += grad_vec[c];
                            }
                        }
                    }
                }

                if self.impulse_pair_metric_angular_scale == 0.0 {
                    continue;
                }
                for c in 0..3 {
                    let h_rot = eps;
                    let dq_plus_b = UnitQuaternion::from_scaled_axis(0.5 * h_rot * axes[c]);
                    let dq_minus_b = UnitQuaternion::from_scaled_axis(-0.5 * h_rot * axes[c]);
                    let dq_plus_a = UnitQuaternion::from_scaled_axis(-0.5 * h_rot * axes[c]);
                    let dq_minus_a = UnitQuaternion::from_scaled_axis(0.5 * h_rot * axes[c]);

                    let mut q_plus = q_mid.clone();
                    q_plus[a] = (dq_plus_a * UnitQuaternion::from_quaternion(q_mid[a].normalize()))
                        .into_inner()
                        .normalize();
                    q_plus[b] = (dq_plus_b * UnitQuaternion::from_quaternion(q_mid[b].normalize()))
                        .into_inner()
                        .normalize();
                    let state_plus = BodyState {
                        lin: (pos_mid.clone(), vel_mid.clone()),
                        ang: (q_plus, omega_mid.clone()),
                    };
                    let ke_plus = self.fluid_ke_only_for_state(&state_plus);

                    let mut q_minus = q_mid.clone();
                    q_minus[a] = (dq_minus_a
                        * UnitQuaternion::from_quaternion(q_mid[a].normalize()))
                    .into_inner()
                    .normalize();
                    q_minus[b] = (dq_minus_b
                        * UnitQuaternion::from_quaternion(q_mid[b].normalize()))
                    .into_inner()
                    .normalize();
                    let state_minus = BodyState {
                        lin: (pos_mid.clone(), vel_mid.clone()),
                        ang: (q_minus, omega_mid.clone()),
                    };
                    let ke_minus = self.fluid_ke_only_for_state(&state_minus);
                    let grad = weight * (ke_plus - ke_minus) / (2.0 * h_rot);
                    torque[3 * a + c] -= grad;
                    torque[3 * b + c] += grad;
                }
            }
        }

        self.solver.set_state(end);
        let _ = self.solver.kinetic_energy_only();
        (force, torque, active_pairs)
    }

    fn pair_relative_translation_state(
        &self,
        state: &BodyState,
        a: usize,
        b: usize,
        delta_relative: Vector3<f64>,
    ) -> BodyState {
        let mut out = state.clone();
        for c in 0..3 {
            out.lin.0[3 * a + c] -= 0.5 * delta_relative[c];
            out.lin.0[3 * b + c] += 0.5 * delta_relative[c];
        }
        out
    }

    fn impulse_pair_metric_weight(&self, sep: f64) -> f64 {
        let legacy_cutoff = self.impulse_pair_metric_cutoff;
        let mut inner = self.impulse_pair_metric_inner_cutoff;
        let mut outer = self.impulse_pair_metric_outer_cutoff;
        if inner <= 0.0 && outer <= 0.0 && legacy_cutoff <= 0.0 {
            return 1.0;
        }
        if inner <= 0.0 {
            inner = legacy_cutoff;
        }
        if outer <= 0.0 {
            outer = legacy_cutoff;
        }
        if outer < inner {
            std::mem::swap(&mut inner, &mut outer);
        }
        if outer <= 0.0 {
            return 1.0;
        }
        if sep <= inner {
            return 1.0;
        }
        if sep >= outer {
            return 0.0;
        }
        let s = (sep - inner) / (outer - inner).max(1.0e-12);
        let smooth = s * s * (3.0 - 2.0 * s);
        1.0 - smooth
    }

    fn project_step_energy_preserving_impulse(&mut self, target_ke: f64, target: &DVector<f64>) {
        if !target_ke.is_finite() || target_ke <= 0.0 {
            return;
        }

        let z_current = self.generalized_velocity_vector();
        let pre_ke = self.evaluate_total_ke_for_velocity(&z_current);
        if pre_ke.is_finite() {
            let energy_err_rel = ((pre_ke - target_ke) / target_ke).abs();
            self.projection_max_energy_err_rel =
                self.projection_max_energy_err_rel.max(energy_err_rel);
        }

        let Some((z_pcon, z_particular, z_null)) =
            self.project_velocity_to_conserved_impulse(&z_current, target)
        else {
            return;
        };
        let energy_floor = self.evaluate_total_ke_for_velocity(&z_particular);
        if energy_floor.is_finite() {
            let floor_abs = (energy_floor - target_ke).max(0.0);
            let floor_rel = floor_abs / target_ke.abs().max(f64::EPSILON);
            self.projection_max_energy_floor_rel =
                self.projection_max_energy_floor_rel.max(floor_rel);
            self.projection_max_energy_floor_abs =
                self.projection_max_energy_floor_abs.max(floor_abs);
            if floor_abs > 1.0e-10 * target_ke.abs().max(1.0) {
                self.projection_floor_hit_count += 1;
                self.projection_last_step_floor_hits += 1;
                self.projection_last_step_max_floor_rel =
                    self.projection_last_step_max_floor_rel.max(floor_rel);
                self.projection_last_step_max_floor_abs =
                    self.projection_last_step_max_floor_abs.max(floor_abs);
            }
        }
        let z_projected = match self.project_velocity_to_energy_in_nullspace(
            target_ke,
            &z_pcon,
            &z_particular,
            &z_null,
        ) {
            Some(z) => z,
            None => {
                self.projection_floor_fallback_count += 1;
                self.projection_last_step_floor_fallbacks += 1;
                z_particular
            }
        };
        let corr_rel = (&z_projected - &z_current).norm() / z_current.norm().max(1.0);
        if corr_rel.is_finite() {
            self.projection_max_corr_rel = self.projection_max_corr_rel.max(corr_rel);
        }
        self.set_generalized_velocity_vector(&z_projected);
        self.solver.set_state(&self.state);
        let _ = self.solver.impulse();

        let constraint_resid = (&self.evaluate_conserved_impulse(&z_projected) - target).norm();
        if constraint_resid.is_finite() {
            self.projection_max_constraint_resid =
                self.projection_max_constraint_resid.max(constraint_resid);
        }
    }

    fn project_velocity_to_conserved_impulse(
        &mut self,
        z_current: &DVector<f64>,
        target: &DVector<f64>,
    ) -> Option<(DVector<f64>, DVector<f64>, DVector<f64>)> {
        let (a, c_zero) = self.conserved_impulse_operator();
        let gram = &a * a.transpose();
        let gram_inv = gram.try_inverse()?;

        let c_current = self.evaluate_conserved_impulse(z_current);
        let delta_p = a.transpose() * (&gram_inv * (target - c_current));
        let z_pcon = z_current + delta_p;

        let mut z_particular = if self.projection_kinetic_metric {
            self.minimum_energy_impulse_solution(&a, &c_zero, target)
                .unwrap_or_else(|| a.transpose() * (&gram_inv * (target - c_zero.clone())))
        } else {
            a.transpose() * (&gram_inv * (target - c_zero.clone()))
        };
        let impulse_resid = target - (&a * &z_particular + &c_zero);
        z_particular += a.transpose() * (&gram_inv * impulse_resid);
        let z_null = &z_pcon - &z_particular;
        Some((z_pcon, z_particular, z_null))
    }

    fn minimum_energy_impulse_solution(
        &mut self,
        a: &DMatrix<f64>,
        c_zero: &DVector<f64>,
        target: &DVector<f64>,
    ) -> Option<DVector<f64>> {
        let k = self.energy_quadratic_matrix();
        let mut k_reg = k.clone();
        for i in 0..k_reg.nrows() {
            k_reg[(i, i)] += 1.0e-12;
        }
        let k_inv = k_reg.try_inverse()?;
        let gram = a * &k_inv * a.transpose();
        let gram_inv = gram.try_inverse()?;
        Some(&k_inv * a.transpose() * (gram_inv * (target - c_zero)))
    }

    fn energy_quadratic_matrix(&mut self) -> DMatrix<f64> {
        let dof = 6 * self.nbody;
        let z_save = self.generalized_velocity_vector();
        let z_zero = DVector::zeros(dof);
        let e_zero = self.evaluate_total_ke_for_velocity(&z_zero);

        let mut basis_energy = vec![0.0; dof];
        for i in 0..dof {
            let mut z = DVector::zeros(dof);
            z[i] = 1.0;
            basis_energy[i] = self.evaluate_total_ke_for_velocity(&z);
        }

        let mut k = DMatrix::zeros(dof, dof);
        for i in 0..dof {
            k[(i, i)] = 2.0 * (basis_energy[i] - e_zero);
            for j in (i + 1)..dof {
                let mut z = DVector::zeros(dof);
                z[i] = 1.0;
                z[j] = 1.0;
                let e_ij = self.evaluate_total_ke_for_velocity(&z);
                let kij = e_ij - basis_energy[i] - basis_energy[j] + e_zero;
                k[(i, j)] = kij;
                k[(j, i)] = kij;
            }
        }

        self.set_generalized_velocity_vector(&z_save);
        self.solver.set_state(&self.state);
        let _ = self.solver.impulse();
        k
    }

    fn project_velocity_to_energy_in_nullspace(
        &mut self,
        target_ke: f64,
        z_pcon: &DVector<f64>,
        z_particular: &DVector<f64>,
        z_null: &DVector<f64>,
    ) -> Option<DVector<f64>> {
        if z_null.norm() <= 1e-14 {
            return None;
        }
        let e0 = self.evaluate_total_ke_for_velocity(&z_particular);
        let e1 = self.evaluate_total_ke_for_velocity(&z_pcon);
        let e_minus = self.evaluate_total_ke_for_velocity(&(z_particular - z_null));
        let qa = 0.5 * (e1 + e_minus) - e0;
        let qb = 0.5 * (e1 - e_minus);
        let qc = e0 - target_ke;

        let alpha = if qa.abs() <= 1e-14 {
            if qb.abs() <= 1e-14 {
                return None;
            }
            -qc / qb
        } else {
            let disc = qb * qb - 4.0 * qa * qc;
            if disc < 0.0 {
                return None;
            }
            let sqrt_disc = disc.sqrt();
            let r1 = (-qb + sqrt_disc) / (2.0 * qa);
            let r2 = (-qb - sqrt_disc) / (2.0 * qa);
            if (r1 - 1.0).abs() <= (r2 - 1.0).abs() {
                r1
            } else {
                r2
            }
        };

        if !alpha.is_finite() {
            return None;
        }

        Some(z_particular + z_null * alpha)
    }

    fn generalized_velocity_vector(&self) -> DVector<f64> {
        self.generalized_velocity_vector_from_state(&self.state)
    }

    fn generalized_velocity_vector_from_state(&self, state: &BodyState) -> DVector<f64> {
        let (_, vel) = &state.lin;
        let (_, omega) = &state.ang;
        let mut out = DVector::zeros(6 * self.nbody);
        for i in 0..3 * self.nbody {
            out[i] = vel[i];
        }
        for b in 0..self.nbody {
            let w = omega[b].imag();
            for c in 0..3 {
                out[3 * self.nbody + 3 * b + c] = w[c];
            }
        }
        out
    }

    fn set_generalized_velocity_vector(&mut self, z: &DVector<f64>) {
        let (_, vel) = &mut self.state.lin;
        for i in 0..3 * self.nbody {
            vel[i] = z[i];
        }
        let (_, omega) = &mut self.state.ang;
        for b in 0..self.nbody {
            omega[b] = Quaternion::new(
                0.0,
                z[3 * self.nbody + 3 * b],
                z[3 * self.nbody + 3 * b + 1],
                z[3 * self.nbody + 3 * b + 2],
            );
        }
    }

    fn evaluate_total_ke_for_velocity(&mut self, z: &DVector<f64>) -> f64 {
        self.set_generalized_velocity_vector(z);
        self.solver.set_state(&self.state);
        let _ = self.solver.impulse();
        self.solid_kinetic_energy(&self.state.lin, &self.state.ang)
            .total
            + self.solver.fluid_kinetic_energy()
    }

    fn evaluate_conserved_impulse(&mut self, z: &DVector<f64>) -> DVector<f64> {
        self.set_generalized_velocity_vector(z);
        self.solver.set_state(&self.state);
        let (l_lin, l_ang) = self.solver.impulse();
        let (pos, vel) = &self.state.lin;
        self.total_conserved_impulse(pos, vel, &self.state.ang, &l_lin, &l_ang)
    }

    fn conserved_impulse_operator(&mut self) -> (DMatrix<f64>, DVector<f64>) {
        let dof = 6 * self.nbody;
        let z_save = self.generalized_velocity_vector();
        let z_zero = DVector::zeros(dof);
        let c_zero = self.evaluate_conserved_impulse(&z_zero);
        let mut a = DMatrix::zeros(6, dof);
        for j in 0..dof {
            let mut z = DVector::zeros(dof);
            z[j] = 1.0;
            let c_eval = self.evaluate_conserved_impulse(&z);
            for row in 0..6 {
                a[(row, j)] = c_eval[row] - c_zero[row];
            }
        }
        self.set_generalized_velocity_vector(&z_save);
        self.solver.set_state(&self.state);
        let _ = self.solver.impulse();
        (a, c_zero)
    }

    fn total_conserved_impulse(
        &self,
        pos: &DVector<f64>,
        vel: &DVector<f64>,
        ang: &AngularState,
        l_lin: &DVector<f64>,
        l_ang: &DVector<f64>,
    ) -> DVector<f64> {
        let (q, omega_lab) = ang;
        let mut p_total = Vector3::zeros();
        let mut h_total = Vector3::zeros();
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
            p_total += p_con;
            h_total += h_con;
        }

        let mut out = DVector::zeros(6);
        for c in 0..3 {
            out[c] = p_total[c];
            out[3 + c] = h_total[c];
        }
        out
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
            self.solver.set_state(&BodyState {
                lin: (p_half, v_half.clone()),
                ang: (q_half.clone(), o_half.clone()),
            });
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
                let r_b = rotation::lab_to_body(&Quaternion::from_imag(r_lab), &q_half[b]).imag();
                let p_mat = Matrix3::identity() + self.added_mass_safe[b] / (2.0 * ms);
                let d_b = p_mat.try_inverse().map(|inv| inv * r_b).unwrap_or(r_b);
                let d_lab = rotation::body_to_lab(&Quaternion::from_imag(d_b), &q_half[b]).imag();
                for c in 0..3 {
                    v_half[3 * b + c] -= d_lab[c];
                }
            }
        }
        self.solver.set_freeze(false);

        // Commit stage B: evaluate at the converged v_half to push φ(v_half).
        let p_half = &p_n + &v_half * self.half_step;
        self.solver.set_state(&BodyState {
            lin: (p_half, v_half.clone()),
            ang: (q_half.clone(), o_half.clone()),
        });
        let (_a_final, torque_half) = self.solver.force();

        // Implicit-midpoint update: v_{n+1} = 2 v_half - v_n, x uses v_half.
        let v_new = 2.0 * &v_half - &v_n;
        let p_new = &p_n + &v_half * self.step_size;
        let x_new = (p_new, v_new);
        let o_new = self.ang_full_step(&torque_half, &(q_half, o_half));

        self.solver.set_state(&BodyState {
            lin: x_new.clone(),
            ang: o_new.clone(),
        });
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
            let a_expl_b = rotation::lab_to_body(&Quaternion::from_imag(a_expl_lab), &q[i]).imag();
            let a_prev_b = rotation::lab_to_body(&Quaternion::from_imag(a_prev_lab), &q[i]).imag();

            let ms_i = Matrix3::identity() * ms;
            let m_a = self.added_mass_safe[i];
            let lhs = ms_i + m_a;
            let rhs = ms_i * a_expl_b + m_a * a_prev_b;
            let a_stab_b = lhs.try_inverse().map(|inv| inv * rhs).unwrap_or(a_expl_b);

            // Rotate body -> lab.
            let a_stab_lab = rotation::body_to_lab(&Quaternion::from_imag(a_stab_b), &q[i]).imag();
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
                    rotation::accel_get(&Quaternion::from_imag(omega_mid_b), &inertia, &n_mid_b)
                        .imag();
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
            let omega_full_lab =
                rotation::body_to_lab(&Quaternion::from_imag(omega_full_b), &q_full);

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

fn norm3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}
