// use std::f64::consts::PI;
// use std::rc::Rc;

// use std::ops::{Deref, DerefMut};
// use csv;
use crate::ellipsoids::body::Body;
use nalgebra as na;

// use crate::solver::symplectic;
// use crate::solver::symplectic::Hamiltonian;
use crate::system::fluid::Fluid;
// use crate::system::hamiltonian::*;
// use crate::system::state::State;

/// Number of provisional first-step passes used to bootstrap the ∂φ/∂t
/// history at t = 0. The dphi force needs temporal φ history that doesn't
/// exist at startup (the initial acceleration and φ̇ are mutually dependent
/// through the added mass), so the integrator runs the first step this many
/// times, rewinding the state after each pass; every pass refines the initial
/// φ̇ estimate by a geometric (added-mass contraction) factor. The integrator
/// loop count and the `bootstrap_redos` bookkeeping in ForceCalculate must
/// both use this constant.
pub const BOOTSTRAP_PASSES: usize = 10;

#[derive(Clone)]
pub struct Simulation {
    pub fluid: Fluid,
    /// All rigid bodies in the simulation. `nbody == bodies.len()`.
    pub bodies: Vec<Body>,
    pub ndiv: u32,
    pub nbody: usize,
    /// Per-body solid mass tensors (index i <-> bodies[i]).
    pub mass_tensors: Vec<na::Matrix3<f64>>,
    /// Per-body solid inertia tensors (index i <-> bodies[i]).
    pub inertia_tensors: Vec<na::Matrix3<f64>>,
    /// Surface-potential history for the same-stage BDF2 ∂φ/∂t stencil.
    /// Each force evaluation pushes its solved φ; entries are previous calls,
    /// most recent last. Bounded to the last 4 calls.
    pub phi_history: std::collections::VecDeque<na::DVector<f64>>,
    /// Remaining first-step bootstrap passes (see [`BOOTSTRAP_PASSES`]). While
    /// positive, a force call that sees exactly two history entries treats them
    /// as the previous provisional pass over [t0, t0+dt/2] and seeds φ̇ from
    /// their forward difference, then clears the history for the next pass.
    pub bootstrap_redos: usize,
    /// Input key `impulse_transport` (default ON): add the rotating-frame
    /// impulse transport terms ω×L_lin (force) and ω×L_ang (torque) that the
    /// exact Lamb-impulse rate dL/dt carries for a rotating body, which the
    /// per-element ∂φ/∂t force omits. Validated against the exact Kirchhoff
    /// reference: with these terms (and the corrected angular_velocity()) the
    /// coupled force converges to the exact value with mesh refinement
    /// (0.84× at ndiv=3, 0.92× at ndiv=4); without them it is ~1.9× too large.
    /// Set `impulse_transport=0` to reproduce the old incomplete force.
    pub impulse_transport: bool,
    /// Opt-in (input key `added_mass_stab`): enable the semi-implicit
    /// added-mass-partitioned (Robin) velocity update in the integrator, which
    /// stabilises the explicit added-mass reaction force at fine meshes
    /// (ndiv=4). Off by default; when off the integrator path is byte-identical
    /// to the explicit scheme.
    pub added_mass_stab: bool,
    /// Optional blend of the BDF2 ∂φ/∂t stencil toward the 1st-order same-stage
    /// difference (input key `phidot_blend`, eps in [0,1]): lowers the BDF2
    /// high-frequency stencil gain (4/dt) toward BDF1's (2/dt) to damp the
    /// fine-mesh explicit added-mass instability. 0 = pure BDF2 (default).
    pub phidot_blend: f64,
    pub step_dt: f64,
    /// Prototype B (input key `strong_couple`): strong (implicit-midpoint)
    /// fluid-structure coupling. The integrator iterates the linear velocity to
    /// consistency with the re-solved BEM force each step, making the added-mass
    /// reaction implicit (cures the explicit added-mass oscillation/instability
    /// without dropping order). When set, the integrator freezes the φ history
    /// during the trial evaluations (see `freeze_phi_history`).
    pub strong_couple: bool,
    /// Set by the integrator during strong-coupling trial force evaluations so
    /// ForceCalculate computes φ̇ from the committed history WITHOUT pushing the
    /// trial φ (preserves the 2-push-per-step BDF2 pattern).
    pub freeze_phi_history: bool,
    /// Approach A (input key `impulse_scheme`): when set, ForceCalculate returns
    /// the lab-frame fluid impulse (L_lin, L_ang) per body instead of the ∂φ/∂t
    /// force, and does not touch the φ history. The integrator differences the
    /// impulse to build an energy-consistent F = -dL/dt.
    pub impulse_mode: bool,
}

impl Simulation {
    pub fn new(fluid: Fluid, bodies: Vec<Body>, ndiv: u32) -> Self {
        let nbody = bodies.len();

        let mass_tensors = bodies.iter().map(|b| b.m_i_tensor().0).collect();
        let inertia_tensors = bodies.iter().map(|b| b.m_i_tensor().1).collect();

        Self {
            fluid,
            bodies,
            ndiv,
            nbody,
            mass_tensors,
            inertia_tensors,
            phi_history: std::collections::VecDeque::new(),
            bootstrap_redos: BOOTSTRAP_PASSES,
            impulse_transport: true,
            added_mass_stab: false,
            phidot_blend: 0.0,
            step_dt: 0.01,
            strong_couple: false,
            freeze_phi_history: false,
            impulse_mode: false,
        }
    }

    // pub fn make_hamiltonian1(&self) -> (Hamiltonian, na::Matrix3<f64>, na::Matrix3<f64>) {
    //     let rho_f = self.fluid.density;
    //
    //     let m_s = na::Matrix3::from_diagonal(&self.body1.shape);
    //     let v_s = 4.0 / 3.0 * PI * self.body1.shape.iter().fold(1.0, |acc, x| acc * x);
    //
    //     let lambda = 10000.0;
    //
    //     let shape_fun = calc_shape_factor(lambda, m_s).unwrap();
    //
    //     let alpha = shape_fun.alpha;
    //     let beta = shape_fun.beta;
    //     let gamma = shape_fun.gamma;
    //     let m_f = mf_calc(alpha, beta, gamma, v_s, rho_f);
    //
    //     let m: na::Matrix3<f64> = m_f + m_s;
    //
    //     let i_f = if_calc(alpha, beta, gamma, v_s, rho_f, m_s);
    //     let i_s = is_calc(m_s, self.body1.density);
    //     let i: na::Matrix3<f64> = i_f + i_s;
    //
    //     // let mut mass_tensor: na::SMatrix<f64, 7, 7> = na::SMatrix::zeros();
    //     // let mut m_slice = mass_tensor.fixed_slice_mut::<3, 3>(0, 0);
    //     // m_slice.copy_from(&m);
    //     //
    //     // let mut i_slice = mass_tensor.fixed_slice_mut::<3, 3>(4, 4);
    //     // i_slice.copy_from(&i);
    //
    //     (
    //         Rc::new(move |q: &State, p: &State| -> f64 {
    //             //let State{v: lin, q:ang} = p
    //             let lin: na::Vector3<f64> = p.v;
    //             let ang: na::Vector3<f64> = p.q.imag() * 2.0;
    //             let orient = na::UnitQuaternion::from_quaternion(q.q);
    //             let rot = orient.to_rotation_matrix();
    //             let m_lab: na::Matrix3<f64> = rot * m;
    //             let i_lab: na::Matrix3<f64> = rot * i;
    //
    //             k_tot(&lin, &ang, &m_lab, &i_lab)
    //         }),
    //         m,
    //         i,
    //     )
    // }

    //
    // fn time_steps(&self) -> i64 {
    //     let time_diff = self.t_end - self.t_begin;
    //     (time_diff / self.t_delta).ceil() as i64
    // }

    // fn advance(
    //     &mut self,
    //     ham: &Hamiltonian,
    //     jac: &na::SMatrix<f64, 7, 7>,
    //     x: &mut State,
    //     y: &mut State,
    // ) {
    //     let q = self.body.q();
    //     let p = self.body.p();
    //     let (q_new, p_new, x_new, y_new) =
    //         symplectic::strang_split(ham, jac, &q, &p, &q, &p, self.t_delta, 1, 1.0);
    //     self.body.update_q(q_new);
    //     self.body.update_p(p_new);
    //     *x = x_new;
    //     *y = y_new;
    // }

    // pub fn run<T: std::io::Write>(&mut self, wtr: &mut csv::Writer<T>) -> csv::Result<()> {
    //     let total_steps = self.time_steps();
    //     let (ham, m, i) = self.make_hamiltonian();
    //
    //     let mut mass_ten: na::SMatrix<f64, 7, 7> = na::SMatrix::zeros();
    //     let mut m_slice = mass_ten.fixed_slice_mut::<3, 3>(0, 0);
    //     m_slice.copy_from(&m);
    //
    //     let mut i_slice = mass_ten.fixed_slice_mut::<3, 3>(4, 4);
    //     i_slice.copy_from(&i);
    //
    //     let mut x = self.body.q();
    //     let mut y = self.body.p();
    //     std::println!("{}", mass_ten);
    //     for n in 0..total_steps {
    //         std::println!("Processing timestep {} -> {}", n, total_steps);
    //         self.advance(&ham, &mass_ten, &mut x, &mut y);
    //         self.fluid.kinetic_energy =
    //             ham(&self.body.q(), &self.body.p()) - self.body.kinetic_energy();
    //         self.time += self.t_delta;
    //     }
    //     wtr.flush()?;
    //     Ok(())
    // }

    /// Added (fluid) inertia tensor for body `i`.
    pub fn added_inertia_calc(&self, i: usize) -> na::Matrix3<f64> {
        let (_, inertia) = self.bodies[i].inertia_tensor(self.fluid.density);
        inertia
    }

    /// Body-frame (diagonal) added-mass tensor M_a for each body, used by the
    /// semi-implicit added-mass stabiliser. Constant in the body frame, so
    /// computed once at setup from the analytic potential-flow shape factors.
    pub fn added_mass_tensors(&self) -> Vec<na::Matrix3<f64>> {
        use crate::system::hamiltonian::{calc_shape_factor, mf_calc};
        use std::f64::consts::PI;
        self.bodies
            .iter()
            .map(|b| {
                let m_s = na::Matrix3::from_diagonal(&b.shape);
                let v_s = 4.0 / 3.0 * PI * b.shape.iter().fold(1.0, |acc, x| acc * x);
                let sf = calc_shape_factor(1e4, m_s).unwrap();
                mf_calc(sf.alpha, sf.beta, sf.gamma, v_s, self.fluid.density)
            })
            .collect()
    }
}
