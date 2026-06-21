use crate::ellipsoids::body::Body;
use crate::system::fluid::Fluid;
use nalgebra as na;

/// Number of provisional first-step passes used to bootstrap the unsteady
/// pressure history at t = 0.
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
    /// Surface-potential history for the same-stage BDF2 unsteady-pressure
    /// stencil. Entries are previous force-call potentials, most recent last.
    pub phi_history: std::collections::VecDeque<na::DVector<f64>>,
    /// Remaining first-step bootstrap passes.
    pub bootstrap_redos: usize,
    /// Enable the semi-implicit added-mass-partitioned velocity update.
    pub added_mass_stab: bool,
    /// Blend the BDF2 potential-time-derivative stencil toward BDF1.
    pub phidot_blend: f64,
    /// Evaluate regular BEM quadrature on the exact ellipsoid surface rather
    /// than the quadratic interpolation of scaled mesh nodes.
    pub exact_ellipsoid_geometry: bool,
    /// Also evaluate singular BEM self-element quadrature on the exact
    /// ellipsoid surface. This is experimental and off by default.
    pub exact_singular_geometry: bool,
    pub step_dt: f64,
    /// During strong-coupling trial evaluations, prevent provisional potentials
    /// from entering the committed history.
    pub freeze_phi_history: bool,
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
            added_mass_stab: false,
            phidot_blend: 0.0,
            exact_ellipsoid_geometry: false,
            exact_singular_geometry: false,
            step_dt: 0.01,
            freeze_phi_history: false,
        }
    }

    /// Added (fluid) inertia tensor for body `i`.
    pub fn added_inertia_calc(&self, i: usize) -> na::Matrix3<f64> {
        let (_, inertia) = self.bodies[i].inertia_tensor(self.fluid.density);
        inertia
    }

    /// Body-frame (diagonal) added-mass tensor M_a for each body, used by the
    /// semi-implicit added-mass stabiliser. Constant in the body frame.
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
