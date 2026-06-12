// use std::f64::consts::PI;
// use std::rc::Rc;

// use std::ops::{Deref, DerefMut};
// use csv;
use nalgebra as na;
use crate::ellipsoids::body::Body;

// use crate::solver::symplectic;
// use crate::solver::symplectic::Hamiltonian;
use crate::system::fluid::Fluid;
// use crate::system::hamiltonian::*;
// use crate::system::state::State;

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
    pub phi_prev: na::DVector<f64>,
    pub phi_committed: na::DVector<f64>,
    pub step_dt: f64,
}





impl Simulation {
    pub fn new(
        fluid: Fluid,
        bodies: Vec<Body>,
        ndiv: u32,
    ) -> Self {
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
            phi_prev: na::DVector::zeros(0),
            phi_committed: na::DVector::zeros(0),
            step_dt: 0.01,
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
}