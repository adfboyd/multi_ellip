use std::f64::consts::PI;
use std::rc::Rc;

use csv;
use nalgebra as na;
use crate::ellipsoids::body::Body;

use crate::solver::symplectic;
use crate::solver::symplectic::Hamiltonian;
use crate::system::fluid::Fluid;
use crate::system::hamiltonian::*;
use crate::system::state::State;

#[derive(Clone)]
pub struct Simulation {
    pub fluid: Fluid,
    pub body1: Body,
    pub body2: Body,
    pub time: f64,
    pub t_begin: f64,
    pub t_end: f64,
    pub t_delta: f64,
    pub samp_rate: u32,
    pub mass_tensor: na::Matrix3<f64>,
    pub inertia_tensor: na::Matrix3<f64>,
    hamiltonian: Hamiltonian,
}

impl Simulation {
    pub fn new(
        fluid: Fluid,
        mut body1: Body,
        mut body2: Body,
        t_end: f64,
        t_delta: f64,
        samp_rate: u32,
        ratio: f64,
    ) -> Self {
        let mut sim = Self {
            fluid,
            body1,
            body2,
            time: 0.0,
            t_begin: 0.0,
            t_end,
            t_delta,
            samp_rate,
            mass_tensor: na::Matrix3::zeros(),
            inertia_tensor: na::Matrix3::zeros(),
            hamiltonian: Rc::new(|q: &State, p: &State| -> f64 { 0.0 }),
        };

        //Normalise rotational velocity to around 2*pi rad/s
        let init_frequency = body.rotational_frequency();
        sim.body.angular_momentum = body.angular_momentum / init_frequency;

        //Setup linear velocity for desired energy ratio
        let init_direction = body.linear_momentum;
        sim.body.linear_momentum = body.ic_generator(init_direction, ratio);

        let (ham, mass, inert) = sim.make_hamiltonian();
        sim.hamiltonian = ham;
        sim.mass_tensor = mass;
        sim.inertia_tensor = inert;
        sim
    }

    pub fn make_hamiltonian(&self) -> (Hamiltonian, na::Matrix3<f64>, na::Matrix3<f64>) {
        let rho_f = self.fluid.density;

        let m_s = na::Matrix3::from_diagonal(&self.body.shape);
        let v_s = 4.0 / 3.0 * PI * self.body.shape.iter().fold(1.0, |acc, x| acc * x);

        let lambda = 10000.0;

        let shape_fun = calc_shape_factor(lambda, m_s).unwrap();

        let alpha = shape_fun.alpha;
        let beta = shape_fun.beta;
        let gamma = shape_fun.gamma;
        let m_f = mf_calc(alpha, beta, gamma, v_s, rho_f);

        let m: na::Matrix3<f64> = m_f + m_s;

        let i_f = if_calc(alpha, beta, gamma, v_s, rho_f, m_s);
        let i_s = is_calc(m_s, self.body.density);
        let i: na::Matrix3<f64> = i_f + i_s;

        // let mut mass_tensor: na::SMatrix<f64, 7, 7> = na::SMatrix::zeros();
        // let mut m_slice = mass_tensor.fixed_slice_mut::<3, 3>(0, 0);
        // m_slice.copy_from(&m);
        //
        // let mut i_slice = mass_tensor.fixed_slice_mut::<3, 3>(4, 4);
        // i_slice.copy_from(&i);

        (
            Rc::new(move |q: &State, p: &State| -> f64 {
                //let State{v: lin, q:ang} = p
                let lin: na::Vector3<f64> = p.v;
                let ang: na::Vector3<f64> = p.q.imag() * 2.0;
                let orient = na::UnitQuaternion::from_quaternion(q.q);
                let rot = orient.to_rotation_matrix();
                let m_lab: na::Matrix3<f64> = rot * m;
                let i_lab: na::Matrix3<f64> = rot * i;

                k_tot(&lin, &ang, &m_lab, &i_lab)
            }),
            m,
            i,
        )
    }

    fn time_steps(&self) -> i64 {
        let time_diff = self.t_end - self.t_begin;
        (time_diff / self.t_delta).ceil() as i64
    }

    fn advance(
        &mut self,
        ham: &Hamiltonian,
        jac: &na::SMatrix<f64, 7, 7>,
        x: &mut State,
        y: &mut State,
    ) {
        let q = self.body.q();
        let p = self.body.p();
        let (q_new, p_new, x_new, y_new) =
            symplectic::strang_split(ham, jac, &q, &p, &q, &p, self.t_delta, 1, 1.0);
        self.body.update_q(q_new);
        self.body.update_p(p_new);
        *x = x_new;
        *y = y_new;
    }

    pub fn run<T: std::io::Write>(&mut self, wtr: &mut csv::Writer<T>) -> csv::Result<()> {
        let total_steps = self.time_steps();
        let (ham, m, i) = self.make_hamiltonian();

        let mut mass_ten: na::SMatrix<f64, 7, 7> = na::SMatrix::zeros();
        let mut m_slice = mass_ten.fixed_slice_mut::<3, 3>(0, 0);
        m_slice.copy_from(&m);

        let mut i_slice = mass_ten.fixed_slice_mut::<3, 3>(4, 4);
        i_slice.copy_from(&i);

        let mut x = self.body.q();
        let mut y = self.body.p();
        std::println!("{}", mass_ten);
        for n in 0..total_steps {
            std::println!("Processing timestep {} -> {}", n, total_steps);
            self.advance(&ham, &mass_ten, &mut x, &mut y);
            self.fluid.kinetic_energy =
                ham(&self.body.q(), &self.body.p()) - self.body.kinetic_energy();
            self.time += self.t_delta;
        }
        wtr.flush()?;
        Ok(())
    }

    pub fn added_inertia_calc(&self) -> na::Matrix3<f64> {
        let (_, _, i) = self.make_hamiltonian();
        i
    }
}