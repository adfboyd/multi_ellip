// use nalgebra as na;
//
// use crate::ode::dop_shared::{IntegrationError, Stats, System2, System3};
// use crate::ode::pcdm::accel_get;
//
//
// pub struct Rk4PCDM_old<V, Q, A, F, G, H>
//     where
//         F: System2<V>,
//         G: System2<Q>,
//         H: System3<A, V, Q>,
// {
//     f: F,
//     g: G,
//     h: H,
//     t: f64,
//     x: V,
//     x_lab: A,
//     o_lab: na::Vector3<f64>,
//     o: Q,
//     inertia: na::Matrix3<f64>,
//     t_begin: f64,
//     t_end: f64,
//     step_size: f64,
//     half_step: f64,
//     quarter_step: f64,
//     pub samp_rate: u32,
//     pub t_out: Vec<f64>,
//     pub x_out: Vec<V>,
//     pub x_lab_out: Vec<A>,
//     pub o_out: Vec<Q>,
//     pub o_lab_out: Vec<na::Vector3<f64>>,
//     stats: Stats,
// }
//
// impl<D, F, G, H> //Need to generalise this to type T instead of f64.
// Rk4PCDM_old<
//     (na::OVector<f64, D>, na::OVector<f64, D>),
//     (na::Quaternion<f64>, na::Quaternion<f64>),
//     na::OVector<f64, D>,
//     F,
//     G,
//     H,
// >
//     where
//         D: na::Dim + na::DimName,
//         F: System2<(na::OVector<f64, D>, na::OVector<f64, D>)>,
//         na::OVector<f64, D>: std::ops::Mul<f64, Output=na::OVector<f64, D>>,
//         G: System2<(na::Quaternion<f64>, na::Quaternion<f64>)>,
//         H: System3<na::OVector<f64, D>, (na::OVector<f64, D>, na::OVector<f64, D>), (na::Quaternion<f64>, na::Quaternion<f64>)>,
//         na::DefaultAllocator: na::allocator::Allocator<f64, D>,
//         na::Owned<f64, D>: Copy,
// {
//     //Function for creating new solver
//     #[allow(clippy::too_many_arguments)]
//     pub fn new(
//         f: F,
//         g: G,
//         h: H,
//         t: f64,
//         x: (na::OVector<f64, D>, na::OVector<f64, D>), // (position, velocity)
//         o: (na::Quaternion<f64>, na::Quaternion<f64>), // (orientation, angular velocity)
//         inertia: na::Matrix3<f64>,
//         t_end: f64,
//         step_size: f64,
//         samp_rate: u32,
//     ) -> Self {
//         Rk4PCDM {
//             f,
//             g,
//             h,
//             t,
//             x,
//             x_lab: na::OVector::zeros(),
//             o,
//             o_lab: na::Vector3::new(1.0, 0.0, 0.0),
//             inertia,
//             t_begin: t,
//             t_end,
//             step_size,
//             half_step: step_size * 0.5,
//             quarter_step: step_size * 0.25,
//             samp_rate,
//             t_out: Vec::new(),
//             x_out: Vec::new(),
//             x_lab_out: vec![],
//             o_out: Vec::new(),
//             o_lab_out: vec![],
//             stats: Stats::new(),
//         }
//     }
//
//     pub fn integrate(&mut self) -> Result<Stats, IntegrationError> {
//         self.t_out.push(self.t);
//         self.x_out.push(self.x.clone());
//         self.o_out.push(self.o.clone());
//         self.x_lab_out.push(self.x_lab.clone());
//         self.o_lab_out.push(self.o_lab.clone());
//
//         let num_steps = ((self.t_end - self.t) / self.step_size).ceil() as usize;
//         let samp_rate = self.samp_rate as usize;
//         let steps_per_sec = (1.0/self.step_size) as usize;
//
//         for i in 0..num_steps {
//             if i % steps_per_sec == 0 {
//                 println!("Time = {:.3}", self.t);  //Print progress
//             };
//             let (t_new, x_new) = self.rk4_step(); //Step forward in linear motion
//             // let (t_new, x_new) = self.euler_step();
//             let (_, o_new) = self.pcdm_step(); //Step forward in rotational motion
//
//             let o_lab_new_v = self.quaternion_to_point(); //Calculate orientation in lab frame
//
//             let (_, x_lab_new) = self.euler_frame_step(); //Calculate position of body in lab frame
//
//             if i % samp_rate == 0 { //Record current state of body
//                 self.t_out.push(t_new);
//                 self.x_out.push(x_new.clone());
//                 self.o_out.push(o_new.clone());
//                 self.x_lab_out.push(x_lab_new.clone());
//                 self.o_lab_out.push(o_lab_new_v.clone());
//             }
//
//
//             self.t = t_new;
//             self.x = x_new;
//             self.o = o_new;
//             self.x_lab = x_lab_new;
//             self.o_lab = o_lab_new_v;
//
//             // self.stats.num_eval += 10;
//             self.stats.accepted_steps += 1;
//         }
//         Ok(self.stats)
//     }
//
//     //Computes the linear motion of the ellipsoid with RK4 integration
//     fn rk4_step(&mut self) -> (f64, (na::OVector<f64, D>, na::OVector<f64, D>)) {
//         let (p, v) = self.x.clone();
//
//
//         let (k0_0, k1_0) = self.f.system(self.t, &(p, v));
//         let (k0_1, k1_1) = self.f.system(
//             self.t + self.half_step,
//             &(
//                 p + k0_0 * self.half_step,
//                 v.clone() + k1_0.clone() * self.half_step,
//             ),
//         );
//
//         let (k0_2, k1_2) = self.f.system(
//             self.t + self.half_step,
//             &(
//                 p.clone() + k0_1.clone() * self.half_step,
//                 v.clone() + k1_1.clone() * self.half_step,
//             ),
//         );
//
//         let (k0_3, k1_3) = self.f.system(
//             self.t + self.step_size,
//             &(
//                 p.clone() + k0_2.clone() * self.step_size,
//                 v.clone() + k1_2.clone() * self.step_size,
//             ),
//         );
//
//         let t_new = self.t + self.step_size;
//         let p_new = &p
//             + (k0_0.clone() + k0_1.clone() * 2.0 + k0_2.clone() * 2.0 + k0_3.clone())
//             * (self.step_size / 6.0);
//         let v_new = &v
//             + (k1_0.clone() + k1_1.clone() * 2.0 + k1_2.clone() * 2.0 + k1_3.clone())
//             * (self.step_size / 6.0);
//         let x_new = (p_new, v_new);
//
//
//         self.stats.num_eval += 6;
//
//         (t_new, x_new)
//     }
//     //
//     // fn euler_step(&mut self) -> (f64, (na::OVector<f64, D>, na::OVector<f64, D>)) {
//     //     let (p, v) = self.x.clone();
//     //
//     //     let (k0_0, k1_0) = self.f.system(self.t, &(p, v));
//     //     let p_new = &p
//     //         + k0_0.clone() * self.step_size;
//     //     let v_new = &v
//     //         + k1_0.clone() * self.step_size;
//     //     let t_new = self.t + self.step_size;
//     //     // println!("p and v = {:?}, {:?}", p, v);
//     //     // println!("dp and dv = {:?}, {:?}", k0_0, k1_0);
//     //     // println!("dp and dv should be = {:?}, {:?}", self.mass * v, k1);
//     //     self.stats.num_eval += 1;
//     //
//     //     (t_new, (p_new, v_new))
//     // } //debugging purposes
//
//     //Computes the rotational motion of the ellipsoid using the "Predictor Corrector Direct Multiplier" method.
//     //Preserves the unit natures of the orientation quaternion inherently.
//     fn pcdm_step(&mut self) -> (f64, (na::Quaternion<f64>, na::Quaternion<f64>)) {
//         let (q, omega_b) = self.o.clone();
//
//         let inertia = self.inertia;
//
//         let (_, torque1) = self.g.system(
//             self.t,
//             &(q, omega_b),
//         );
//
//         let ang_accel_b = accel_get(&omega_b, &inertia, &torque1);
//
//         let omega_n_quarter_b = self.omega_stepper(&omega_b, &ang_accel_b, self.quarter_step);
//         let omega_n_half_b = self.omega_stepper(&omega_b, &ang_accel_b, self.half_step);
//
//         let omega_n_quarter = self.body_to_lab(&omega_n_quarter_b, &q);
//         let q_half_predict = self.orientation_stepper(&q, &omega_n_quarter, self.half_step);
//
//         let (_, torque2) = self.g.system(
//             self.t,
//             &(q_half_predict, omega_n_half_b),
//         );
//
//
//         let ang_accel_half_b = accel_get(&omega_n_half_b, &self.inertia, &torque2);
//         let omega_n_half = self.body_to_lab(&omega_n_half_b, &q_half_predict);
//
//         let q1 = self.orientation_stepper(&q, &omega_n_half, self.step_size);
//
//         let omega1_b = self.omega_stepper(&omega_b, &ang_accel_half_b, self.step_size);
//         let omega1 = self.body_to_lab(&omega1_b, &q1);
//
//         self.stats.num_eval += 2;
//
//         (self.t + self.step_size, (q1, omega1_b))
//     }
//
//     // fn rk4_frame_step(&mut self) -> (f64, (OVector<f64, D>)) {
//     //     let (x, v) = self.x.clone();
//     //     // let (rows, cols) = x.shape_generic();
//     //     // let mut k = vec![OVector::zeros_generic(rows, cols); 12];
//     //
//     //     let k_0 = self.h.system(self.t, &x);
//     //     let k_1 = self.h.system(
//     //         self.t + self.half_step,
//     //         &(x.clone() + k_0.clone() * self.half_step),
//     //     );
//     //     let k_2 = self.h.system(
//     //         self.t + self.half_step,
//     //         &(x.clone() + k_1.clone() * self.half_step),
//     //     );
//     //     let k_3 = self.h.system(
//     //         self.t + self.step_size,
//     //         &(x.clone() + k_2.clone() * self.step_size),
//     //     );
//     //
//     //     let t_new = self.t + self.step_size;
//     //     let x_new = &x
//     //         + (k_0.clone() + k_1.clone() * 2.0 + k_2.clone() * 2.0 + k_3.clone())
//     //             * (self.step_size / 6.0);
//     //
//     //     self.stats.num_eval += 4;
//     //     (t_new, x_new)
//     // }
//
//     //Computes the evolution of the body with respect to the lab frame.
//     pub fn euler_frame_step(&self) -> (f64, na::OVector<f64, D>) {
//         let t = self.t.clone();
//         let (p, v) = self.x.clone();
//         let (q, o) = self.o.clone();
//         let p_lab = self.x_lab.clone();
//
//         let dp = self.h.system(t, &p_lab, &(p, v), &(q, o));
//         let dt = self.step_size;
//         (t, p_lab + dt * dp)
//     }
//
//
//     fn quaternion_to_point(&self) -> na::Vector3<f64> {
//         let (q, _) = self.o;
//         let o_lab_new = self.body_to_lab(&na::Quaternion::from_imag(
//             na::Vector3::new(1.0, 0.0, 0.0)), &q);
//         let o_lab_new_v = o_lab_new.vector();
//         na::Vector3::new(o_lab_new_v[0], o_lab_new_v[1], o_lab_new_v[2])
//     }
//
//     //Converts a (pure) quaternion p_space from lab space to body space for a body of orientation q.
//     fn lab_to_body(
//         &self,
//         p_space: &na::Quaternion<f64>,
//         q: &na::Quaternion<f64>,
//     ) -> na::Quaternion<f64> {
//         // let &q_quaternion = q.quaternion();
//         let q_inv = q.try_inverse().unwrap();
//         let p_body = q_inv * (p_space * q);
//         p_body
//     }
//
//     //Converts a (pure) quaternion p_space from body space to lab space for a body of orientation q.
//     fn body_to_lab(
//         &self,
//         p_body: &na::Quaternion<f64>,
//         q: &na::Quaternion<f64>,
//     ) -> na::Quaternion<f64> {
//         // let &q_quaternion = q.quaternion();
//         // println!("Norm of q = {}, q = {:?}", q.norm(),q);
//         let q_inv = q.try_inverse().unwrap();
//         let p_space = q * (p_body * q_inv);
//         p_space
//     }
//
//     //Steps forward the rotational velocity omega_n according to a given angular acceleration/torque.
//     fn omega_stepper(
//         &self,
//         omega_n: &na::Quaternion<f64>,
//         ang_accel: &na::Quaternion<f64>,
//         dt: f64,
//     ) -> na::Quaternion<f64> {
//         let omega_n1 = omega_n + ang_accel * dt;
//         omega_n1
//     }
//
//
//     //Steps forward the orientation of a body given initial orientation q1 and rotational velocity omega.
//     fn orientation_stepper(
//         &self,
//         q1: &na::Quaternion<f64>,
//         omega: &na::Quaternion<f64>,
//         dt: f64,
//     ) -> na::Quaternion<f64> {
//         let mag = omega.norm();
//         let real_part = (mag * dt * 0.5).cos();
//         let imag_scalar = (mag * dt * 0.5).sin() / mag;
//         let imag_part = imag_scalar * omega.vector();
//         let omega_n1 = na::Quaternion::from_parts(real_part, imag_part);
//         // let omega_n1_unit = na::UnitQuaternion::from_quaternion(omega_n1);
//         let q_n1 = omega_n1 * q1;
//         // println!("q1 = {:?}, omega = {:?}, mag = {:?}", q1, omega, mag);
//         // println!("{:?} -> {:?}", q1, q_n1);
//         // println!("omega = {:?}", omega_n1);
//         q_n1
//     }
// }
