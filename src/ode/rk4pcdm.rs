use nalgebra as na;
use nalgebra::{Quaternion, Vector3, Vector6};

use crate::ode::dop_shared::{IntegrationError, Stats, System2, System4};
use crate::ode::pcdm::accel_get;



pub struct Rk4PCDM<Q, W, F, G, I>
    where
        F: System2<W>,
        G: System2<Q>,
        // H: System3<A, W, Q>,
        I: System4<W>,
{
    f: F,
    g: G,
    // h: H,
    i: I,
    t: f64,
    x: W,
    // x_lab: A,
    o_lab: Vector6<f64>,
    o: Q,
    inertia: na::Matrix3<f64>,
    inertia2: na::Matrix3<f64>,
    t_begin: f64,
    t_end: f64,
    step_size: f64,
    half_step: f64,
    quarter_step: f64,
    pub samp_rate: u32,
    pub t_out: Vec<f64>,
    pub x_out: Vec<W>,
    // pub x_lab_out: Vec<A>,
    pub o_out: Vec<Q>,
    pub o_lab_out: Vec<Vector6<f64>>,
    stats: Stats,
}

impl<F, G, I> //Need to generalise this to type T instead of f64.
Rk4PCDM<
    // (na::OVector<f64, D>, na::OVector<f64, D>),
    ((Quaternion<f64>, Quaternion<f64>),(Quaternion<f64>, Quaternion<f64>)),
    // OVector<f64, D>,
    (Vector6<f64>, Vector6<f64>),
    F,
    G,
    // H,
    I,
>
    where
        // D: na::Dim + na::DimName,
        F: System2<(Vector6<f64>, Vector6<f64>)>,
        // OVector<f64, D>: std::ops::Mul<f64, Output=OVector<f64, D>>,
        G: System2<((Quaternion<f64>, Quaternion<f64>),(Quaternion<f64>, Quaternion<f64>))>,
        // H: System3<OVector<f64, D>, (Vector6<f64>, Vector6<f64>), ((Quaternion<f64>, Quaternion<f64>),(Quaternion<f64>, Quaternion<f64>))>,
        I: System4<(Vector6<f64>, Vector6<f64>)>,
        // na::DefaultAllocator: na::allocator::Allocator<f64, D>,
        // na::Owned<f64, D>: Copy,
{
    //Function for creating new solver
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        f: F,
        g: G,
        i: I,
        t_begin: f64,
        x: (Vector6<f64>, Vector6<f64>), // (position, velocity)
        o1: (Quaternion<f64>, Quaternion<f64>),// (orientation, angular velocity) of body 1
        o2: (Quaternion<f64>, Quaternion<f64>),// (orientation, angular velocity) of body 2
        inertia: na::Matrix3<f64>,
        inertia2: na::Matrix3<f64>,
        t_end: f64,
        step_size: f64,
        samp_rate: u32,
    ) -> Self {
        Rk4PCDM {
            f,
            g,
            i,
            t: t_begin,
            x,
            // x_lab: na::OVector::zeros(),
            o: ((o1.0, o2.0), (o1.1, o2.1)),
            o_lab: Vector6::new(1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
            inertia,
            inertia2,
            t_begin,
            t_end,
            step_size,
            half_step: step_size * 0.5,
            quarter_step: step_size * 0.25,
            samp_rate,
            t_out: Vec::new(),
            x_out: Vec::new(),
            // x_lab_out: vec![],
            o_out: Vec::new(),
            o_lab_out: vec![],
            stats: Stats::new(),
        }
    }

    pub fn integrate(&mut self) -> Result<Stats, IntegrationError> {
        self.t_out.push(self.t);
        self.x_out.push(self.x.clone());
        self.o_out.push(self.o.clone());
        // self.x_lab_out.push(self.x_lab.clone());
        self.o_lab_out.push(self.o_lab.clone());

        let num_steps = ((self.t_end - self.t_begin) / self.step_size).ceil() as usize;
        let samp_rate = self.samp_rate as usize;
        let steps_per_sec = 1 as usize;

        for i in 0..num_steps {
            if i % steps_per_sec == 0 {
                println!("Time = {:.3}", self.t);  //Print progress
            };

            //Get (lin,ang) forces for bodies 1 & 2.
            let (linear_accel, angular_force) = self.force_get();
            println!("{:?}, {:?}", linear_accel, angular_force);

            //Calculate new positions and velocities at half time-step
            let (p_half, v_half) = self.lin_half_step(&linear_accel);

            let (q_half, o_half) = self.ang_half_step(&angular_force);

            //Update the bodies' positions and velocities
            let _ = self.f.system(0.0, &(p_half, v_half));
            let _ = self.g.system(0.0, &(q_half, o_half));

            let (linear_force_half, angular_force_half) = self.force_get();

            let t_new = self.t + self.step_size;

            let x_new = self.lin_full_step(&linear_force_half);

            let o_new = self.ang_full_step(&angular_force_half, &(q_half, o_half));

            //Update the bodies' positions and velocities
            let _ = self.f.system(0.0, &x_new);
            let _ = self.g.system(0.0, &o_new);


            let o_lab_new_v = self.quaternion_to_point(); //Calculate orientation vector position in lab frame

            // let (_, x_lab_new) = self.euler_frame_step(); //Calculate position of body in lab frame

            if i % samp_rate == 0 { //Record current state of body
                self.t_out.push(t_new);
                self.x_out.push(x_new.clone());
                self.o_out.push(o_new.clone());
            //     self.x_lab_out.push(x_lab_new.clone());
                self.o_lab_out.push(o_lab_new_v.clone());
            }


            self.t = t_new;
            self.x = x_new;
            self.o = o_new;
            // self.x_lab = x_lab_new;
            self.o_lab = o_lab_new_v;

            // self.stats.num_eval += 10;
            self.stats.accepted_steps += 1;
        }
        Ok(self.stats)
    }

    fn force_get(&mut self) -> (Vector6<f64>, Vector6<f64>) {
        let (lin, ang) = self.i.system();
        (lin, ang)
    }

    // fn force_split(&mut self, f1 :na::Vector6<f64>, f2 :na::Vector6<f64>) -> ((Vector3<f64>, Vector3<f64>), (Vector3<f64>, Vector3<f64>)){
    //     let f1_lin = Vector3::new(f1[0], f1[1], f1[2]);
    //     let f2_lin = Vector3::new(f1[3], f1[4], f1[5]);
    //
    //     let f1_ang = Vector3::new(f2[0], f2[1], f2[2]);
    //     let f2_ang = Vector3::new(f2[3], f2[4], f2[5]);
    //
    //     ((f1_lin, f1_ang),(f2_lin, f2_ang))
    // }

    fn lin_half_step(&mut self, lin_force :&Vector6<f64>) -> (Vector6<f64>, Vector6<f64>){

        let (p, v) = self.x.clone();


        let p_new = p + v * self.half_step;

        let v_new = v + lin_force * self.half_step;

        (p_new, v_new)


    }

    fn lin_full_step(&mut self, lin_force :&Vector6<f64>) -> (Vector6<f64>, Vector6<f64>) {

        let (p, v) = self.x.clone();

        let p_new = p + v * self.step_size;

        let v_new = v + lin_force * self.step_size;

        (p_new, v_new)
    }

    fn ang_half_step(&mut self, ang :&Vector6<f64>) -> ((Quaternion<f64>, Quaternion<f64>),(Quaternion<f64>, Quaternion<f64>)) {
        let (q, omega_b) = self.o.clone();

        let (q1, q2) = q;
        let (omega_lab1, omega_lab2) = omega_b;
        let inertia1 = self.inertia;
        let inertia2 = self.inertia2;

        println!("Doin ang half step");

        let omega_b1 = self.body_to_lab(&omega_lab1, &q1);
        let omega_b2 = self.body_to_lab(&omega_lab2, &q2);
        println!("Done ang half step");

        let torque1_lab = Quaternion::new(0.0, ang[0], ang[1], ang[2]);
        let torque2_lab = Quaternion::new(0.0, ang[3], ang[4], ang[5]);

        let torque1 = self.body_to_lab(&torque1_lab, &q1);
        let torque2 = self.body_to_lab(&torque2_lab, &q2);

        let ang_accel_b1 = accel_get(&omega_b1, &inertia1, &torque1);
        let ang_accel_b2 = accel_get(&omega_b2, &inertia2, &torque2);


        let omega_n_quarter_b1 = self.omega_stepper(&omega_b1, &ang_accel_b1, self.quarter_step);
        let omega_n_half_b1 = self.omega_stepper(&omega_b1, &ang_accel_b1, self.half_step);

        let omega_n_quarter1 = self.body_to_lab(&omega_n_quarter_b1, &q1);
        let q1_half_predict = self.orientation_stepper(&q1, &omega_n_quarter1, self.half_step);

        let omega_n_quarter_b2 = self.omega_stepper(&omega_b2, &ang_accel_b2, self.quarter_step);
        let omega_n_half_b2 = self.omega_stepper(&omega_b2, &ang_accel_b2, self.half_step);

        let omega_n_quarter2 = self.body_to_lab(&omega_n_quarter_b2, &q2);
        let q2_half_predict = self.orientation_stepper(&q2, &omega_n_quarter2, self.half_step);

        let omega_n_half_lab1 = self.body_to_lab(&omega_n_half_b1, &q1_half_predict);
        let omega_n_half_lab2 = self.body_to_lab(&omega_n_half_b2, &q2_half_predict);

        let q_new = (q1_half_predict, q2_half_predict);
        let o_new = (omega_n_half_lab1, omega_n_half_lab2);

        (q_new, o_new)

    }

    fn ang_full_step(&mut self,
                     ang :&Vector6<f64>,
                     half_qo :&((Quaternion<f64>, Quaternion<f64>),(Quaternion<f64>, Quaternion<f64>)))
        ->((Quaternion<f64>, Quaternion<f64>),
           (Quaternion<f64>, Quaternion<f64>)) {

        let (q, omega_lab) = self.o.clone();
        let (q_half, o_half) = half_qo;
        let (q1_half, q2_half) = q_half;
        let (omega_n_half_lab1, omega_n_half_lab2) = o_half;

        let (q1, q2) = q;
        let (omega_lab1, omega_lab2) = omega_lab;

        let omega_b1 = self.body_to_lab(&omega_lab1, &q1);
        let omega_b2 = self.body_to_lab(&omega_lab2, &q2);

        let omega_n_half_b1 = self.lab_to_body(&omega_n_half_lab1, &q1_half);
        let omega_n_half_b2 = self.lab_to_body(&omega_n_half_lab2, &q2_half);

        let inertia1 = self.inertia;
        let inertia2 = self.inertia2;

        let torque1_lab = Quaternion::new(0.0, ang[0], ang[1], ang[2]);
        let torque2_lab = Quaternion::new(0.0, ang[3], ang[4], ang[5]);

        let torque1 = self.body_to_lab(&torque1_lab, &q1_half);
        let torque2 = self.body_to_lab(&torque2_lab, &q2_half);

        let ang_accel_half_b1 = accel_get(&omega_n_half_b1, &inertia1, &torque1);
        let omega_n_half1 = self.body_to_lab(&omega_n_half_b1, &q1_half);

        let q1_full = self.orientation_stepper(&q1, &omega_n_half1, self.step_size);

        let omega1_b = self.omega_stepper(&omega_b1, &ang_accel_half_b1, self.step_size);
        let omega1 = self.body_to_lab(&omega1_b, &q1_full);

        let ang_accel_half_b2 = accel_get(&omega_n_half_b2, &inertia2, &torque2);
        let omega_n_half2 = self.body_to_lab(&omega_n_half_b2, &q2_half);

        let q2_full = self.orientation_stepper(&q2, &omega_n_half2, self.step_size);

        let omega2_b = self.omega_stepper(&omega_b2, &ang_accel_half_b2, self.step_size);
        let omega2 = self.body_to_lab(&omega2_b, &q2_full);

        self.stats.num_eval += 2;

        let q_full = (q1_full, q2_full);
        let omega = (omega1, omega2);

        (q_full, omega)


    }


    // fn force_get(&mut self) -> (na::Vector6<f64>, na::Vector6<f64>) {
    //     let force = (Vector6::zeros(), Vector6::zeros());
    //     force
    // }

    //Computes the linear motion of the ellipsoid with RK4 integration
    // fn rk4_step(&mut self) -> (f64, (na::OVector<f64, D>, na::OVector<f64, D>)) {
    //     let (p, v) = self.x.clone();
    //
    //
    //     let (k0_0, k1_0) = self.f.system(self.t, &(p, v));
    //     let (k0_1, k1_1) = self.f.system(
    //         self.t + self.half_step,
    //         &(
    //             p + k0_0 * self.half_step,
    //             v.clone() + k1_0.clone() * self.half_step,
    //         ),
    //     );
    //
    //     let (k0_2, k1_2) = self.f.system(
    //         self.t + self.half_step,
    //         &(
    //             p.clone() + k0_1.clone() * self.half_step,
    //             v.clone() + k1_1.clone() * self.half_step,
    //         ),
    //     );
    //
    //     let (k0_3, k1_3) = self.f.system(
    //         self.t + self.step_size,
    //         &(
    //             p.clone() + k0_2.clone() * self.step_size,
    //             v.clone() + k1_2.clone() * self.step_size,
    //         ),
    //     );
    //
    //     let t_new = self.t + self.step_size;
    //     let p_new = &p
    //         + (k0_0.clone() + k0_1.clone() * 2.0 + k0_2.clone() * 2.0 + k0_3.clone())
    //         * (self.step_size / 6.0);
    //     let v_new = &v
    //         + (k1_0.clone() + k1_1.clone() * 2.0 + k1_2.clone() * 2.0 + k1_3.clone())
    //         * (self.step_size / 6.0);
    //     let x_new = (p_new, v_new);
    //
    //
    //     self.stats.num_eval += 6;
    //
    //     (t_new, x_new)
    // }
    //
    // fn euler_step(&mut self) -> (f64, (na::OVector<f64, D>, na::OVector<f64, D>)) {
    //     let (p, v) = self.x.clone();
    //
    //     let (k0_0, k1_0) = self.f.system(self.t, &(p, v));
    //     let p_new = &p
    //         + k0_0.clone() * self.step_size;
    //     let v_new = &v
    //         + k1_0.clone() * self.step_size;
    //     let t_new = self.t + self.step_size;
    //     // println!("p and v = {:?}, {:?}", p, v);
    //     // println!("dp and dv = {:?}, {:?}", k0_0, k1_0);
    //     // println!("dp and dv should be = {:?}, {:?}", self.mass * v, k1);
    //     self.stats.num_eval += 1;
    //
    //     (t_new, (p_new, v_new))
    // } //debugging purposes



    //Computes the rotational motion of the ellipsoid using the "Predictor Corrector Direct Multiplier" method.
    //Preserves the unit natures of the orientation quaternion inherently.
    // fn pcdm_step(&mut self) -> (f64, (na::Quaternion<f64>, na::Quaternion<f64>)) {
    //     let (q, omega_b) = self.o.clone();
    //
    //     let inertia = self.inertia;
    //
    //     let (_, torque1) = self.g.system(
    //         self.t,
    //         &(q, omega_b),
    //     );
    //
    //     let ang_accel_b = accel_get(&omega_b, &inertia, &torque1);
    //
    //     let omega_n_quarter_b = self.omega_stepper(&omega_b, &ang_accel_b, self.quarter_step);
    //     let omega_n_half_b = self.omega_stepper(&omega_b, &ang_accel_b, self.half_step);
    //
    //     let omega_n_quarter = self.body_to_lab(&omega_n_quarter_b, &q);
    //     let q_half_predict = self.orientation_stepper(&q, &omega_n_quarter, self.half_step);
    //
    //     let (_, torque2) = self.g.system(
    //         self.t,
    //         &(q_half_predict, omega_n_half_b),
    //     );
    //
    //
    //     let ang_accel_half_b = accel_get(&omega_n_half_b, &self.inertia, &torque2);
    //     let omega_n_half = self.body_to_lab(&omega_n_half_b, &q_half_predict);
    //
    //     let q1 = self.orientation_stepper(&q, &omega_n_half, self.step_size);
    //
    //     let omega1_b = self.omega_stepper(&omega_b, &ang_accel_half_b, self.step_size);
    //     let _omega1 = self.body_to_lab(&omega1_b, &q1);
    //
    //     self.stats.num_eval += 2;
    //
    //     (self.t + self.step_size, (q1, omega1_b))
    // }

    // fn rk4_frame_step(&mut self) -> (f64, (OVector<f64, D>)) {
    //     let (x, v) = self.x.clone();
    //     // let (rows, cols) = x.shape_generic();
    //     // let mut k = vec![OVector::zeros_generic(rows, cols); 12];
    //
    //     let k_0 = self.h.system(self.t, &x);
    //     let k_1 = self.h.system(
    //         self.t + self.half_step,
    //         &(x.clone() + k_0.clone() * self.half_step),
    //     );
    //     let k_2 = self.h.system(
    //         self.t + self.half_step,
    //         &(x.clone() + k_1.clone() * self.half_step),
    //     );
    //     let k_3 = self.h.system(
    //         self.t + self.step_size,
    //         &(x.clone() + k_2.clone() * self.step_size),
    //     );
    //
    //     let t_new = self.t + self.step_size;
    //     let x_new = &x
    //         + (k_0.clone() + k_1.clone() * 2.0 + k_2.clone() * 2.0 + k_3.clone())
    //             * (self.step_size / 6.0);
    //
    //     self.stats.num_eval += 4;
    //     (t_new, x_new)
    // }

    //Computes the evolution of the body with respect to the lab frame.
    // pub fn euler_frame_step(&self) -> (f64, OVector<f64, D>) {
    //     let t = self.t.clone();
    //     let (p, v) = self.x.clone();
    //     let (q, o) = self.o.clone();
    //     let p_lab = self.x_lab.clone();
    //
    //     let dp = self.h.system(t, &p_lab, &(p, v), &(q, o));
    //     let dt = self.step_size;
    //     (t, p_lab + dt * dp)
    // }


    fn quaternion_to_point(&self) -> Vector6<f64> {
        let (q, _) = self.o;
        let (q1, q2) = q;

        let o_lab_new1 = self.body_to_lab(&Quaternion::from_imag(
            Vector3::new(1.0, 0.0, 0.0)), &q1);
        let o_lab_new_v = o_lab_new1.vector();
        let qp1 = Vector3::new(o_lab_new_v[0], o_lab_new_v[1], o_lab_new_v[2]);

        let o_lab_new2 = self.body_to_lab(&Quaternion::from_imag(
            Vector3::new(1.0, 0.0, 0.0)), &q2);
        let o_lab_new_v = o_lab_new2.vector();
        let qp2 = Vector3::new(o_lab_new_v[0], o_lab_new_v[1], o_lab_new_v[2]);

        let qp = Vector6::new(qp1[0], qp1[1], qp1[2], qp2[0], qp2[1], qp2[2]);

        qp
    }

    //Converts a (pure) quaternion p_space from lab space to body space for a body of orientation q.
    fn lab_to_body(
        &self,
        p_space: &Quaternion<f64>,
        q: &Quaternion<f64>,
    ) -> Quaternion<f64> {
        // let &q_quaternion = q.quaternion();
        let q_inv = q.try_inverse().unwrap();
        let p_body = q_inv * (p_space * q);
        p_body
    }

    //Converts a (pure) quaternion p_space from body space to lab space for a body of orientation q.
    fn body_to_lab(
        &self,
        p_body: &Quaternion<f64>,
        q: &Quaternion<f64>,
    ) -> Quaternion<f64> {
        // let &q_quaternion = q.quaternion();
        // println!("Norm of q = {}, q = {:?}", q.norm(),q);
        let q_inv = q.try_inverse().unwrap();
        let p_space = q * (p_body * q_inv);
        p_space
    }

    //Steps forward the rotational velocity omega_n according to a given angular acceleration/torque.
    fn omega_stepper(
        &self,
        omega_n: &Quaternion<f64>,
        ang_accel: &Quaternion<f64>,
        dt: f64,
    ) -> Quaternion<f64> {
        let omega_n1 = omega_n + ang_accel * dt;
        omega_n1
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
        let imag_scalar = (mag * dt * 0.5).sin() / mag;
        let imag_part = imag_scalar * omega.vector();
        let omega_n1 = Quaternion::from_parts(real_part, imag_part);
        // let omega_n1_unit = na::UnitQuaternion::from_quaternion(omega_n1);
        let q_n1 = omega_n1 * q1;
        // println!("q1 = {:?}, omega = {:?}, mag = {:?}", q1, omega, mag);
        // println!("{:?} -> {:?}", q1, q_n1);
        // println!("omega = {:?}", omega_n1);
        q_n1
    }
}
