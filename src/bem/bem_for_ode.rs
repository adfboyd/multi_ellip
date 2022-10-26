use std::rc::Rc;
use std::sync::Mutex;
use indicatif::ParallelProgressIterator;
use nalgebra::{DMatrix, DVector, Dynamic, OMatrix, U1, UnitQuaternion, Vector3};
use rayon::prelude::*;
use crate::bem::geom::*;
use crate::bem::integ::*;
use crate::bem::potentials::{dfdn_single, vec_concat};
use crate::system::system::Simulation;

use nalgebra as na;

type State = (na::Vector3<f64>, na::Vector3<f64>);
type State2 = (na::Quaternion<f64>, na::Quaternion<f64>);
type State3 = na::Vector3<f64>;
type Time = f64;

type DoubleState = (State, State);
type DoubleState2 = (State2, State2);
type PhiState = OMatrix<f64, Dynamic, U1>;



pub struct PhiCalculate {
    pub system: Rc<Simulation>,
}

pub struct ForceCalculate {
    pub system: Rc<Simulation>,
}

pub struct AngForceCalculate {
    pub system: Rc<Simulation>,
}


impl crate::ode::System4<PhiState> for PhiCalculate {
    fn system(&self) -> PhiState {

        let (nq, mint) = (12_usize, 13_usize);
        let ndiv = self.system.ndiv;
        let s1 = self.system.body1.shape;
        let req1 = 1.0 / (s1[0] * s1[1] * s1[2]).powf(1.0/3.0);
        // self.system.body1.linear_momentum = self.system.body1.linear_momentum * 0.5;

        let orientation1 = UnitQuaternion::from_quaternion(self.system.body1.orientation);
        let (nelm1, npts1,p1, n1) = ellip_gridder(ndiv, req1, self.system.body1.shape, self.system.body1.position, orientation1);

        let s2 = self.system.body2.shape;
        let req2 = 1.0 / (s2[0] * s2[1] * s2[2]).powf(1.0/3.0);

        let orientation2 = UnitQuaternion::from_quaternion(self.system.body2.orientation);
        let (nelm2, npts2, p2, n2) = ellip_gridder(ndiv, req2, self.system.body2.shape, self.system.body2.position, orientation2);

        let (nelm, npts, p, n) = combiner(nelm1, nelm2, npts1, npts2, &p1, &p2, &n1, &n2);

        let (zz, ww) = gauss_leg(nq);
        let (xiq, etq, wq) = gauss_trgl(mint);

        let (alpha, beta, gamma) = abc_vec(nelm, &p, &n);

        let (vna, _vlm, _sa) = elm_geom(npts, nelm, mint,
                                        &p, &n,
                                        &alpha, &beta, &gamma,
                                        &xiq, &etq, &wq);

        let mut vna1 = DMatrix::zeros(npts1, 3);
        let mut vna2 = DMatrix::zeros(npts2, 3);

        //Can loop over both since npts1 == npts2
        if npts1 == npts2 {
            for i in 0..npts1 {
                for j in 0..3 {
                    vna1[(i, j)] = vna[(i, j)];
                    vna2[(i, j)] = vna[(i + npts1, j)];
                }
            }
        };

        let dfdn_1 = dfdn_single(&self.system.body1.position, &self.system.body1.linear_velocity(), &self.system.body1.angular_velocity().imag(), npts1, &p1, &vna1);
        let dfdn_2 = dfdn_single(&self.system.body2.position, &self.system.body2.linear_velocity(), &self.system.body2.angular_velocity().imag(), npts2, &p2, &vna2);
        let dfdn = vec_concat(&dfdn_1, &dfdn_2);

        let rhs = lslp_3d(npts, nelm, mint, nq,
                          &dfdn, &p, &n, &vna,
                          &alpha, &beta, &gamma,
                          &xiq, &etq, &wq, &zz, &ww);



        // println!("Grids created");
        let amat_1 = DMatrix::zeros(npts, npts);
        let amat = Mutex::from(amat_1);

        let js = (0..npts).collect::<Vec<usize>>();

        println!("Computing columns of influence matrix");

        js.par_iter().for_each(|&j|  {
            // println!("Computing column {} of the influence matrix", j);
            let mut q = DVector::zeros(npts);
            q[j] = 1.0;

            let dlp = ldlp_3d(npts, nelm, mint,
                              &q, &p, &n, &vna,
                              &alpha, &beta, &gamma,
                              &xiq, &etq, &wq);

            for k in 0..npts {
                let mut amat = amat.lock().unwrap();
                amat[(k, j)] = dlp[k];
            }
            q[j] = 0.0;
        });

        let amat_final = amat.into_inner().unwrap();
        // println!("Matrix created");

        let decomp = amat_final.lu();
        // println!("Matrix decomposed");

        let f = decomp.solve(&rhs).expect("Linear resolution failed");
        println!("Linear system solved!");

        f

    }

}

impl crate::ode::System2<State> for ForceCalculate {
    fn system(&self, _x: f64, y: &State) -> (na::Vector6<f64>, na::Vector6<f64>) {



        // let (m, i) = self.system.body1.inertia_tensor(self.system.fluid.density);
        //
        // let (_, v) = y;
        //
        // let dx0 = m * v;
        // let dx1 = na::Vector3::zeros();

        let (nq, mint) = (12_usize, 13_usize);
        let ndiv = self.system.ndiv;
        let s1 = self.system.body1.shape;
        let req1 = 1.0 / (s1[0] * s1[1] * s1[2]).powf(1.0/3.0);
        // self.system.body1.linear_momentum = self.system.body1.linear_momentum * 0.5;

        let orientation1 = UnitQuaternion::from_quaternion(self.system.body1.orientation);
        let (nelm1, npts1,p1, n1) = ellip_gridder(ndiv, req1, self.system.body1.shape, self.system.body1.position, orientation1);

        let s2 = self.system.body2.shape;
        let req2 = 1.0 / (s2[0] * s2[1] * s2[2]).powf(1.0/3.0);

        let orientation2 = UnitQuaternion::from_quaternion(self.system.body2.orientation);
        let (nelm2, npts2, p2, n2) = ellip_gridder(ndiv, req2, self.system.body2.shape, self.system.body2.position, orientation2);

        let (nelm, npts, p, n) = combiner(nelm1, nelm2, npts1, npts2, &p1, &p2, &n1, &n2);

        let (zz, ww) = gauss_leg(nq);
        let (xiq, etq, wq) = gauss_trgl(mint);

        let (alpha, beta, gamma) = abc_vec(nelm, &p, &n);

        let (vna, _vlm, _sa) = elm_geom(npts, nelm, mint,
                                        &p, &n,
                                        &alpha, &beta, &gamma,
                                        &xiq, &etq, &wq);

        let mut vna1 = DMatrix::zeros(npts1, 3);
        let mut vna2 = DMatrix::zeros(npts2, 3);

        //Can loop over both since npts1 == npts2
        if npts1 == npts2 {
            for i in 0..npts1 {
                for j in 0..3 {
                    vna1[(i, j)] = vna[(i, j)];
                    vna2[(i, j)] = vna[(i + npts1, j)];
                }
            }
        };

        let dfdn_1 = dfdn_single(&self.system.body1.position, &self.system.body1.linear_velocity(), &self.system.body1.angular_velocity().imag(), npts1, &p1, &vna1);
        let dfdn_2 = dfdn_single(&self.system.body2.position, &self.system.body2.linear_velocity(), &self.system.body2.angular_velocity().imag(), npts2, &p2, &vna2);
        let dfdn = vec_concat(&dfdn_1, &dfdn_2);

        let rhs = lslp_3d(npts, nelm, mint, nq,
                          &dfdn, &p, &n, &vna,
                          &alpha, &beta, &gamma,
                          &xiq, &etq, &wq, &zz, &ww);



        // println!("Grids created");
        let amat_1 = DMatrix::zeros(npts, npts);
        let amat = Mutex::from(amat_1);

        let js = (0..npts).collect::<Vec<usize>>();

        println!("Computing columns of influence matrix");

        js.par_iter().for_each(|&j|  {
            // println!("Computing column {} of the influence matrix", j);
            let mut q = DVector::zeros(npts);
            q[j] = 1.0;

            let dlp = ldlp_3d(npts, nelm, mint,
                              &q, &p, &n, &vna,
                              &alpha, &beta, &gamma,
                              &xiq, &etq, &wq);

            for k in 0..npts {
                let mut amat = amat.lock().unwrap();
                amat[(k, j)] = dlp[k];
            }
            q[j] = 0.0;
        });

        let amat_final = amat.into_inner().unwrap();
        // println!("Matrix created");

        let decomp = amat_final.lu();
        // println!("Matrix decomposed");

        let f = decomp.solve(&rhs).expect("Linear resolution failed");
        let df = dfdn.clone();
        // println!("Linear system solved!");

        let mut linear_pressure1 = Vector3::new(0.0, 0.0, 0.0);
        let mut angular_pressure1 = Vector3::new(0.0, 0.0, 0.0);

        let mut linear_pressure2 = Vector3::new(0.0, 0.0, 0.0);
        let mut angular_pressure2 = Vector3::new(0.0, 0.0, 0.0);

        let nelm1 = nelm / 2;
        let eps = 0.01;

        for k in 0..nelm1 {

            let i1 = n[(k, 0)] - 1;
            let i2 = n[(k, 1)] - 1;
            let i3 = n[(k, 2)] - 1;
            let i4 = n[(k, 3)] - 1;
            let i5 = n[(k, 4)] - 1;
            let i6 = n[(k, 5)] - 1;

            let p1 = Vector3::new(p[(i1, 0)], p[(i1, 1)], p[(i1, 2)]);
            let p2 = Vector3::new(p[(i2, 0)], p[(i2, 1)], p[(i2, 2)]);
            let p3 = Vector3::new(p[(i3, 0)], p[(i3, 1)], p[(i3, 2)]);
            let p4 = Vector3::new(p[(i4, 0)], p[(i4, 1)], p[(i4, 2)]);
            let p5 = Vector3::new(p[(i5, 0)], p[(i5, 1)], p[(i5, 2)]);
            let p6 = Vector3::new(p[(i6, 0)], p[(i6, 1)], p[(i6, 2)]);

            let vna1 = Vector3::new(vna[(i1, 0)], vna[(i1, 1)], vna[(i1, 2)]);
            let vna2 = Vector3::new(vna[(i2, 0)], vna[(i2, 1)], vna[(i2, 2)]);
            let vna3 = Vector3::new(vna[(i3, 0)], vna[(i3, 1)], vna[(i3, 2)]);
            let vna4 = Vector3::new(vna[(i4, 0)], vna[(i4, 1)], vna[(i4, 2)]);
            let vna5 = Vector3::new(vna[(i5, 0)], vna[(i5, 1)], vna[(i5, 2)]);
            let vna6 = Vector3::new(vna[(i6, 0)], vna[(i6, 1)], vna[(i6, 2)]);

            let (f1, f2, f3, f4, f5, f6) = (f[i1], f[i2], f[i3], f[i4], f[i5], f[i6]);
            let (df1, df2, df3, df4, df5, df6) = (df[i1], df[i2], df[i3], df[i4], df[i5], df[i6]);

            let (al, be, ga) = (alpha[k], beta[k], gamma[k]);


            let (xi, eta) = (1.0/3.0, 1.0/3.0);

            let (x_vec, vn, hs, fint, dfdn_int) = lsdlpp_3d_interp(p1, p2, p3, p4, p5, p6,
                                                                  vna1, vna2, vna3, vna4, vna5, vna6,
                                                                  f1, f2, f3, f4, f5, f6,
                                                                  df1, df2, df3, df4, df5, df6,
                                                                  al, be, ga, xi, eta);

            let testvec = Vector3::new(1.0, 1.0, 1.0);

            let dx_1 = vn.cross(&testvec).normalize(); //Generate set of three perpendicular directions.
            let dx_2 = vn.cross(&dx_1).normalize();

            let x_body = if k < nelm1 {
                x_vec - self.system.body1.position
            } else {
                x_vec - self.system.body2.position
            };

            let p0 = if k < nelm1 {
                self.system.body1.position + x_body * 1.01
            } else {
                self.system.body2.position + x_body * 1.01
            };

            let u1 = grad_3d(npts, nelm, mint,
                                    &f, &dfdn, &p, &n, &vna,
                                    &alpha, &beta, &gamma,
                                    &xiq, &etq, &wq, &p0, &vn, eps);

            let u2 = grad_3d(npts, nelm, mint,
                             &f, &dfdn, &p, &n, &vna,
                             &alpha, &beta, &gamma,
                             &xiq, &etq, &wq, &p0, &dx_1, eps);

            let u3 = grad_3d(npts, nelm, mint,
                             &f, &dfdn, &p, &n, &vna,
                             &alpha, &beta, &gamma,
                             &xiq, &etq, &wq, &p0, &dx_2, eps);

            let u_square = u1.powi(2) + u2.powi(2) + u3.powi(2);

            let pressure = -u_square;

            let linearity = vn.dot(&x_cen);
            let perpendicularity = vn.cross(&x_cen).norm();

            let lin_pressure = pressure * linearity;
            let ang_pressure = pressure * perpendicularity;

            let torque_vec = vn.cross(&x_cen);
            // let angular_vec = x_cen.cross(&perp_vec);

            let lin_inc = lin_pressure * vn;
            let ang_inc = ang_pressure * torque_vec;

            if k < nelm1 {
                linear_pressure1 += lin_inc;
                angular_pressure1 += ang_inc;
            } else {
                linear_pressure2 += lin_inc;
                angular_pressure2 += ang_inc; }

        }

        let linearForceTotal = na::Vector6::new(linear_pressure1[0], linear_pressure1[1], linear_pressure1[2],
                    linear_pressure2[0], linear_pressure2[1], linear_pressure2[2]);

        let angularForceTotal = na::Vector6::new(angular_pressure1[0], angular_pressure1[1], angular_pressure1[2],
                                                 angular_pressure2[0], angular_pressure2[1], angular_pressure2[2]);

        let results = (linearForceTotal, angularForceTotal);




        results


    }
}