// use std::rc::Rc;
use std::sync::{Arc, Mutex};
// use indicatif::ParallelProgressIterator;
use nalgebra::{ArrayStorage, DMatrix, DVector, Dynamic, Matrix, OMatrix, Quaternion, U1, U9, UnitQuaternion, Vector3, Vector6};
use rayon::prelude::*;
use crate::bem::geom::*;
use crate::bem::integ::*;
use crate::bem::potentials::{dfdn_single, vec_concat};
use crate::system::system::Simulation;


type State2 = (Quaternion<f64>, Quaternion<f64>);
type State3 = (Quaternion<f64>, Quaternion<f64>, Quaternion<f64>);

type Vector9<T>=Matrix<T, U9, U1, ArrayStorage<T, 9, 1>>;

type Linear2State = (Vector6<f64>, Vector6<f64>);
type Linear3State = (Vector9<f64>, Vector9<f64>);
type Angular2State = (State2, State2);

type Angular3State = (State3, State3);
type PhiState = OMatrix<f64, Dynamic, U1>;



pub struct PhiCalculate {
    pub system: Arc<Mutex<Simulation>>,
}

pub struct ForceCalculate {
    pub system: Arc<Mutex<Simulation>>,
}

pub struct LinearUpdate {
    pub system: Arc<Mutex<Simulation>>,
}

pub struct AngularUpdate {
    pub system: Arc<Mutex<Simulation>>,
}


impl crate::ode::System4<PhiState> for PhiCalculate {
    fn system(&self) -> PhiState {
        let sys_ref = self.system.lock().unwrap();

        let (nq, mint) = (12_usize, 13_usize);
        let ndiv = sys_ref.ndiv;
        let s1 = sys_ref.body1.shape;
        let req1 = 1.0 / (s1[0] * s1[1] * s1[2]).powf(1.0 / 3.0);
        // sys_ref.body1.linear_momentum = sys_ref.body1.linear_momentum * 0.5;

        let split_axis_x = Vector3::new(1, 0, 0);
        let split_axis_y = Vector3::new(0, 1, 0);
        let split_axis_z = Vector3::new(0, 0, 1);

        let orientation1 = UnitQuaternion::from_quaternion(sys_ref.body1.orientation);
        let (nelm1, npts1, p1, n1) = ellip_gridder(ndiv, req1, &sys_ref.body1.shape, &sys_ref.body1.position, &orientation1);


        let s2 = sys_ref.body2.shape;
        let req2 = 1.0 / (s2[0] * s2[1] * s2[2]).powf(1.0 / 3.0);

        let orientation2 = UnitQuaternion::from_quaternion(sys_ref.body2.orientation);
        let (nelm2, npts2, p2, n2) = ellip_gridder(ndiv, req2, &sys_ref.body2.shape, &sys_ref.body2.position, &orientation2);
        // let (nelm2, npts2, p2, n2, n2_yline, y_elms_pos, y_elms_neg) = ellip_gridder_splitter(ndiv, req2, sys_ref.body2.shape, sys_ref.body2.position, orientation2, split_axis_y);
        // let (nelm2, npts2, p2, n2, n2_zline, z_elms_pos, z_elms_neg) = ellip_gridder_splitter(ndiv, req2, sys_ref.body2.shape, sys_ref.body2.position, orientation2, split_axis_z);

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

        let dfdn_1 = dfdn_single(&sys_ref.body1.position, &sys_ref.body1.linear_velocity(), &sys_ref.body1.angular_velocity().imag(), npts1, &p1, &vna1);
        let dfdn_2 = dfdn_single(&sys_ref.body2.position, &sys_ref.body2.linear_velocity(), &sys_ref.body2.angular_velocity().imag(), npts2, &p2, &vna2);
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
        // println!("Linear system solved!");


        f

    }

}

impl crate::ode::System4<Linear3State> for ForceCalculate {
    ///Returns ((linear force on body1, body2), (torque on body1, body2))
    fn system(&self) -> Linear3State {
        let sys_ref = self.system.lock().unwrap();


        let (nq, mint) = (12_usize, 13_usize);
        let ndiv = sys_ref.ndiv;
        let nbody = sys_ref.nbody;
        let s1 = sys_ref.body1.shape;
        // println!("Shape = {:?}", s1);
        let req1 = (s1[0] * s1[1] * s1[2]).powf(1.0 / 3.0);


        let orientation1 = UnitQuaternion::from_quaternion(sys_ref.body1.orientation);
        let (nelm1, npts1, p1, n1)
            = ellip_gridder(ndiv, req1, &sys_ref.body1.shape,
                            &sys_ref.body1.position,
                            &orientation1);

        // println!("NDIV = {:?}, nbody = {:?}, Nelm1 = {:?}, npts1 = {:?}", ndiv, nbody,nelm1, npts1);


        let s2 = sys_ref.body2.shape;
        let req2 = 1.0 / (s2[0] * s2[1] * s2[2]).powf(1.0 / 3.0);

        let orientation2 = UnitQuaternion::from_quaternion(sys_ref.body2.orientation);

        let (nelm2, npts2, p2, n2)
            = if nbody == 2 || nbody == 3 {
            ellip_gridder(ndiv, req2, &sys_ref.body2.shape,
                          &sys_ref.body2.position,
                          &orientation2)
        } else if nbody == 1 {
            (nelm1, npts1, DMatrix::<f64>::from_element(npts1, 3, 0.0), DMatrix::<usize>::from_element(nelm1, 6, 0))
        } else {
            panic!("Number of bodies not supported.)")
        };

        let s3 = sys_ref.body3.shape;
        let req3 = 1.0 / (s3[0] * s3[1] * s3[2]).powf(1.0 / 3.0);

        let orientation3 = UnitQuaternion::from_quaternion(sys_ref.body3.orientation);

        let (nelm3, npts3, p3, n3)
            = if nbody == 3 {
            ellip_gridder(ndiv, req3, &sys_ref.body3.shape,
                          &sys_ref.body3.position,
                          &orientation3)
        } else if nbody == 1 || nbody == 2 {
            (nelm1, npts1, DMatrix::<f64>::from_element(npts1, 3, 0.0), DMatrix::<usize>::from_element(nelm1, 6, 0))
        } else {
            panic!("Number of bodies not supported.)")
        };


        let (nelm, mut npts, p, n) = if nbody == 2 {
            combiner(nelm1, nelm2, npts1, npts2, &p1, &p2, &n1, &n2)
        } else if nbody == 1 {
            (nelm1, npts1, p1.clone(), n1.clone())
        } else if nbody == 3 {
            let (nelm12, npts12, p12, n12) = combiner(nelm1, nelm2, npts1, npts2, &p1, &p2, &n1, &n2);
            combiner(nelm12, nelm3, npts12, npts3, &p12, &p3, &n12, &n3)
        } else {
            panic!("Number of bodies not supported.")
        };


        let (zz, ww) = gauss_leg(nq);
        let (xiq, etq, wq) = gauss_trgl(mint);

        let (alpha, beta, gamma) =
            abc_vec(nelm, &p, &n);

        let (vna, _vlm, _sa) = elm_geom(npts, nelm, mint,
                                        &p, &n,
                                        &alpha, &beta, &gamma,
                                        &xiq, &etq, &wq);


        let mut vna1 = DMatrix::zeros(npts1, 3);
        let mut vna2 = DMatrix::zeros(npts2, 3);
        let mut vna3 = DMatrix::zeros(npts3, 3);

        if nbody == 2 {
            //Can loop over both since npts1 == npts2
            if npts1 == npts2 {
                for i in 0..npts1 {
                    for j in 0..3 {
                        vna1[(i, j)] = vna[(i, j)];
                        vna2[(i, j)] = vna[(i + npts1, j)];
                    }
                }
            };
        };

        if nbody == 3 {
            if npts1 == npts2 && npts1 == npts3 {
                for i in 0..npts1 {
                    for j in 0..3 {
                        vna1[(i, j)] = vna[(i, j)];
                        vna2[(i, j)] = vna[(i + npts1, j)];
                        vna3[(i, j)] = vna[(i + npts1 + npts2, j)];
                    }
                }
            }
        }


        let dfdn_1 = dfdn_single(&sys_ref.body1.position, &sys_ref.body1.linear_velocity(), &sys_ref.body1.angular_velocity().imag(), npts1, &p1, &vna1);
        let dfdn_2 = dfdn_single(&sys_ref.body2.position, &sys_ref.body2.linear_velocity(), &sys_ref.body2.angular_velocity().imag(), npts2, &p2, &vna2);
        let dfdn_3 = dfdn_single(&sys_ref.body3.position, &sys_ref.body3.linear_velocity(), &sys_ref.body3.angular_velocity().imag(), npts3, &p3, &vna3);
        let mut dfdn = DVector::zeros(npts);
        dfdn = if nbody == 2 {
            vec_concat(&dfdn_1, &dfdn_2)
        } else if nbody == 1 {
            dfdn_single(&sys_ref.body1.position, &sys_ref.body1.linear_velocity(), &sys_ref.body1.angular_velocity().imag(), npts1, &p1, &vna)
        } else if nbody == 3 {
            let dfdn_12 = vec_concat(&dfdn_1, &dfdn_2);
            vec_concat(&dfdn_12, &dfdn_3)
        } else {
            panic!("Other number of bodies not supported.");
        };
        // sys_ref.body2.print_stats();
        // println!("dfdn_2 = {:?}", dfdn_2);
        // println!("dfdn = {:?}", dfdn);

        let rhs = lslp_3d(npts, nelm, mint, nq,
                          &dfdn, &p, &n, &vna,
                          &alpha, &beta, &gamma,
                          &xiq, &etq, &wq, &zz, &ww);


        // println!("Grids created");
        let amat_1 = DMatrix::zeros(npts, npts);
        let amat = Mutex::from(amat_1);

        let js = (0..npts).collect::<Vec<usize>>();

        // println!("Computing columns of influence matrix");

        js.par_iter().for_each(|&j| {
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
        // println!("F = {:?}", f);
        //The value of phi at any point in the domain can be calculated as follows:
        //
        // let test_p = Vector3::new(0.0, 0.0, 0.0);
        //
        // let phi_eg = lsdlpp_3d(npts, nelm, mint, &f, &dfdn, &p, &n, &vna,
        //                             &alpha, &beta, &gamma,
        //                             &xiq, &etq, &wq,
        //                             &test_p);
        //
        // // println!("Test value of phi is {:?} at {:?}", phi_eg, test_p);
        //
        // let grad_phi_eg = grad_3d(nelm, mint, &f, &dfdn, &p, &n, &vna,
        //                               &alpha, &beta, &gamma,
        //                               &xiq, &etq, &wq,
        //                               &test_p);
        //
        // println!("The test value of gradphi is {:?}", grad_phi_eg);

        let mut linear_pressure1 = Vector3::new(0.0, 0.0, 0.0);
        let mut angular_pressure1 = Vector3::new(0.0, 0.0, 0.0);

        let mut linear_pressure2 = Vector3::new(0.0, 0.0, 0.0);
        let mut angular_pressure2 = Vector3::new(0.0, 0.0, 0.0);

        let mut linear_pressure3 = Vector3::new(0.0, 0.0, 0.0);
        let mut angular_pressure3 = Vector3::new(0.0, 0.0, 0.0);

        let ks = (0..nelm).collect::<Vec<usize>>();
        // let ks = vec![0_usize,nelm1];

        let m_linear_pressure1 = Mutex::from(linear_pressure1);
        let m_angular_pressure1 = Mutex::from(angular_pressure1);

        let m_linear_pressure2 = Mutex::from(linear_pressure2);
        let m_angular_pressure2 = Mutex::from(angular_pressure2);

        let m_linear_pressure3 = Mutex::from(linear_pressure3);
        let m_angular_pressure3 = Mutex::from(angular_pressure3);
//should be 0..nelm

        if nbody == 2 {
            ks.par_iter().for_each(|&k| {


                // println!();
                // println!("Iterating over {:?}th element", k);



                let i1 = n[(k, 0)];
                let i2 = n[(k, 1)];
                let i3 = n[(k, 2)];
                let i4 = n[(k, 3)];
                let i5 = n[(k, 4)];
                let i6 = n[(k, 5)];

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
                //
                let (f1, f2, f3, f4, f5, f6) = (f[i1], f[i2], f[i3], f[i4], f[i5], f[i6]);
                let (df1, df2, df3, df4, df5, df6) = (df[i1], df[i2], df[i3], df[i4], df[i5], df[i6]);

                let (al, be, ga) = (alpha[k], beta[k], gamma[k]);


                let (u_0, area) = gradient_interp_3d_integral(k, mint,
                                                              &f, &dfdn,
                                                              &p, &n, &vna,
                                                              &alpha, &beta, &gamma,
                                                              &xiq, &etq, &wq);

                let (p0, vn, _hs, _f_p0, _df_p0,
                    _dfdxi, _dfdet, _ddxi, _ddet) =
                    gradient_interp(p1, p2, p3, p4, p5, p6,
                                    vna1, vna2, vna3, vna4, vna5, vna6,
                                    f1, f2, f3, f4, f5, f6,
                                    df1, df2, df3, df4, df5, df6,
                                    al, be, ga, 1. / 3., 1. / 3.);


                //
                // let p0_n = vn; //Another name for the normal vector at p0.
                //
                let which_body = if k < nelm1 {  //Which body is the point we are integrating round on?
                    1
                } else {
                    2
                };


                // let u_square = u_0.norm_squared();
                let u_square = u_0;
                // println!("u1, u2, u3, u^2 = {:?}, {:?}, {:?}, {:?}", u1, u2, u3, u_square);
                // println!("nelm = {:?}, nelm1 = {:?}, nsize = {:?}", nelm,nelm1, n.shape());

                // let concat_test = vec_concat(&dfdn_1, &dfdn_2);
                // println!("{:?}",dfdn_2);
                // println!("dfdn= {:?}", concat_test);

                let pressure = -u_square * 0.5 * sys_ref.fluid.density;

                // println!("Pressure = {:?}",pressure);

                let p0_lab = if which_body == 1 {
                    p0 - sys_ref.body1.position
                } else if which_body == 2 {
                    p0 - sys_ref.body2.position
                } else {
                    panic!("Not in either body?!!")
                };

                let linearity = vn.dot(&p0_lab.normalize());
                let perpendicularity = vn.cross(&p0_lab).norm();

                let lin_pressure = pressure * linearity;
                let ang_pressure = pressure * perpendicularity;

                let torque_vec = vn.cross(&p0_lab);
                // let angular_vec = x_cen.cross(&perp_vec);

                let lin_inc = lin_pressure * vn;
                let ang_inc = ang_pressure * torque_vec;

                //Unlock correct cumulative pressure on correct body
                let mut linear_pressure = if which_body == 1_usize {
                    m_linear_pressure1.lock().unwrap()
                } else if which_body == 2_usize {
                    m_linear_pressure2.lock().unwrap()
                } else {
                    panic!("Not in either body!");
                };

                let mut angular_pressure = if which_body == 1_usize {
                    m_angular_pressure1.lock().unwrap()
                } else if which_body == 2_usize {
                    m_angular_pressure2.lock().unwrap()
                } else {
                    panic!("Not in either body!");
                };

                //add result to the right body.
                *linear_pressure += lin_inc;
                *angular_pressure += ang_inc;
            });
        } else if nbody == 3 { ks.par_iter().for_each(|&k| {


            // println!();
            // println!("Iterating over {:?}th element", k);



            let i1 = n[(k, 0)];
            let i2 = n[(k, 1)];
            let i3 = n[(k, 2)];
            let i4 = n[(k, 3)];
            let i5 = n[(k, 4)];
            let i6 = n[(k, 5)];

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
            //
            let (f1, f2, f3, f4, f5, f6) = (f[i1], f[i2], f[i3], f[i4], f[i5], f[i6]);
            let (df1, df2, df3, df4, df5, df6) = (df[i1], df[i2], df[i3], df[i4], df[i5], df[i6]);

            let (al, be, ga) = (alpha[k], beta[k], gamma[k]);


            let (u_0, area) = gradient_interp_3d_integral(k, mint,
                                                          &f, &dfdn,
                                                          &p, &n, &vna,
                                                          &alpha, &beta, &gamma,
                                                          &xiq, &etq, &wq);

            let (p0, vn, _hs, _f_p0, _df_p0,
                _dfdxi, _dfdet, _ddxi, _ddet) =
                gradient_interp(p1, p2, p3, p4, p5, p6,
                                vna1, vna2, vna3, vna4, vna5, vna6,
                                f1, f2, f3, f4, f5, f6,
                                df1, df2, df3, df4, df5, df6,
                                al, be, ga, 1. / 3., 1. / 3.);


            //
            // let p0_n = vn; //Another name for the normal vector at p0.
            //
            let which_body = if k < nelm1 {  //Which body is the point we are integrating round on?
                1
            } else if k < nelm2 + nelm1 {
                2
            } else {
                3
            };

            // println!("Nbody = {:?}, k = {:?}, position of element = {:?}", which_body, k, p0);

            // let u_square = u_0.norm_squared();
            let u_square = u_0;


            let pressure = -u_square * 0.5 * sys_ref.fluid.density;

            // println!("Pressure = {:?}",pressure);

            let p0_lab = if which_body == 1 {
                p0 - sys_ref.body1.position
            } else if which_body == 2 {
                p0 - sys_ref.body2.position
            } else if which_body == 3 {
                p0 - sys_ref.body3.position
            } else {
                panic!("Not in either body?!!")
            };

            let linearity = vn.dot(&p0_lab.normalize());
            let perpendicularity = vn.cross(&p0_lab).norm();

            let lin_pressure = pressure * linearity;
            let ang_pressure = pressure * perpendicularity;

            let torque_vec = vn.cross(&p0_lab);
            // let angular_vec = x_cen.cross(&perp_vec);

            let lin_inc = lin_pressure * vn;
            let ang_inc = ang_pressure * torque_vec;

            //Unlock correct cumulative pressure on correct body
            let mut linear_pressure = if which_body == 1_usize {
                m_linear_pressure1.lock().unwrap()
            } else if which_body == 2_usize {
                m_linear_pressure2.lock().unwrap()
            } else if which_body == 3_usize {
                m_linear_pressure3.lock().unwrap()
            } else {
                panic!("Not in either body!");
            };

            let mut angular_pressure = if which_body == 1_usize {
                m_angular_pressure1.lock().unwrap()
            } else if which_body == 2_usize {
                m_angular_pressure2.lock().unwrap()
            } else if which_body == 3_usize {
                m_angular_pressure3.lock().unwrap()
            } else {
                panic!("Not in either body!");
            };

            //add result to the right body.
            *linear_pressure += lin_inc;
            *angular_pressure += ang_inc;
        });
        } else if nbody == 1 {
            ks.par_iter().for_each(|&k| {


                // println!();
                // println!("Iterating over {:?}th element", k);



                let i1 = n[(k, 0)];
                let i2 = n[(k, 1)];
                let i3 = n[(k, 2)];
                let i4 = n[(k, 3)];
                let i5 = n[(k, 4)];
                let i6 = n[(k, 5)];

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
                //
                let (f1, f2, f3, f4, f5, f6) = (f[i1], f[i2], f[i3], f[i4], f[i5], f[i6]);
                let (df1, df2, df3, df4, df5, df6) = (df[i1], df[i2], df[i3], df[i4], df[i5], df[i6]);

                let (al, be, ga) = (alpha[k], beta[k], gamma[k]);


                let (u_0, area) = gradient_interp_3d_integral(k, mint,
                                                              &f, &dfdn,
                                                              &p, &n, &vna,
                                                              &alpha, &beta, &gamma,
                                                              &xiq, &etq, &wq);

                let (p0, vn, _hs, _f_p0, _df_p0,
                    _dfdxi, _dfdet, _ddxi, _ddet) =
                    gradient_interp(p1, p2, p3, p4, p5, p6,
                                    vna1, vna2, vna3, vna4, vna5, vna6,
                                    f1, f2, f3, f4, f5, f6,
                                    df1, df2, df3, df4, df5, df6,
                                    al, be, ga, 1. / 3., 1. / 3.);


                //
                // let p0_n = vn; //Another name for the normal vector at p0.
                //
                let which_body = 1;  //Which body is the point we are integrating round on?



                // let u_square = u_0.norm_squared();
                let u_square = u_0;
                // println!("u1, u2, u3, u^2 = {:?}, {:?}, {:?}, {:?}", u1, u2, u3, u_square);
                // println!("nelm = {:?}, nelm1 = {:?}, nsize = {:?}", nelm,nelm1, n.shape());

                // let concat_test = vec_concat(&dfdn_1, &dfdn_2);
                // println!("{:?}",dfdn_2);
                // println!("dfdn= {:?}", concat_test);

                let pressure = -u_square * 0.5 * sys_ref.fluid.density;

                // println!("Pressure = {:?}",pressure);

                let p0_lab = p0 - sys_ref.body1.position;


                let linearity = vn.dot(&p0_lab.normalize());
                let perpendicularity = vn.cross(&p0_lab).norm();

                let lin_pressure = pressure * linearity;
                let ang_pressure = pressure * perpendicularity;

                let torque_vec = vn.cross(&p0_lab);
                // let angular_vec = x_cen.cross(&perp_vec);

                let lin_inc = lin_pressure * vn;
                let ang_inc = ang_pressure * torque_vec;

                //Unlock correct cumulative pressure on correct body
                let mut linear_pressure = if which_body == 1_usize {
                    m_linear_pressure1.lock().unwrap()
                } else {
                    panic!("Not in either body!");
                };

                let mut angular_pressure = if which_body == 1_usize {
                    m_angular_pressure1.lock().unwrap()

                } else {
                    panic!("Not in either body!");
                };

                //add result to the right body.
                *linear_pressure += lin_inc;
                *angular_pressure += ang_inc;
            });
        }

        // println!("Body1 force = {:?}, {:?}", linear_pressure1, angular_pressure1);
        // println!("Body2 force = {:?}, {:?}", linear_pressure2, angular_pressure2);
        let linear_pressure1 = m_linear_pressure1.into_inner().unwrap();
        let angular_pressure1 = m_angular_pressure1.into_inner().unwrap();

        let linear_pressure2 = m_linear_pressure2.into_inner().unwrap();
        let angular_pressure2 = m_angular_pressure2.into_inner().unwrap();

        let linear_pressure3 = m_linear_pressure3.into_inner().unwrap();
        let angular_pressure3 = m_angular_pressure3.into_inner().unwrap();

        let m1 = sys_ref.body1.mass();
        let m2 = sys_ref.body2.mass();
        let m3 = sys_ref.body3.mass();


        let lin_accel_1 = linear_pressure1 / m1;
        let lin_accel_2 = linear_pressure2 / m2;
        let lin_accel_3 = linear_pressure3 / m3;


        let torque1 = angular_pressure1;
        let torque2 = angular_pressure2;
        let torque3 = angular_pressure3;

        let lin_accel = Vector9::from_row_slice(&[lin_accel_1[0], lin_accel_1[1], lin_accel_1[2], lin_accel_2[0], lin_accel_2[1], lin_accel_2[2], lin_accel_3[0], lin_accel_3[1], lin_accel_3[2]]);
        let ang_accel = Vector9::from_row_slice(&[torque1[0], torque1[1], torque1[2], torque2[0], torque2[1], torque2[2], torque3[0], torque3[1], torque3[2]]);

        (lin_accel, ang_accel)

        // let v1 = Vector6::new(1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        // let v2 = Vector6::new(1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        // (v1, v2)

    }
}

impl crate::ode::System2<Linear3State> for LinearUpdate {
    fn system(&self, _x: f64, y: &Linear3State) -> Linear3State {

        let mut sys_ref = self.system.lock().unwrap();

        let (p, v) = y.clone();

        let p1 = Vector3::new(p[0], p[1], p[2]);
        let p2 = Vector3::new(p[3], p[4], p[5]);
        let p3 = Vector3::new(p[6], p[7], p[8]);


        let v1 = Vector3::new(v[0], v[1], v[2]);
        let v2 = Vector3::new(v[3], v[4], v[5]);
        let v3 = Vector3::new(v[6], v[7], v[8]);


        sys_ref.body1.position = p1;
        sys_ref.body2.position = p2;
        sys_ref.body3.position = p3;

        sys_ref.body1.linear_momentum = v1 * sys_ref.body1.mass();
        sys_ref.body2.linear_momentum = v2 * sys_ref.body2.mass();
        sys_ref.body3.linear_momentum = v3 * sys_ref.body3.mass();

        // sys_ref.body1.print_stats();
        // sys_ref.body2.print_stats();

        (p, v)
    }
}

impl crate::ode::System2<Angular3State> for AngularUpdate {
    fn system(&self, _x: f64, y: &Angular3State) -> Angular3State {

        let mut sys_ref = self.system.lock().unwrap();

        let (q, omega) = y.clone();

        let (q1, q2, q3) = q;
        let (omega1, omega2, omega3) = omega;

        sys_ref.body1.orientation = q1;
        sys_ref.body2.orientation = q2;
        sys_ref.body3.orientation = q3;

        let i1 = sys_ref.body1.inertia;
        let i2 = sys_ref.body2.inertia;
        let i3 = sys_ref.body3.inertia;

        let o1_vec = omega1.vector();
        let o2_vec = omega2.vector();
        let o3_vec = omega3.vector();

        let ang_mom_vec1 = i1.try_inverse().unwrap() * o1_vec;
        sys_ref.body1.angular_momentum = Quaternion::from_imag(ang_mom_vec1);

        let ang_mom_vec2 =  i2.try_inverse().unwrap() * o2_vec;
        sys_ref.body2.angular_momentum = Quaternion::from_imag(ang_mom_vec2);

        let ang_mom_vec3 = i3.try_inverse().unwrap() * o3_vec;
        sys_ref.body3.angular_momentum = Quaternion::from_imag(ang_mom_vec3);

        (q, omega)
    }
}