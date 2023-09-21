// use std::rc::Rc;
use std::sync::{Arc, Mutex};
// use indicatif::ParallelProgressIterator;
use nalgebra::{DMatrix, DVector, Dynamic, OMatrix, Quaternion, U1, UnitQuaternion, Vector3, Vector6};
use rayon::prelude::*;
use crate::bem::geom::*;
use crate::bem::integ::*;
use crate::bem::potentials::{dfdn_single, vec_concat};
use crate::system::system::Simulation;


type State2 = (Quaternion<f64>, Quaternion<f64>);


type Linear2State = (Vector6<f64>, Vector6<f64>);
type Angular2State = (State2, State2);
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
        println!("Linear system solved!");


        f

    }

}

impl crate::ode::System4<Linear2State> for ForceCalculate {
    ///Returns ((linear force on body1, body2), (torque on body1, body2))
    fn system(&self) -> Linear2State {


        let sys_ref = self.system.lock().unwrap();


        let (nq, mint) = (12_usize, 13_usize);
        let ndiv = sys_ref.ndiv;
        let s1 = sys_ref.body1.shape;
        // println!("Shape = {:?}", s1);
        let req1 = (s1[0] * s1[1] * s1[2]).powf(1.0 / 3.0);
        // println!("Req = {:?}", req1);
        // sys_ref.body1.linear_momentum = sys_ref.body1.linear_momentum * 0.5;


        let split_axis_x = Vector3::new(1, 0, 0);
        let split_axis_y = Vector3::new(0, 1, 0);
        let split_axis_z = Vector3::new(0, 0, 1);

        let orientation1 = UnitQuaternion::from_quaternion(sys_ref.body1.orientation);
        // println!("Splitting in x axis");
        let (nelm1, npts1, p1, n1,
            n1_xline, x_elms_pos_1, x_elms_neg_1)
            = ellip_gridder_splitter(ndiv, req1, &sys_ref.body1.shape, &sys_ref.body1.position,
                                     &orientation1, &split_axis_x, 0_usize);
        // println!("Splitting in y axis");

        let (nelm1, npts1, p1, n1,
            n1_yline, y_elms_pos_1, y_elms_neg_1)
            = ellip_gridder_splitter(ndiv, req1, &sys_ref.body1.shape, &sys_ref.body1.position,
                                     &orientation1, &split_axis_y, 0_usize);
        // println!("Splitting in z axis");

        let (nelm1, npts1, p1, n1,
            n1_zline, z_elms_pos_1, z_elms_neg_1)
            = ellip_gridder_splitter(ndiv, req1, &sys_ref.body1.shape, &sys_ref.body1.position,
                                     &orientation1, &split_axis_z, 0_usize);

        let mut body1_lines :Vec<DMatrix<usize>> = Vec::new(); //Vec containing the points required to split the body among any axis.
        body1_lines.push(n1_xline.clone());
        body1_lines.push(n1_yline.clone());
        body1_lines.push(n1_zline.clone());

        let mut body1_elms_pos :Vec<Vec<usize>> = Vec::new(); //Vec containing the list of elements with positive coordinates in each axis.
        body1_elms_pos.push(x_elms_pos_1.clone());
        body1_elms_pos.push(y_elms_pos_1.clone());
        body1_elms_pos.push(z_elms_pos_1.clone());

        let mut body1_elms_neg :Vec<Vec<usize>> = Vec::new(); //Vec containing the list of elements with negative coordinates in each axis.
        body1_elms_neg.push(x_elms_neg_1.clone());
        body1_elms_neg.push(y_elms_neg_1.clone());
        body1_elms_neg.push(z_elms_neg_1.clone());

        let mut body1_elms_all :Vec<Vec<Vec<usize>>> = Vec::new(); //Vec containing both all the lists of elements.
        body1_elms_all.push(body1_elms_neg.clone());
        body1_elms_all.push(body1_elms_pos.clone());

        // println!("Body 1 elms = {:?}", body1_elms_all);
        //
        //
        // println!("Generated splits for body1");

        let s2 = sys_ref.body2.shape;
        let req2 = 1.0 / (s2[0] * s2[1] * s2[2]).powf(1.0 / 3.0);

        let orientation2 = UnitQuaternion::from_quaternion(sys_ref.body2.orientation);
        //"Splitting in x axis
        let (nelm2, npts2, p2, n2,
            n2_xline, x_elms_pos_2, x_elms_neg_2)
            = ellip_gridder_splitter(ndiv, req2, &sys_ref.body2.shape, &sys_ref.body2.position,
                                     &orientation2, &split_axis_x, 1_usize);
        //println!("Splitting in y axis");
        let (nelm2, npts2, p2, n2,
            n2_yline, y_elms_pos_2, y_elms_neg_2)
            = ellip_gridder_splitter(ndiv, req2, &sys_ref.body2.shape, &sys_ref.body2.position,
                                     &orientation2, &split_axis_y, 1_usize);
        //println!("Splitting in z axis");
        let (nelm2, npts2, p2, n2,
            n2_zline, z_elms_pos_2, z_elms_neg_2)
            = ellip_gridder_splitter(ndiv, req2, &sys_ref.body2.shape, &sys_ref.body2.position,
                                     &orientation2, &split_axis_z, 1_usize);

        let mut body2_lines :Vec<DMatrix<usize>> = Vec::new(); //Vec containing the points required to split the body among any axis.
        body2_lines.push(n2_xline.clone());
        body2_lines.push(n2_yline.clone());
        body2_lines.push(n2_zline.clone());

        let mut body2_elms_pos :Vec<Vec<usize>> = Vec::new(); //Vec containing the list of elements with positive coordinates in each axis.
        body2_elms_pos.push(x_elms_pos_2.clone());
        body2_elms_pos.push(y_elms_pos_2.clone());
        body2_elms_pos.push(z_elms_pos_2.clone());

        let mut body2_elms_neg :Vec<Vec<usize>> = Vec::new(); //Vec containing the list of elements with negative coordinates in each axis.
        body2_elms_neg.push(x_elms_neg_2.clone());
        body2_elms_neg.push(y_elms_neg_2.clone());
        body2_elms_neg.push(z_elms_neg_2.clone());

        let mut body2_elms_all :Vec<Vec<Vec<usize>>> = Vec::new(); //Vec containing both all the lists of elements.
        body2_elms_all.push(body2_elms_neg.clone());
        body2_elms_all.push(body2_elms_pos.clone());

        let body1_elms :Vec<usize> = (0..nelm1).collect();
        let body2_elms :Vec<usize> = (nelm1..(nelm1+nelm2)).collect();

        for mut i in &mut body2_elms_all {
            for mut j in i {
                for mut k in j {
                    *k += nelm1;
                }
            }
        }

        for mut i in &mut body2_lines {
            for mut j in i {
                *j += npts1;
            }
        }

        // let  body2_elms_all_copy = body2_elms_all.clone();

        // println!("Body 2 elms = {:?}", body2_elms_all);
        //
        // println!("Generated splits for body2");

        // println!("Body 1 lines = {:?}", body1_lines);
        // println!("Body 2 lines = {:?}", body2_lines);


        let (nelm, npts, p, n) = combiner(nelm1, nelm2, npts1, npts2, &p1, &p2, &n1, &n2);

        let (nelm, npts, n_xline, x_elms_pos, x_elms_neg) = combiner_splitter(nelm1, nelm2, npts1, npts2,
                                                                              &n1_xline, &n2_xline,
                                                                              &x_elms_pos_1, &x_elms_pos_2,
                                                                              &x_elms_neg_1, &x_elms_neg_2);

        let (nelm, npts, n_yline, y_elms_pos, y_elms_neg) = combiner_splitter(nelm1, nelm2, npts1, npts2,
                                                                              &n1_yline, &n2_yline,
                                                                              &y_elms_pos_1, &y_elms_pos_2,
                                                                              &y_elms_neg_1, &y_elms_neg_2);

        let (nelm, npts, n_zline, z_elms_pos, z_elms_neg) = combiner_splitter(nelm1, nelm2, npts1, npts2,
                                                                              &n1_zline,&n2_zline,
                                                                              &z_elms_pos_1, &z_elms_pos_2,
                                                                              &z_elms_neg_1, &z_elms_neg_2);


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
        // println!("F = {:?}", f);

        let mut linear_pressure1 = Vector3::new(0.0, 0.0, 0.0);
        let mut angular_pressure1 = Vector3::new(0.0, 0.0, 0.0);

        let mut linear_pressure2 = Vector3::new(0.0, 0.0, 0.0);
        let mut angular_pressure2 = Vector3::new(0.0, 0.0, 0.0);

        let ks = (0..nelm).collect::<Vec<usize>>();
        let ks = vec![0_usize,nelm1];

        let m_linear_pressure1 = Mutex::from(linear_pressure1);
        let m_angular_pressure1 = Mutex::from(angular_pressure1);

        let m_linear_pressure2 = Mutex::from(linear_pressure2);
        let m_angular_pressure2 = Mutex::from(angular_pressure2);
//should be 0..nelm
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

            let (f1, f2, f3, f4, f5, f6) = (f[i1], f[i2], f[i3], f[i4], f[i5], f[i6]);
            let (df1, df2, df3, df4, df5, df6) = (df[i1], df[i2], df[i3], df[i4], df[i5], df[i6]);

            let (al, be, ga) = (alpha[k], beta[k], gamma[k]);


            let (xi, eta) = (1.0/3.0, 1.0/3.0);

            let (p0, vn, _hs, f_p0, dfdn_p0) = lsdlpp_3d_interp(p1, p2, p3, p4, p5, p6,
                                                                  vna1, vna2, vna3, vna4, vna5, vna6,
                                                                  f1, f2, f3, f4, f5, f6,
                                                                  df1, df2, df3, df4, df5, df6,
                                                                  al, be, ga, xi, eta);



            let p0_n = vn; //Another name for the normal vector at p0.

            let which_body = if k < nelm1 {  //Which body is the point we are integrating round on?
                1
            } else {
                2
            };


            // println!("Splitting on body no. {:?}", which_body);

            // let p0_body = sys_ref.body1.lab_body_convert(&p0);

            // println!("Transformed point {:?}, to {:?}", p0, p0_body);

            // println!("{:?}, {:?}, {:?}, {:?}, {:?}, {:?}", i1, i2, i3, i4, i5, i6);
            // println!("{:?}", p1);
            // println!("{:?}", p2);
            // println!("{:?}", p3);
            // println!("{:?}", p4);
            // println!("{:?}", p5);
            // println!("{:?}", p6);


            let (mut axis_index, mut axis_direction) = (0_usize, 0_usize); //axis_index gives index of the axis. axis_direction = 0 if negative, 1 if positive.
            if which_body == 1_usize {
                (axis_index, axis_direction) = sys_ref.body1.surface_splitter(&p0);
            } else if which_body == 2_usize {
                (axis_index, axis_direction) = sys_ref.body2.surface_splitter(&p0);
            } else {
                panic!("Not in either body?!!")
            };

            // println!("Splitting on {:?} index, positivity = {:?}", axis_index, axis_direction);

            let mut n_line  = if which_body == 1 {
                body1_lines.get(axis_index).unwrap().clone()
            } else if which_body == 2 {
                body2_lines.get(axis_index).unwrap().clone()
            } else {
                panic!("Not in either body?!!")
            };

            // println!("Line points are {:?}", n_line);

            let mut sing_elms = if which_body == 1 {
                let correct_direction_vec = body1_elms_all.get(axis_direction).unwrap().clone();
                correct_direction_vec.get(axis_index).unwrap().clone()
            } else if which_body == 2 {
                let correct_direction_vec = body2_elms_all.get(axis_direction).unwrap().clone();
                correct_direction_vec.get(axis_index).unwrap().clone()
            } else {
                panic!("Not in either body?!!")
            };

            // println!("Singular elements are {:?}", sing_elms);

            let mut non_sing_elms = if which_body == 1 {
                let correct_direction_vec = body1_elms_all.get(1_usize - axis_direction).unwrap().clone();  //Get opposite index of axis direction.
                let mut non_sing_body_elms_temp = correct_direction_vec.get(axis_index).unwrap().clone();
                non_sing_body_elms_temp.extend(body2_elms.clone());
                non_sing_body_elms_temp
            } else if which_body == 2 {
                let correct_direction_vec = body2_elms_all.get(1_usize - axis_direction).unwrap().clone();  //Get opposite index of axis direction.
                let mut non_sing_body_elms_temp = correct_direction_vec.get(axis_index).unwrap().clone();
                non_sing_body_elms_temp.extend(body1_elms.clone());
                non_sing_body_elms_temp
            } else {
                panic!("Not in either body?!!")
            };

            // println!("Non-singular elements are {:?}", non_sing_elms);



            let rhs = grad_3d_all_rhs(&sing_elms, &non_sing_elms, mint,
                                      &f, &dfdn,
                                      &p, &n, &n_line,  &vna,
                                      &alpha, &beta, &gamma,
                                      &xiq, &etq, &wq,
                                      &p0,  f_p0, dfdn_p0);

            let lhs_matrix = grad_3d_all_lhs(&sing_elms, &non_sing_elms, mint,
                                              &f, &dfdn,
                                              &p, &n, &n_line, &vna,
                                              &alpha, &beta, &gamma,
                                              &xiq, &etq, &wq,
                                              &p0, &p0_n, f_p0, dfdn_p0);




            let decomp_lhs = lhs_matrix.lu();
            let u = decomp_lhs.solve(&rhs).expect("Linear resolution of eq(28) failed");
            // println!("rhs = {:?}", rhs);
            // println!("lhs = {:?}", lhs_matrix);
            // println!("At this element the fluid velocity is {:?}", u);



            let u_square = u.norm_squared();
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

            let linearity = vn.dot(&p0_lab);
            let perpendicularity = vn.cross(&p0_lab).norm();

            let lin_pressure = pressure * linearity;
            let ang_pressure = pressure * perpendicularity;

            let torque_vec = vn.cross(&p0_lab);
            // let angular_vec = x_cen.cross(&perp_vec);

            let lin_inc = lin_pressure * vn;
            let ang_inc = ang_pressure * torque_vec;

            //Unlock correct cumulative pressure on correct body
            let mut linear_pressure = if which_body == 1_usize {
                m_linear_pressure1.lock().unwrap()}
            else if which_body == 2_usize {
                m_linear_pressure2.lock().unwrap()
            } else {
                panic!("Not in either body!");
            };

            let mut angular_pressure = if which_body == 1_usize {
                m_angular_pressure1.lock().unwrap()}
            else if which_body == 2_usize {
                m_angular_pressure2.lock().unwrap()
            } else {
                panic!("Not in either body!");
            };

            //add result to the right body.
            *linear_pressure += lin_inc;
            *angular_pressure += ang_inc;

        });

        // println!("Body1 force = {:?}, {:?}", linear_pressure1, angular_pressure1);
        // println!("Body2 force = {:?}, {:?}", linear_pressure2, angular_pressure2);
        let linear_pressure1 = m_linear_pressure1.into_inner().unwrap();
        let angular_pressure1 = m_angular_pressure1.into_inner().unwrap();

        let linear_pressure2 = m_linear_pressure2.into_inner().unwrap();
        let angular_pressure2 = m_angular_pressure2.into_inner().unwrap();

        let m1 = sys_ref.body1.mass();
        let m2 = sys_ref.body2.mass();

        let lin_accel_1 = linear_pressure1 / m1;
        let lin_accel_2 = linear_pressure2 / m2;

        let torque1 = angular_pressure1;
        let torque2 = angular_pressure2;

        let lin_accel = Vector6::new(lin_accel_1[0], lin_accel_1[1], lin_accel_1[2], lin_accel_2[0], lin_accel_2[1], lin_accel_2[2]);
        let ang_accel = Vector6::new(torque1[0], torque1[1], torque1[2], torque2[0], torque2[1], torque2[2]);

        (lin_accel, ang_accel)

        // let v1 = Vector6::new(1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        // let v2 = Vector6::new(1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        // (v1, v2)

    }
}

impl crate::ode::System2<Linear2State> for LinearUpdate {
    fn system(&self, _x: f64, y: &Linear2State) -> Linear2State {

        let mut sys_ref = self.system.lock().unwrap();

        let (p, v) = y.clone();

        let p1 = Vector3::new(p[0], p[1], p[2]);
        let p2 = Vector3::new(p[3], p[4], p[5]);

        let v1 = Vector3::new(v[0], v[1], v[2]);
        let v2 = Vector3::new(v[3], v[4], v[5]);

        let a1 = sys_ref.body1.shape[0];
        let a2 = sys_ref.body2.shape[0];

        let dist = (p1-p2).norm();

        if (dist < (a1+a2)) {
            panic!("Ellipsoids are too close together.")
        }

        sys_ref.body1.position = p1;
        sys_ref.body2.position = p2;

        sys_ref.body1.linear_momentum = v1 * sys_ref.body1.mass();
        sys_ref.body2.linear_momentum = v2 * sys_ref.body2.mass();

        // sys_ref.body1.print_stats();
        // sys_ref.body2.print_stats();

        (p, v)
    }
}

impl crate::ode::System2<Angular2State> for AngularUpdate {
    fn system(&self, _x: f64, y: &Angular2State) -> Angular2State {

        let mut sys_ref = self.system.lock().unwrap();

        let (q, omega) = y.clone();

        let (q1, q2) = q;
        let (omega1, omega2) = omega;

        sys_ref.body1.orientation = q1;
        sys_ref.body2.orientation = q2;

        let i1 = sys_ref.body1.inertia;
        let i2 = sys_ref.body2.inertia;

        let o1_vec = omega1.vector();
        let o2_vec = omega2.vector();

        let ang_mom_vec1 = i1.try_inverse().unwrap() * o1_vec;
        sys_ref.body1.angular_momentum = Quaternion::from_imag(ang_mom_vec1);

        let ang_mom_vec2 =  i2.try_inverse().unwrap() * o2_vec;
        sys_ref.body2.angular_momentum = Quaternion::from_imag(ang_mom_vec2);

        (q, omega)
    }
}