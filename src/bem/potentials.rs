use std::sync::Mutex;
use nalgebra::{DMatrix, DVector, UnitQuaternion, Vector3};
use indicatif::{ParallelProgressIterator, ProgressBar};
use indicatif::ProgressIterator;
use rayon::prelude::*;
use crate::bem::geom::*;
use crate::bem::integ::*;
use crate::ellipsoids::body::Body;



///Calculates correct Neumann boundary conditions for a given discretized body with specified linear and rotational velocity.
pub fn dfdn_single(c :&Vector3<f64>, u :&Vector3<f64>, omega :&Vector3<f64>, npts :usize, p :&DMatrix<f64>, vna :&DMatrix<f64>) -> DVector<f64> {


    let mut dfdn = DVector::zeros(npts);

    for i in 0..npts{

        let p_i = Vector3::new(p[(i, 0)], p[(i, 1)], p[(i, 2)]);
        let df = u + (p_i - c).cross(omega);

        let dn = Vector3::new(vna[(i, 0)], vna[(i, 1)], vna[(i, 2)]);

        dfdn[i] = dn.dot(&df);
    }

    dfdn
}

///Concatenates two vectors vertically
pub fn vec_concat(v1 :&DVector<f64>, v2 :&DVector<f64>) -> DVector<f64> {

    let npts1 = v1.shape().0;
    let npts2 = v2.shape().0;

    let mut v = DVector::zeros(npts1 + npts2);

    for i in 0..npts1 {
        v[i] = v1[i];
    }
    for i in 0..npts2 {
        v[i + npts1] = v2[i];
    }

    v

}

///Calculates the correct value of phi on the boundary by solving the BEM.
pub fn f_finder(ndiv :u32, req :f64, shape :Vector3<f64>, centre :Vector3<f64>, orientation :UnitQuaternion<f64>,
                  velocity :Vector3<f64>, omega :Vector3<f64>, nq :usize, mint :usize) -> DVector<f64> {

    let (nelm, npts, p, n) = ellip_gridder(ndiv, req, shape, centre, orientation);

    let (zz, ww) = gauss_leg(nq);
    let (xiq, etq, wq) = gauss_trgl(mint);

    let (alpha, beta, gamma) = abc_vec(nelm, &p, &n);

    let (vna, _vlm, _sa) = elm_geom(npts, nelm, mint,
                                  &p, &n,
                                  &alpha, &beta, &gamma,
                                  &xiq, &etq, &wq);

    //Calculate dfdn

    let dfdn = dfdn_single(&centre, &velocity, &omega, npts, &p, &vna);

    //Calculate RHS of linear equation

    let rhs = lslp_3d(npts, nelm, mint, nq,
                      &dfdn, &p, &n, &vna,
                      &alpha, &beta, &gamma,
                      &xiq, &etq, &wq, &zz, &ww);

    println!("RHS is calculated");


    let mut q = DVector::zeros(npts);
    let mut amat = DMatrix::zeros(npts, npts);

    let js = (0..npts).collect::<Vec<usize>>();

    let bar = ProgressBar::new(npts as u64);

    for &j in js.iter() {
        // println!("Computing column {} of the influence matrix", j);
        q[j] = 1.0;

        let dlp = ldlp_3d(npts, nelm, mint,
                          &q, &p, &n, &vna,
                          &alpha, &beta, &gamma,
                          &xiq, &etq, &wq);

        for k in 0..npts {
            amat[(k, j)] = dlp[k];
        }
        q[j] = 0.0;
        bar.inc(1);
    }

    let decomp = amat.lu();

    let f = decomp.solve(&rhs).expect("Linear resolution failed");

    println!("Linear system solved!");

    f
}

///Calculates the correct value of phi on the boundary by solving the BEM (un-parallelised).
pub fn phi_1body_serial(body :&Body, ndiv :u32, nq :usize, mint :usize) -> DVector<f64> {

    let s = body.shape;
    let req = 1.0 / (s[0] * s[1] * s[2]).powf(1.0/3.0);

    let orientation = UnitQuaternion::from_quaternion(body.orientation);
    let (nelm, npts, p, n) = ellip_gridder(ndiv, req, body.shape, body.position, orientation);

    let (zz, ww) = gauss_leg(nq);
    let (xiq, etq, wq) = gauss_trgl(mint);

    let (alpha, beta, gamma) = abc_vec(nelm, &p, &n);

    let (vna, _vlm, _sa) = elm_geom(npts, nelm, mint,
                                    &p, &n,
                                    &alpha, &beta, &gamma,
                                    &xiq, &etq, &wq);

    //Calculate dfdn

    let dfdn = dfdn_single(&body.position, &body.linear_velocity(), &body.angular_velocity().imag(), npts, &p, &vna);

    //Calculate RHS of linear equation

    let rhs = lslp_3d(npts, nelm, mint, nq,
                      &dfdn, &p, &n, &vna,
                      &alpha, &beta, &gamma,
                      &xiq, &etq, &wq, &zz, &ww);

    println!("RHS is calculated");


    let mut q = DVector::zeros(npts);
    let mut amat = DMatrix::zeros(npts, npts);

    let js = (0..npts).collect::<Vec<usize>>();

    for &j in js.iter().progress() {
        // println!("Computing column {} of the influence matrix", j);
        q[j] = 1.0;

        let dlp = ldlp_3d(npts, nelm, mint,
                          &q, &p, &n, &vna,
                          &alpha, &beta, &gamma,
                          &xiq, &etq, &wq);

        for k in 0..npts {
            amat[(k, j)] = dlp[k];
        }
        q[j] = 0.0;
    }

    let decomp = amat.lu();

    let f = decomp.solve(&rhs).expect("Linear resolution failed");

    println!("Linear system solved!");

    f

}

///Calculates the correct phi for a given body used as the boundary, using BEM.
pub fn f_1body(body :&Body, ndiv :u32, nq :usize, mint :usize) -> DVector<f64> {

    let s = body.shape;
    let req = 1.0 / (s[0] * s[1] * s[2]).powf(1.0/3.0);

    let orientation = UnitQuaternion::from_quaternion(body.orientation);
    let (nelm, npts, p, n) = ellip_gridder(ndiv, req, body.shape, body.position, orientation);

    let (zz, ww) = gauss_leg(nq);
    let (xiq, etq, wq) = gauss_trgl(mint);

    let (alpha, beta, gamma) = abc_vec(nelm, &p, &n);

    let (vna, _vlm, _sa) = elm_geom(npts, nelm, mint,
                                    &p, &n,
                                    &alpha, &beta, &gamma,
                                    &xiq, &etq, &wq);

    //Calculate dfdn

    let dfdn = dfdn_single(&body.position, &body.linear_velocity(), &body.angular_velocity().imag(), npts, &p, &vna);

    //Calculate RHS of linear equation

    let rhs = lslp_3d(npts, nelm, mint, nq,
                      &dfdn, &p, &n, &vna,
                      &alpha, &beta, &gamma,
                      &xiq, &etq, &wq, &zz, &ww);

    println!("RHS is calculated");


    let amat_1 = DMatrix::zeros(npts, npts);
    let amat = Mutex::new(amat_1);

    let js = (0..npts).collect::<Vec<usize>>();

    js.par_iter().progress_count(npts as u64).for_each(|&j| {
        // println!("Computing column {} of the influence matrix", j);
        let mut q = DVector::zeros(npts);

        q[j] = 1.0;

        let dlp = ldlp_3d(npts, nelm, mint,
                          &q, &p, &n, &vna,
                          &alpha, &beta, &gamma,
                          &xiq, &etq, &wq);

        let mut amat = amat.lock().unwrap();

        for k in 0..npts {
            amat[(k, j)] = dlp[k];
        }
    });

    let amat_final = amat.into_inner().unwrap();

    let decomp = amat_final.lu();

    let f = decomp.solve(&rhs).expect("Linear resolution failed");

    println!("Linear system solved!");

    f

}

pub fn phi_eval_1body(body :&Body, ndiv :u32, nq :usize, mint :usize, p0 :Vector3<f64>) -> f64 {

    let s = body.shape;
    let req = 1.0 / (s[0] * s[1] * s[2]).powf(1.0/3.0);

    let orientation = UnitQuaternion::from_quaternion(body.orientation);
    let (nelm, npts, p, n) = ellip_gridder(ndiv, req, body.shape, body.position, orientation);

    let (zz, ww) = gauss_leg(nq);
    let (xiq, etq, wq) = gauss_trgl(mint);

    let (alpha, beta, gamma) = abc_vec(nelm, &p, &n);

    let (vna, _vlm, _sa) = elm_geom(npts, nelm, mint,
                                    &p, &n,
                                    &alpha, &beta, &gamma,
                                    &xiq, &etq, &wq);

    //Calculate dfdn

    let dfdn = dfdn_single(&body.position, &body.linear_velocity(), &body.angular_velocity().imag(), npts, &p, &vna);

    //Calculate RHS of linear equation

    let rhs = lslp_3d(npts, nelm, mint, nq,
                      &dfdn, &p, &n, &vna,
                      &alpha, &beta, &gamma,
                      &xiq, &etq, &wq, &zz, &ww);

    println!("RHS is calculated");


    let amat_1 = DMatrix::zeros(npts, npts);
    let amat = Mutex::new(amat_1);

    let js = (0..npts).collect::<Vec<usize>>();

    js.par_iter().progress_count(npts as u64).for_each(|&j| {
        // println!("Computing column {} of the influence matrix", j);
        let mut q = DVector::zeros(npts);

        q[j] = 1.0;

        let dlp = ldlp_3d(npts, nelm, mint,
                          &q, &p, &n, &vna,
                          &alpha, &beta, &gamma,
                          &xiq, &etq, &wq);

        let mut amat = amat.lock().unwrap();

        for k in 0..npts {
            amat[(k, j)] = dlp[k];
        }
    });

    let amat_final = amat.into_inner().unwrap();

    let decomp = amat_final.lu();

    let f = decomp.solve(&rhs).expect("Linear resolution failed");

    println!("Linear system solved!");

    let phi_val = lsdlpp_3d(npts, nelm, mint, &f, &dfdn, &p, &n, &vna, &alpha, &beta, &gamma, &xiq, &etq, &wq, p0);

    phi_val

}

///Calculates the total KE of the fluid due to the movement of 1 body.
pub fn ke_1body(body :&Body, ndiv :u32, nq :usize, mint :usize) -> f64 {

    let f = f_1body(&body, ndiv, nq, mint);

    let s = body.shape;
    let req = 1.0 / (s[0] * s[1] * s[2]).powf(1.0/3.0);

    let orientation = UnitQuaternion::from_quaternion(body.orientation);
    let (nelm, npts, p, n) = ellip_gridder(ndiv, req, body.shape, body.position, orientation);

    let (zz, ww) = gauss_leg(nq);
    let (xiq, etq, wq) = gauss_trgl(mint);

    let (alpha, beta, gamma) = abc_vec(nelm, &p, &n);

    let (vna, _vlm, _sa) = elm_geom(npts, nelm, mint,
                                    &p, &n,
                                    &alpha, &beta, &gamma,
                                    &xiq, &etq, &wq);

    let dfdn = dfdn_single(&body.position, &body.linear_velocity(), &body.angular_velocity().imag(), npts, &p, &vna);


    let ke = ke_3d(npts, nelm, mint, &f, &dfdn,&p, &n, &vna, &alpha, &beta, &gamma, &xiq, &etq, &wq);

    ke
}

///Calculates the correct f for 2 given bodies by solving the BEM.
pub fn f_2body(body1 :&Body, body2 :&Body, ndiv :u32, nq :usize, mint :usize) -> DVector<f64> {

    let s1 = body1.shape;
    let req1 = 1.0 / (s1[0] * s1[1] * s1[2]).powf(1.0/3.0);

    let orientation1 = UnitQuaternion::from_quaternion(body1.orientation);
    let (nelm1, npts1,p1, n1) = ellip_gridder(ndiv, req1, body1.shape, body1.position, orientation1);

    let s2 = body2.shape;
    let req2 = 1.0 / (s2[0] * s2[1] * s2[2]).powf(1.0/3.0);

    let orientation2 = UnitQuaternion::from_quaternion(body2.orientation);
    let (nelm2, npts2, p2, n2) = ellip_gridder(ndiv, req2, body2.shape, body2.position, orientation2);

    let (nelm, npts, p, n) = combiner(nelm1, nelm2, npts1, npts2, &p1, &p2, &n1, &n2);

    println!("Grids created");

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
    }
    println!("Geometry arrays created.");

    let dfdn_1 = dfdn_single(&body1.position, &body1.linear_velocity(), &body1.angular_velocity().imag(), npts1, &p1, &vna1);
    let dfdn_2 = dfdn_single(&body2.position, &body2.linear_velocity(), &body2.angular_velocity().imag(), npts2, &p2, &vna2);

    let dfdn = vec_concat(&dfdn_1, &dfdn_2);

    //Calculate RHS of linear equation

    let rhs = lslp_3d(npts, nelm, mint, nq,
                      &dfdn, &p, &n, &vna,
                      &alpha, &beta, &gamma,
                      &xiq, &etq, &wq, &zz, &ww);

    println!("RHS is calculated");


    let amat_1 = DMatrix::zeros(npts, npts);
    let amat = Mutex::from(amat_1);

    let js = (0..npts).collect::<Vec<usize>>();

    println!("Computing columns of influence matrix");

    js.par_iter().progress_count(npts as u64).for_each(|&j|  {
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
    println!("Matrix created");

    let decomp = amat_final.lu();
    println!("Matrix decomposed");

    let f = decomp.solve(&rhs).expect("Linear resolution failed");
    println!("Linear system solved!");

    f
}

pub fn f_2body_serial(body1 :&Body, body2 :&Body, ndiv :u32, nq :usize, mint :usize) -> DVector<f64> {

    let s1 = body1.shape;
    let req1 = 1.0 / (s1[0] * s1[1] * s1[2]).powf(1.0/3.0);

    let orientation1 = UnitQuaternion::from_quaternion(body1.orientation);
    let (nelm1, npts1,p1, n1) = ellip_gridder(ndiv, req1, body1.shape, body1.position, orientation1);

    let s2 = body2.shape;
    let req2 = 1.0 / (s2[0] * s2[1] * s2[2]).powf(1.0/3.0);

    let orientation2 = UnitQuaternion::from_quaternion(body2.orientation);
    let (nelm2, npts2, p2, n2) = ellip_gridder(ndiv, req2, body2.shape, body2.position, orientation2);

    let (nelm, npts, p, n) = combiner(nelm1, nelm2, npts1, npts2, &p1, &p2, &n1, &n2);

    println!("Grids created");

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
    }
    println!("Geometry arrays created.");

    let dfdn_1 = dfdn_single(&body1.position, &body1.linear_velocity(), &body1.angular_velocity().imag(), npts1, &p1, &vna1);
    let dfdn_2 = dfdn_single(&body2.position, &body2.linear_velocity(), &body2.angular_velocity().imag(), npts2, &p2, &vna2);

    let dfdn = vec_concat(&dfdn_1, &dfdn_2);

    //Calculate RHS of linear equation

    let rhs = lslp_3d(npts, nelm, mint, nq,
                      &dfdn, &p, &n, &vna,
                      &alpha, &beta, &gamma,
                      &xiq, &etq, &wq, &zz, &ww);

    println!("RHS is calculated");


    let mut amat_1 = DMatrix::zeros(npts, npts);
    let mut amat = Mutex::from(amat_1);

    let js = (0..npts).collect::<Vec<usize>>();

    println!("Computing columns of influence matrix");

    js.iter().progress_count(npts as u64).for_each(|&j|  {
        // println!("Computing column {} of the influence matrix", j);
        let mut q = DVector::zeros(npts);
        q[j] = 1.0;

        let dlp = ldlp_3d(npts, nelm, mint,
                          &q, &p, &n, &vna,
                          &alpha, &beta, &gamma,
                          &xiq, &etq, &wq);

        let mut amat = amat.lock().unwrap();
        for k in 0..npts {

            amat[(k, j)] = dlp[k];
        }
        q[j] = 0.0;
    });

    let amat_final = amat.into_inner().unwrap();
    println!("Matrix created");

    let decomp = amat_final.lu();
    println!("Matrix decomposed");

    let f = decomp.solve(&rhs).expect("Linear resolution failed");
    println!("Linear system solved!");

    f
}

