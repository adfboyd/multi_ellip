#![allow(unused_doc_comments, unused_variables, unused_assignments, unused_mut, unused_parens)]
use std::f64::consts::PI;
use nalgebra::{DMatrix, DVector, Matrix3, Vector3, Vector6};
use rayon::prelude::*;

///The free-space green's function for potential flow in 3d, and its derivative.
pub fn lgf_3d_fs(x :&Vector3<f64>, x0 :&Vector3<f64>) -> (f64, Vector3<f64>) {

    let pi = PI;
    let pi4 = pi * 4.0;

    let dx = x - x0;
    let r = dx.norm();
    let g = 1.0 / (pi4 * r);

    let mut dg = Vector3::zeros();
    let den = pi4 * r * r * r;
    for i in 0..3 {
        dg[i] = - dx[i] / den;
    }

    (g, dg)
}

pub fn d_lgf_3d_fs(x :&Vector3<f64>, x0 :&Vector3<f64>, xi :&Vector3<f64>) -> (f64, Vector3<f64>) {

    let (_, dg) = lgf_3d_fs(x, x0);

    let ri = x - x0;
    let r = ri.norm();
    let dgdx = dg.dot(&xi);

    let mut dd_g = Matrix3::zeros();

    for i in 0..3 {
        for j in 0..3 {
            let mut delta = 0f64;
            if i==j {
                delta = 1f64;
            }
            dd_g[(i, j)] = delta / r.powi(3) - ( 3.0 / (4.0 * PI * r.powi(5))) * ri[i] * ri[j];
        }
    }


    let dd_gdx2 = dd_g * xi;
    (dgdx, dd_gdx2)
}

    pub fn d_lgf_3d_fs_full(x :&Vector3<f64>, x0 :&Vector3<f64>) -> (Vector3<f64>, Matrix3<f64>) {

        let ri = x - x0;
        let r = ri.norm();
        let den = PI * 4.0 * r * r * r;

        let mut dg = Vector3::zeros();
        for i in 0..3 {
            dg[i] = - ri[i] / den;
        }

        let mut dd_g = Matrix3::zeros();

        for i in 0..3 {
            for j in 0..3 {
                let mut delta = 0f64;
                if i==j {
                    delta = 1f64;
                }
                dd_g[(i, j)] = (-1.0 / (4.0 * PI )) * (delta / r.powi(3) - ( 3.0 / (r.powi(5))) * ri[i] * ri[j]);
            }
        }

        (dg, dd_g)
    }

///Interpolates over the triangle, also interpolates the force f.
pub fn lslp_3d_interp(p1 :Vector3<f64>,
                p2 :Vector3<f64>,
                p3 :Vector3<f64>,
                p4 :Vector3<f64>,
                p5 :Vector3<f64>,
                p6 :Vector3<f64>,
                f1 :f64, f2 :f64, f3 :f64, f4 :f64, f5 :f64, f6 :f64,
                al :f64, be :f64, ga :f64,
                xi :f64, eta :f64) ->
                (Vector3<f64>, f64, f64) {

    let (alc, bec, gac) = (1.0-al, 1.0-be, 1.0-ga);
    let (alalc, bebec, gagac) = (al * alc, be * bec, ga * gac);

    //Evaluate Basis functions
    let ph2 = xi * (xi - al + eta * (al- ga)/gac)/alc;
    let ph3 = eta * (eta - be + xi * (be + ga - 1.0)/ga)/bec;
    let ph4 = xi * (1.0 - xi - eta)/alalc;
    let ph5 = xi * eta / gagac;
    let ph6 = eta * (1.0 - xi -eta) / bebec;
    let ph1 = 1.0 - ph2 - ph3 - ph4 - ph5 - ph6;

    //Interpolate position vector
    let x = p1[0]*ph1 + p2[0]*ph2 + p3[0]*ph3 + p4[0]*ph4 + p5[0]*ph5 + p6[0]*ph6;
    let y = p1[1]*ph1 + p2[1]*ph2 + p3[1]*ph3 + p4[1]*ph4 + p5[1]*ph5 + p6[1]*ph6;
    let z = p1[2]*ph1 + p2[2]*ph2 + p3[2]*ph3 + p4[2]*ph4 + p5[2]*ph5 + p6[2]*ph6;
    let xvec = Vector3::new(x, y, z);

    //Interpolate the density f
    let f = f1*ph1 + f2 * ph2 + f3 * ph3 + f4 * ph4 + f5 * ph5 + f6 * ph6;

    //Evaluate xi derivatives of basis functions
    let dph2 = (2.0 * xi - al + eta * (al - ga)/gac) / alc;
    let dph3 = eta * (be + ga - 1.0) / (ga * bec);
    let dph4 = (1.0 - 2.0 * xi - eta) / alalc;
    let dph5 = eta / gagac;
    let dph6 = -eta / bebec;
    let dph1 = -dph2 - dph3 - dph4 - dph5 - dph6;

    //Compute dx/dxi from xi derivatives of phi
    let dx_dxi = p1[0]*dph1 + p2[0]*dph2 + p3[0]*dph3 + p4[0]*dph4 + p5[0]*dph5 + p6[0]*dph6;
    let dy_dxi = p1[1]*dph1 + p2[1]*dph2 + p3[1]*dph3 + p4[1]*dph4 + p5[1]*dph5 + p6[1]*dph6;
    let dz_dxi = p1[2]*dph1 + p2[2]*dph2 + p3[2]*dph3 + p4[2]*dph4 + p5[2]*dph5 + p6[2]*dph6;
    let ddxi = Vector3::new(dx_dxi, dy_dxi, dz_dxi);

    //Evaluate eta derivatives of basis functions
    let pph2 = xi * (al - ga) / (alc * gac);
    let pph3 = (2.0 * eta - be + xi *(be + ga - 1.0) / ga) / bec;
    let pph4 = -xi / alalc;
    let pph5 = xi / gagac;
    let pph6 = (1.0 - xi - 2.0 * eta) / bebec;
    let pph1 = -pph2 - pph3 - pph4 - pph5 - pph6;

    //Compute Dx/Deta from eta derivatives of phi
    let dx_det = p1[0]*pph1 + p2[0]*pph2 + p3[0]*pph3 + p4[0]*pph4 + p5[0]*pph5 + p6[0]*pph6;
    let dy_det = p1[1]*pph1 + p2[1]*pph2 + p3[1]*pph3 + p4[1]*pph4 + p5[1]*pph5 + p6[1]*pph6;
    let dz_det = p1[2]*pph1 + p2[2]*pph2 + p3[2]*pph3 + p4[2]*pph4 + p5[2]*pph5 + p6[2]*pph6;
    let ddet = Vector3::new(dx_det, dy_det, dz_det);

    let vn = ddxi.cross(&ddet);
    let hs = vn.norm();

    // vn = vn.normalize();

    (xvec, hs, f)
}

///Interpolates over the triangle, including the force q and the normal vector.
pub fn ldlp_3d_interp(p1 :Vector3<f64>,
                      p2 :Vector3<f64>,
                      p3 :Vector3<f64>,
                      p4 :Vector3<f64>,
                      p5 :Vector3<f64>,
                      p6 :Vector3<f64>,
                      v1 :Vector3<f64>,
                      v2 :Vector3<f64>,
                      v3 :Vector3<f64>,
                      v4 :Vector3<f64>,
                      v5 :Vector3<f64>,
                      v6 :Vector3<f64>,
                      q1 :f64, q2 :f64, q3 :f64, q4 :f64, q5 :f64, q6 :f64,
                      al :f64, be :f64, ga :f64,
                      xi :f64, eta :f64) ->
                      (Vector3<f64>, Vector3<f64>, f64, f64) {

    let (alc, bec, gac) = (1.0-al, 1.0-be, 1.0-ga);
    let (alalc, bebec, gagac) = (al * alc, be * bec, ga * gac);

    //Evaluate Basis functions
    let ph2 = xi * (xi - al + eta * (al- ga)/gac)/alc;
    let ph3 = eta * (eta - be + xi * (be + ga - 1.0)/ga)/bec;
    let ph4 = xi * (1.0 - xi - eta)/alalc;
    let ph5 = xi * eta / gagac;
    let ph6 = eta * (1.0 - xi -eta) / bebec;
    let ph1 = 1.0 - ph2 - ph3 - ph4 - ph5 - ph6;

    //Interpolate position vector
    let x = p1[0]*ph1 + p2[0]*ph2 + p3[0]*ph3 + p4[0]*ph4 + p5[0]*ph5 + p6[0]*ph6;
    let y = p1[1]*ph1 + p2[1]*ph2 + p3[1]*ph3 + p4[1]*ph4 + p5[1]*ph5 + p6[1]*ph6;
    let z = p1[2]*ph1 + p2[2]*ph2 + p3[2]*ph3 + p4[2]*ph4 + p5[2]*ph5 + p6[2]*ph6;
    let xvec = Vector3::new(x, y, z);

    //Interpolate the density q
    let q = q1*ph1 + q2 * ph2 + q3 * ph3 + q4 * ph4 + q5 * ph5 + q6 * ph6;

    //Interpolate the normal vectors (vx, vy, vz)
    let vx = v1[0]*ph1 + v2[0]*ph2 + v3[0]*ph3 + v4[0]*ph4 + v5[0]*ph5 + v6[0]*ph6;
    let vy = v1[1]*ph1 + v2[1]*ph2 + v3[1]*ph3 + v4[1]*ph4 + v5[1]*ph5 + v6[1]*ph6;
    let vz = v1[2]*ph1 + v2[2]*ph2 + v3[2]*ph3 + v4[2]*ph4 + v5[2]*ph5 + v6[2]*ph6;
    let v = Vector3::new(vx, vy, vz);

    //Evaluate xi derivatives of basis functions
    let dph2 = (2.0 * xi - al + eta * (al - ga)/gac) / alc;
    let dph3 = eta * (be + ga - 1.0) / (ga * bec);
    let dph4 = (1.0 - 2.0 * xi - eta) / alalc;
    let dph5 = eta / gagac;
    let dph6 = -eta / bebec;
    let dph1 = -dph2 - dph3 - dph4 - dph5 - dph6;

    //Compute dx/dxi from xi derivatives of phi
    let dx_dxi = p1[0]*dph1 + p2[0]*dph2 + p3[0]*dph3 + p4[0]*dph4 + p5[0]*dph5 + p6[0]*dph6;
    let dy_dxi = p1[1]*dph1 + p2[1]*dph2 + p3[1]*dph3 + p4[1]*dph4 + p5[1]*dph5 + p6[1]*dph6;
    let dz_dxi = p1[2]*dph1 + p2[2]*dph2 + p3[2]*dph3 + p4[2]*dph4 + p5[2]*dph5 + p6[2]*dph6;
    let ddxi = Vector3::new(dx_dxi, dy_dxi, dz_dxi);

    //Evaluate eta derivatives of basis functions
    let pph2 = xi * (al - ga) / (alc * gac);
    let pph3 = (2.0 * eta - be + xi *(be + ga - 1.0) / ga) / bec;
    let pph4 = -xi / alalc;
    let pph5 = xi / gagac;
    let pph6 = (1.0 - xi - 2.0 * eta) / bebec;
    let pph1 = -pph2 - pph3 - pph4 - pph5 - pph6;

    //Compute Dx/Deta from eta derivatives of phi
    let dx_det = p1[0]*pph1 + p2[0]*pph2 + p3[0]*pph3 + p4[0]*pph4 + p5[0]*pph5 + p6[0]*pph6;
    let dy_det = p1[1]*pph1 + p2[1]*pph2 + p3[1]*pph3 + p4[1]*pph4 + p5[1]*pph5 + p6[1]*pph6;
    let dz_det = p1[2]*pph1 + p2[2]*pph2 + p3[2]*pph3 + p4[2]*pph4 + p5[2]*pph5 + p6[2]*pph6;
    let ddet = Vector3::new(dx_det, dy_det, dz_det);

    let vn = ddxi.cross(&ddet);
    let hs = vn.norm();

    // vn = vn.normalize();

    (xvec, v, hs, q)
}

///Integrates the single-layer potential over element k from point x0.
pub fn lslp_3d_integral(x0 :Vector3<f64>, k :usize,
                        mint :usize, f :&DVector<f64>,
                        p :&DMatrix<f64>, n :&DMatrix<usize>,
                        alpha :&DVector<f64>, beta :&DVector<f64>, gamma :&DVector<f64>,
                        xiq :&DVector<f64>, etq :&DVector<f64>, wq :&DVector<f64>) -> (f64, f64) {

    let mut area = 0.0;
    let mut slp = 0.0;

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

    let (f1, f2, f3, f4, f5, f6) = (f[i1], f[i2], f[i3], f[i4], f[i5], f[i6]);

    let (al, be, ga) = (alpha[k], beta[k], gamma[k]);

    //Loop over integration points
    for i in 0..mint {

        let (xi, eta) = (xiq[i], etq[i]);

        let (x, hs, f_int) = lslp_3d_interp(p1, p2, p3, p4, p5, p6,
                                           f1, f2, f3, f4, f5, f6,
                                           al, be, ga, xi, eta);

        //Compute Greens fn
        let (g, _dg) = lgf_3d_fs(&x, &x0);

        //Apply triangle quadrature
        let cf = 0.5 * hs * wq[i];

        area += cf;
        slp += f_int * g * cf;
    }
    (slp, area)
}

///Integrates the double-layer potential over element k from point x0.

pub fn ldlp_3d_integral (x0 :Vector3<f64>, k :usize,
                         mint :usize, q :&DVector<f64>, q0 :f64,
                         p :&DMatrix<f64>, n :&DMatrix<usize>,
                         vna :&DMatrix<f64>, alpha :&DVector<f64>, beta :&DVector<f64>, gamma :&DVector<f64>,
                         xiq :&DVector<f64>, etq :&DVector<f64>, wq :&DVector<f64>) -> (f64, f64) {

    let mut area = 0.0;
    let mut ptl = 0.0;

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

    let (q1, q2, q3, q4, q5, q6) = (q[i1], q[i2], q[i3], q[i4], q[i5], q[i6]);

    let (al, be, ga) = (alpha[k], beta[k], gamma[k]);

    for i in 0..mint {

        let (xi, eta) = (xiq[i], etq[i]);

        let (xvec, v, hs, qint) = ldlp_3d_interp(p1, p2, p3, p4, p5, p6,
                                                 vna1, vna2, vna3, vna4, vna5, vna6,
                                                    q1, q2, q3, q4, q5, q6,
                                                    al, be, ga, xi, eta);

        let (_g, dg) = lgf_3d_fs(&xvec, &x0);

        let cf = 0.5 * hs * wq[i];

        area += cf;
        ptl += (qint - q0) * v.dot(&dg) * cf;
    }

    (ptl, area)
}

///Integrates the singular potential over the triangle in the case that the integral is singular (ie x0 is part of element k).
///Compute the laplace single-layer potential over a flat triangle with points p1-3
///Integrate in local polar coordinates with origin at p1
pub fn lslp_3d_integral_sing(ngl :usize,
                             p1 :Vector3<f64>, p2 :Vector3<f64>, p3 :Vector3<f64>,
                             f1 :f64, f2 :f64, f3 :f64,
                             zz :&DVector<f64>, ww :&DVector<f64>) -> (f64, f64) {


    //Compute triangle area and surface metric

    let vn = (p2 - p1).cross(&(p3 - p1));

    let area = 0.5 * vn.norm();
    let hs = 2.0 * area;

    let mut asm = 0.0;
    let mut slp = 0.0;
    let pi = PI;

    for i in 0..ngl {

        let ph = (pi / 4.0) * (1.0 + zz[i]);
        let cph = ph.cos();
        let sph = ph.sin();

        let rmax = 1.0 / (cph + sph);
        let rmaxh = 0.5 * rmax;

        let mut bsm = 0.0;
        let mut btl = 0.0;

        for j in 0..ngl {

            let r = rmaxh * (1.0 + zz[j]);

            let (xi, et) = (r * cph, r * sph);
            let zt = 1.0 - xi - et;

            let x = p1[0] * zt + p2[0] * xi + p3[0] * et;
            let y = p1[1] * zt + p2[1] * xi + p3[1] * et;
            let z = p1[2] * zt + p2[2] * xi + p3[2] * et;
            let f = f1 * zt + f2 * xi + f3 * et;

            let xvec = Vector3::new(x, y, z);

            let (g, _dg) = lgf_3d_fs(&xvec, &p1);

            let cf = r * ww[j];

            bsm += cf;
            btl += f * g * cf;
        }

        let cf = ww[i] * rmaxh;

        asm += bsm * cf;
        slp += btl * cf;
    }
    let cf = (pi / 4.0) * hs;

    asm = asm * cf;
    slp = slp * cf;

    //If all works, asm = area.
    (area, slp)
}

///Polar-coordinate singular integration for the double-layer potential
///when x0 = p1 lies on the element. Mirrors lslp_3d_integral_sing.
pub fn ldlp_3d_integral_sing(ngl: usize,
    p1: Vector3<f64>, p2: Vector3<f64>, p3: Vector3<f64>,
    q1: f64, q2: f64, q3: f64, q0: f64,
    v1: Vector3<f64>, v2: Vector3<f64>, v3: Vector3<f64>,
    zz: &DVector<f64>, ww: &DVector<f64>) -> f64 {

    let vn_flat = (p2 - p1).cross(&(p3 - p1));
    let hs = vn_flat.norm(); // = 2 * area_flat

    let mut dlp = 0.0;
    let pi = PI;

    for i in 0..ngl {
        let ph = (pi / 4.0) * (1.0 + zz[i]);
        let cph = ph.cos();
        let sph = ph.sin();
        let rmax = 1.0 / (cph + sph);
        let rmaxh = 0.5 * rmax;

        let mut btl = 0.0;

        for j in 0..ngl {
            let r = rmaxh * (1.0 + zz[j]);
            let (xi, et) = (r * cph, r * sph);
            let zt = 1.0 - xi - et;

            let x = p1[0]*zt + p2[0]*xi + p3[0]*et;
            let y = p1[1]*zt + p2[1]*xi + p3[1]*et;
            let z = p1[2]*zt + p2[2]*xi + p3[2]*et;
            let xvec = Vector3::new(x, y, z);

            let q_interp = q1*zt + q2*xi + q3*et;
            let vx = v1[0]*zt + v2[0]*xi + v3[0]*et;
            let vy = v1[1]*zt + v2[1]*xi + v3[1]*et;
            let vz = v1[2]*zt + v2[2]*xi + v3[2]*et;
            let v = Vector3::new(vx, vy, vz);

            let (_g, dg) = lgf_3d_fs(&xvec, &p1);

            // r from polar Jacobian regularises the O(1/r) near-singularity
            let cf = r * ww[j];

            btl += (q_interp - q0) * v.dot(&dg) * cf;
        }

        dlp += btl * ww[i] * rmaxh;
    }

    dlp * (pi / 4.0) * hs
}

///Computes the double-layer potential for a given initial condition q = phi(p)

pub fn ldlp_3d(npts :usize, nelm :usize,
               mint :usize, nq :usize, q :&DVector<f64>,
               p :&DMatrix<f64>, n :&DMatrix<usize>, vna :&DMatrix<f64>,
               alpha :&DVector<f64>, beta :&DVector<f64>, gamma :&DVector<f64>,
               xiq :&DVector<f64>, etq :&DVector<f64>, wq :&DVector<f64>,
               zz :&DVector<f64>, ww :&DVector<f64>) -> DVector<f64> {

    let mut dlp = DVector::zeros(npts);
    let tol = 1e-8;

    for i in 0..npts {

        let p0 = Vector3::new(p[(i, 0)], p[(i, 1)], p[(i, 2)]);
        let q0 = q[i];

        let mut ptl = 0.0;

        for k in 0..nelm {

            let i1 = n[(k, 0)];
            let i2 = n[(k, 1)];
            let i3 = n[(k, 2)];
            let i4 = n[(k, 3)];
            let i5 = n[(k, 4)];
            let i6 = n[(k, 5)];

            let (q1, q2, q3, q4, q5, q6) = (q[i1], q[i2], q[i3], q[i4], q[i5], q[i6]);

            let test = Vector6::new(q1.abs(), q2.abs(), q3.abs(), q4.abs(), q5.abs(), q6.abs()).sum();

            if test > tol {
                if i == i1 {
                    let pa = Vector3::new(p[(i1,0)], p[(i1,1)], p[(i1,2)]);
                    let pb = Vector3::new(p[(i2,0)], p[(i2,1)], p[(i2,2)]);
                    let pc = Vector3::new(p[(i3,0)], p[(i3,1)], p[(i3,2)]);
                    let va = Vector3::new(vna[(i1,0)], vna[(i1,1)], vna[(i1,2)]);
                    let vb = Vector3::new(vna[(i2,0)], vna[(i2,1)], vna[(i2,2)]);
                    let vc = Vector3::new(vna[(i3,0)], vna[(i3,1)], vna[(i3,2)]);
                    ptl += ldlp_3d_integral_sing(nq, pa, pb, pc, q1, q2, q3, q0, va, vb, vc, zz, ww);

                } else if i == i2 {
                    let (ia, ib, ic) = (i2, i3, i1);
                    let pa = Vector3::new(p[(ia,0)], p[(ia,1)], p[(ia,2)]);
                    let pb = Vector3::new(p[(ib,0)], p[(ib,1)], p[(ib,2)]);
                    let pc = Vector3::new(p[(ic,0)], p[(ic,1)], p[(ic,2)]);
                    let va = Vector3::new(vna[(ia,0)], vna[(ia,1)], vna[(ia,2)]);
                    let vb = Vector3::new(vna[(ib,0)], vna[(ib,1)], vna[(ib,2)]);
                    let vc = Vector3::new(vna[(ic,0)], vna[(ic,1)], vna[(ic,2)]);
                    ptl += ldlp_3d_integral_sing(nq, pa, pb, pc, q[ia], q[ib], q[ic], q0, va, vb, vc, zz, ww);

                } else if i == i3 {
                    let (ia, ib, ic) = (i3, i1, i2);
                    let pa = Vector3::new(p[(ia,0)], p[(ia,1)], p[(ia,2)]);
                    let pb = Vector3::new(p[(ib,0)], p[(ib,1)], p[(ib,2)]);
                    let pc = Vector3::new(p[(ic,0)], p[(ic,1)], p[(ic,2)]);
                    let va = Vector3::new(vna[(ia,0)], vna[(ia,1)], vna[(ia,2)]);
                    let vb = Vector3::new(vna[(ib,0)], vna[(ib,1)], vna[(ib,2)]);
                    let vc = Vector3::new(vna[(ic,0)], vna[(ic,1)], vna[(ic,2)]);
                    ptl += ldlp_3d_integral_sing(nq, pa, pb, pc, q[ia], q[ib], q[ic], q0, va, vb, vc, zz, ww);

                } else if i == i4 {
                    for &(ia, ib, ic) in &[(i4,i6,i1),(i4,i3,i6),(i4,i5,i3),(i4,i2,i5)] {
                        let pa = Vector3::new(p[(ia,0)], p[(ia,1)], p[(ia,2)]);
                        let pb = Vector3::new(p[(ib,0)], p[(ib,1)], p[(ib,2)]);
                        let pc = Vector3::new(p[(ic,0)], p[(ic,1)], p[(ic,2)]);
                        let va = Vector3::new(vna[(ia,0)], vna[(ia,1)], vna[(ia,2)]);
                        let vb = Vector3::new(vna[(ib,0)], vna[(ib,1)], vna[(ib,2)]);
                        let vc = Vector3::new(vna[(ic,0)], vna[(ic,1)], vna[(ic,2)]);
                        ptl += ldlp_3d_integral_sing(nq, pa, pb, pc, q[ia], q[ib], q[ic], q0, va, vb, vc, zz, ww);
                    }

                } else if i == i5 {
                    for &(ia, ib, ic) in &[(i5,i4,i2),(i5,i1,i4),(i5,i6,i1),(i5,i3,i6)] {
                        let pa = Vector3::new(p[(ia,0)], p[(ia,1)], p[(ia,2)]);
                        let pb = Vector3::new(p[(ib,0)], p[(ib,1)], p[(ib,2)]);
                        let pc = Vector3::new(p[(ic,0)], p[(ic,1)], p[(ic,2)]);
                        let va = Vector3::new(vna[(ia,0)], vna[(ia,1)], vna[(ia,2)]);
                        let vb = Vector3::new(vna[(ib,0)], vna[(ib,1)], vna[(ib,2)]);
                        let vc = Vector3::new(vna[(ic,0)], vna[(ic,1)], vna[(ic,2)]);
                        ptl += ldlp_3d_integral_sing(nq, pa, pb, pc, q[ia], q[ib], q[ic], q0, va, vb, vc, zz, ww);
                    }

                } else if i == i6 {
                    for &(ia, ib, ic) in &[(i6,i1,i4),(i6,i4,i2),(i6,i2,i5),(i6,i5,i3)] {
                        let pa = Vector3::new(p[(ia,0)], p[(ia,1)], p[(ia,2)]);
                        let pb = Vector3::new(p[(ib,0)], p[(ib,1)], p[(ib,2)]);
                        let pc = Vector3::new(p[(ic,0)], p[(ic,1)], p[(ic,2)]);
                        let va = Vector3::new(vna[(ia,0)], vna[(ia,1)], vna[(ia,2)]);
                        let vb = Vector3::new(vna[(ib,0)], vna[(ib,1)], vna[(ib,2)]);
                        let vc = Vector3::new(vna[(ic,0)], vna[(ic,1)], vna[(ic,2)]);
                        ptl += ldlp_3d_integral_sing(nq, pa, pb, pc, q[ia], q[ib], q[ic], q0, va, vb, vc, zz, ww);
                    }

                } else {
                    let (pptl, _arelm) = ldlp_3d_integral(p0, k, mint, q, q0,
                                                         p, n, vna,
                                                         alpha, beta, gamma,
                                                         xiq, etq, wq);
                    ptl += pptl;
                }
            }
        }
        dlp[i] += ptl - 0.5 * q0;
    }
    dlp

}

/// Assembles the full double-layer influence matrix `A` (npts x npts) directly.
///
/// This is equivalent to building the matrix column-by-column with
/// `ldlp_3d(e_j)` for each unit vector `e_j`, but avoids the O(N^3) cost of
/// scanning every element for every (column, field-point) pair: for a unit
/// source `e_j` the only elements that contribute are those containing node
/// `j`, so we precompute a node->element adjacency and iterate just those.
/// The per-element integration reuses the identical `ldlp_3d_integral` /
/// `ldlp_3d_integral_sing` routines, so the assembled matrix is bit-identical
/// to the column-by-column build. Columns are filled in parallel.
pub fn ldlp_3d_assemble(npts :usize, nelm :usize,
                        mint :usize, nq :usize,
                        p :&DMatrix<f64>, n :&DMatrix<usize>, vna :&DMatrix<f64>,
                        alpha :&DVector<f64>, beta :&DVector<f64>, gamma :&DVector<f64>,
                        xiq :&DVector<f64>, etq :&DVector<f64>, wq :&DVector<f64>,
                        zz :&DVector<f64>, ww :&DVector<f64>) -> DMatrix<f64> {

    // node -> elements containing it
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); npts];
    for k in 0..nelm {
        for c in 0..6 {
            adj[n[(k, c)]].push(k);
        }
    }

    let mut amat = DMatrix::zeros(npts, npts);

    // DMatrix is column-major, so each contiguous npts-chunk is one column.
    amat.as_mut_slice()
        .par_chunks_mut(npts)
        .enumerate()
        .for_each(|(j, col)| {
            // Unit source on node j.
            let mut q = DVector::zeros(npts);
            q[j] = 1.0;

            for i in 0..npts {
                let p0 = Vector3::new(p[(i, 0)], p[(i, 1)], p[(i, 2)]);
                let q0 = q[i];

                let mut ptl = 0.0;

                for &k in &adj[j] {
                    let i1 = n[(k, 0)];
                    let i2 = n[(k, 1)];
                    let i3 = n[(k, 2)];
                    let i4 = n[(k, 3)];
                    let i5 = n[(k, 4)];
                    let i6 = n[(k, 5)];

                    let (q1, q2, q3) = (q[i1], q[i2], q[i3]);

                    if i == i1 {
                        let pa = Vector3::new(p[(i1,0)], p[(i1,1)], p[(i1,2)]);
                        let pb = Vector3::new(p[(i2,0)], p[(i2,1)], p[(i2,2)]);
                        let pc = Vector3::new(p[(i3,0)], p[(i3,1)], p[(i3,2)]);
                        let va = Vector3::new(vna[(i1,0)], vna[(i1,1)], vna[(i1,2)]);
                        let vb = Vector3::new(vna[(i2,0)], vna[(i2,1)], vna[(i2,2)]);
                        let vc = Vector3::new(vna[(i3,0)], vna[(i3,1)], vna[(i3,2)]);
                        ptl += ldlp_3d_integral_sing(nq, pa, pb, pc, q1, q2, q3, q0, va, vb, vc, zz, ww);

                    } else if i == i2 {
                        let (ia, ib, ic) = (i2, i3, i1);
                        let pa = Vector3::new(p[(ia,0)], p[(ia,1)], p[(ia,2)]);
                        let pb = Vector3::new(p[(ib,0)], p[(ib,1)], p[(ib,2)]);
                        let pc = Vector3::new(p[(ic,0)], p[(ic,1)], p[(ic,2)]);
                        let va = Vector3::new(vna[(ia,0)], vna[(ia,1)], vna[(ia,2)]);
                        let vb = Vector3::new(vna[(ib,0)], vna[(ib,1)], vna[(ib,2)]);
                        let vc = Vector3::new(vna[(ic,0)], vna[(ic,1)], vna[(ic,2)]);
                        ptl += ldlp_3d_integral_sing(nq, pa, pb, pc, q[ia], q[ib], q[ic], q0, va, vb, vc, zz, ww);

                    } else if i == i3 {
                        let (ia, ib, ic) = (i3, i1, i2);
                        let pa = Vector3::new(p[(ia,0)], p[(ia,1)], p[(ia,2)]);
                        let pb = Vector3::new(p[(ib,0)], p[(ib,1)], p[(ib,2)]);
                        let pc = Vector3::new(p[(ic,0)], p[(ic,1)], p[(ic,2)]);
                        let va = Vector3::new(vna[(ia,0)], vna[(ia,1)], vna[(ia,2)]);
                        let vb = Vector3::new(vna[(ib,0)], vna[(ib,1)], vna[(ib,2)]);
                        let vc = Vector3::new(vna[(ic,0)], vna[(ic,1)], vna[(ic,2)]);
                        ptl += ldlp_3d_integral_sing(nq, pa, pb, pc, q[ia], q[ib], q[ic], q0, va, vb, vc, zz, ww);

                    } else if i == i4 {
                        for &(ia, ib, ic) in &[(i4,i6,i1),(i4,i3,i6),(i4,i5,i3),(i4,i2,i5)] {
                            let pa = Vector3::new(p[(ia,0)], p[(ia,1)], p[(ia,2)]);
                            let pb = Vector3::new(p[(ib,0)], p[(ib,1)], p[(ib,2)]);
                            let pc = Vector3::new(p[(ic,0)], p[(ic,1)], p[(ic,2)]);
                            let va = Vector3::new(vna[(ia,0)], vna[(ia,1)], vna[(ia,2)]);
                            let vb = Vector3::new(vna[(ib,0)], vna[(ib,1)], vna[(ib,2)]);
                            let vc = Vector3::new(vna[(ic,0)], vna[(ic,1)], vna[(ic,2)]);
                            ptl += ldlp_3d_integral_sing(nq, pa, pb, pc, q[ia], q[ib], q[ic], q0, va, vb, vc, zz, ww);
                        }

                    } else if i == i5 {
                        for &(ia, ib, ic) in &[(i5,i4,i2),(i5,i1,i4),(i5,i6,i1),(i5,i3,i6)] {
                            let pa = Vector3::new(p[(ia,0)], p[(ia,1)], p[(ia,2)]);
                            let pb = Vector3::new(p[(ib,0)], p[(ib,1)], p[(ib,2)]);
                            let pc = Vector3::new(p[(ic,0)], p[(ic,1)], p[(ic,2)]);
                            let va = Vector3::new(vna[(ia,0)], vna[(ia,1)], vna[(ia,2)]);
                            let vb = Vector3::new(vna[(ib,0)], vna[(ib,1)], vna[(ib,2)]);
                            let vc = Vector3::new(vna[(ic,0)], vna[(ic,1)], vna[(ic,2)]);
                            ptl += ldlp_3d_integral_sing(nq, pa, pb, pc, q[ia], q[ib], q[ic], q0, va, vb, vc, zz, ww);
                        }

                    } else if i == i6 {
                        for &(ia, ib, ic) in &[(i6,i1,i4),(i6,i4,i2),(i6,i2,i5),(i6,i5,i3)] {
                            let pa = Vector3::new(p[(ia,0)], p[(ia,1)], p[(ia,2)]);
                            let pb = Vector3::new(p[(ib,0)], p[(ib,1)], p[(ib,2)]);
                            let pc = Vector3::new(p[(ic,0)], p[(ic,1)], p[(ic,2)]);
                            let va = Vector3::new(vna[(ia,0)], vna[(ia,1)], vna[(ia,2)]);
                            let vb = Vector3::new(vna[(ib,0)], vna[(ib,1)], vna[(ib,2)]);
                            let vc = Vector3::new(vna[(ic,0)], vna[(ic,1)], vna[(ic,2)]);
                            ptl += ldlp_3d_integral_sing(nq, pa, pb, pc, q[ia], q[ib], q[ic], q0, va, vb, vc, zz, ww);
                        }

                    } else {
                        let (pptl, _arelm) = ldlp_3d_integral(p0, k, mint, &q, q0,
                                                             p, n, vna,
                                                             alpha, beta, gamma,
                                                             xiq, etq, wq);
                        ptl += pptl;
                    }
                }

                col[i] = ptl - 0.5 * q0;
            }
        });

    amat
}

/// The six quadratic shape functions and their `xi`/`eta` derivatives at a
/// parametric point, for the curvilinear element parameters (al, be, ga).
/// Matches the basis used in `ldlp_3d_interp`.
#[inline]
fn ldlp_3d_shape(al :f64, be :f64, ga :f64, xi :f64, eta :f64)
    -> ([f64; 6], [f64; 6], [f64; 6]) {

    let (alc, bec, gac) = (1.0 - al, 1.0 - be, 1.0 - ga);
    let (alalc, bebec, gagac) = (al * alc, be * bec, ga * gac);

    // Values.
    let ph2 = xi * (xi - al + eta * (al - ga) / gac) / alc;
    let ph3 = eta * (eta - be + xi * (be + ga - 1.0) / ga) / bec;
    let ph4 = xi * (1.0 - xi - eta) / alalc;
    let ph5 = xi * eta / gagac;
    let ph6 = eta * (1.0 - xi - eta) / bebec;
    let ph1 = 1.0 - ph2 - ph3 - ph4 - ph5 - ph6;

    // d/d(xi).
    let dph2 = (2.0 * xi - al + eta * (al - ga) / gac) / alc;
    let dph3 = eta * (be + ga - 1.0) / (ga * bec);
    let dph4 = (1.0 - 2.0 * xi - eta) / alalc;
    let dph5 = eta / gagac;
    let dph6 = -eta / bebec;
    let dph1 = -dph2 - dph3 - dph4 - dph5 - dph6;

    // d/d(eta).
    let pph2 = xi * (al - ga) / (alc * gac);
    let pph3 = (2.0 * eta - be + xi * (be + ga - 1.0) / ga) / bec;
    let pph4 = -xi / alalc;
    let pph5 = xi / gagac;
    let pph6 = (1.0 - xi - 2.0 * eta) / bebec;
    let pph1 = -pph2 - pph3 - pph4 - pph5 - pph6;

    (
        [ph1, ph2, ph3, ph4, ph5, ph6],
        [dph1, dph2, dph3, dph4, dph5, dph6],
        [pph1, pph2, pph3, pph4, pph5, pph6],
    )
}

/// Per-element quadrature geometry, precomputed once and reused across all
/// (non-singular) field points. For each quadrature point it stores the surface
/// position, the interpolated normal, the quadrature factor `0.5 * hs * wq`, and
/// the six shape-function values.
struct ElemQuad {
    nodes: [usize; 6],
    x: Vec<Vector3<f64>>,
    v: Vec<Vector3<f64>>,
    cf: Vec<f64>,
    ph: Vec<[f64; 6]>,
}

/// Precompute the quadrature geometry of element `k`.
fn elem_quad(k :usize, mint :usize,
             p :&DMatrix<f64>, n :&DMatrix<usize>, vna :&DMatrix<f64>,
             alpha :&DVector<f64>, beta :&DVector<f64>, gamma :&DVector<f64>,
             xiq :&DVector<f64>, etq :&DVector<f64>, wq :&DVector<f64>) -> ElemQuad {

    let nodes = [n[(k, 0)], n[(k, 1)], n[(k, 2)], n[(k, 3)], n[(k, 4)], n[(k, 5)]];
    let pv: [Vector3<f64>; 6] = std::array::from_fn(|m| {
        Vector3::new(p[(nodes[m], 0)], p[(nodes[m], 1)], p[(nodes[m], 2)])
    });
    let vv: [Vector3<f64>; 6] = std::array::from_fn(|m| {
        Vector3::new(vna[(nodes[m], 0)], vna[(nodes[m], 1)], vna[(nodes[m], 2)])
    });
    let (al, be, ga) = (alpha[k], beta[k], gamma[k]);

    let mut x = Vec::with_capacity(mint);
    let mut v = Vec::with_capacity(mint);
    let mut cf = Vec::with_capacity(mint);
    let mut ph_all = Vec::with_capacity(mint);

    for qi in 0..mint {
        let (xi, eta) = (xiq[qi], etq[qi]);
        let (ph, dph, pph) = ldlp_3d_shape(al, be, ga, xi, eta);

        let mut xvec = Vector3::zeros();
        let mut vvec = Vector3::zeros();
        let mut ddxi = Vector3::zeros();
        let mut ddet = Vector3::zeros();
        for m in 0..6 {
            xvec += ph[m] * pv[m];
            vvec += ph[m] * vv[m];
            ddxi += dph[m] * pv[m];
            ddet += pph[m] * pv[m];
        }
        let hs = ddxi.cross(&ddet).norm();

        x.push(xvec);
        v.push(vvec);
        cf.push(0.5 * hs * wq[qi]);
        ph_all.push(ph);
    }

    ElemQuad { nodes, x, v, cf, ph: ph_all }
}

/// The six per-basis double-layer contributions of one element to field point
/// `x0` (non-singular case). The kernel `v . grad G` is evaluated once per
/// quadrature point and shared across all six shape functions.
#[inline]
fn elem_basis_row(eq :&ElemQuad, x0 :&Vector3<f64>) -> [f64; 6] {
    let mut g = [0.0_f64; 6];
    for q in 0..eq.x.len() {
        let (_gf, dg) = lgf_3d_fs(&eq.x[q], x0);
        let kq = eq.v[q].dot(&dg) * eq.cf[q];
        let ph = &eq.ph[q];
        for m in 0..6 {
            g[m] += ph[m] * kq;
        }
    }
    g
}

/// Assembles only the *interaction* (off-diagonal-block) part of the influence
/// matrix: entries `A[i,j]` where field node `i` and source node `j` belong to
/// different bodies. The diagonal blocks (each body's self-influence) are left
/// zero, since they are time-invariant and cached separately.
///
/// Cross-body entries are always non-singular (the field point is never a node
/// of a source element on another body) and the `-q0` desingularisation term
/// vanishes (`q0 = q[i] = delta_ij = 0`). Assembly is element-centric: each
/// element's quadrature geometry is precomputed once, and at every field point
/// the kernel is evaluated once per quadrature point and shared across the six
/// shape functions, giving all six matrix entries from a single pass (vs the
/// previous column build, which re-integrated each element once per node).
/// `nptss` gives the per-body node counts (nodes are blocked by body).
pub fn ldlp_3d_assemble_interactions(npts :usize, nelm :usize, mint :usize,
                                     nptss :&[usize],
                                     p :&DMatrix<f64>, n :&DMatrix<usize>, vna :&DMatrix<f64>,
                                     alpha :&DVector<f64>, beta :&DVector<f64>, gamma :&DVector<f64>,
                                     xiq :&DVector<f64>, etq :&DVector<f64>, wq :&DVector<f64>) -> DMatrix<f64> {

    // node -> body index
    let mut node_body = vec![0_usize; npts];
    {
        let mut off = 0;
        for (b, &np) in nptss.iter().enumerate() {
            for t in 0..np {
                node_body[off + t] = b;
            }
            off += np;
        }
    }

    // Every node of an element shares its body; use the first node.
    let elem_body: Vec<usize> = (0..nelm).map(|k| node_body[n[(k, 0)]]).collect();

    // Precompute each element's quadrature geometry once.
    let eqs: Vec<ElemQuad> = (0..nelm)
        .into_par_iter()
        .map(|k| elem_quad(k, mint, p, n, vna, alpha, beta, gamma, xiq, etq, wq))
        .collect();

    // Build A^T column-major (column i of A^T = row i of A), parallel over field
    // points. Each thread owns one field-point row, so the scatter is race-free.
    let mut at = DMatrix::zeros(npts, npts);
    at.as_mut_slice()
        .par_chunks_mut(npts)
        .enumerate()
        .for_each(|(i, row_i)| {
            let ibody = node_body[i];
            let x0 = Vector3::new(p[(i, 0)], p[(i, 1)], p[(i, 2)]);
            for k in 0..nelm {
                // Same-body => self-block (cached separately); skip.
                if elem_body[k] == ibody {
                    continue;
                }
                let g = elem_basis_row(&eqs[k], &x0);
                let nd = &eqs[k].nodes;
                for m in 0..6 {
                    row_i[nd[m]] += g[m];
                }
            }
        });

    at.transpose()
}

///Computes the single-layer potential for a given f = d(phi)/dn

pub fn lslp_3d(npts :usize, nelm :usize,
               mint :usize, nq :usize, f :&DVector<f64>,
               p :&DMatrix<f64>, n :&DMatrix<usize>, vna :&DMatrix<f64>,
               alpha :&DVector<f64>, beta :&DVector<f64>, gamma :&DVector<f64>,
               xiq :&DVector<f64>, etq :&DVector<f64>, wq :&DVector<f64>,
               zz :&DVector<f64>, ww :&DVector<f64>) -> DVector<f64> {
    let tol = 1e-8;

    // Precompute each element's quadrature geometry once, reused across all
    // (non-singular) field points (the single layer needs position, Jacobian
    // and shape functions; the cached normal is simply unused here).
    let eqs: Vec<ElemQuad> = (0..nelm)
        .into_par_iter()
        .map(|k| elem_quad(k, mint, p, n, vna, alpha, beta, gamma, xiq, etq, wq))
        .collect();

    let slp_vals: Vec<f64> = (0..npts)
        .into_par_iter()
        .map(|i| {
            let p0 = Vector3::new(p[(i, 0)], p[(i, 1)], p[(i, 2)]);

            let mut srf_area = 0.0;
            let mut ptl = 0.0;

            for k in 0..nelm {

                let i1 = n[(k, 0)];
                let i2 = n[(k, 1)];
                let i3 = n[(k, 2)];
                let i4 = n[(k, 3)];
                let i5 = n[(k, 4)];
                let i6 = n[(k, 5)];

                let (f1, f2, f3, f4, f5, f6) = (f[i1], f[i2], f[i3], f[i4], f[i5], f[i6]);

                let test = Vector6::new(f1.abs(), f2.abs(), f3.abs(), f4.abs(), f5.abs(), f6.abs()).sum();

                if test > tol {
                    //Check if singular point is one of the corner nodes i1, i2, i3
                    if i == i1 {

                        let p1 = Vector3::new(p[(i1, 0)], p[(i1, 1)], p[(i1, 2)]);
                        let p2 = Vector3::new(p[(i2, 0)], p[(i2, 1)], p[(i2, 2)]);
                        let p3 = Vector3::new(p[(i3, 0)], p[(i3, 1)], p[(i3, 2)]);

                        let (f1, f2, f3) = (f[i1], f[i2], f[i3]);

                        let (pptl, arelm) = lslp_3d_integral_sing(nq, p1, p2, p3,
                                                                  f1, f2, f3,
                                                                  zz, ww);

                        ptl += pptl;
                        srf_area = srf_area + arelm;

                    }
                    else if i == i2 {
                        let (i1, i2, i3) = (i2, i3, i1);

                        let p1 = Vector3::new(p[(i1, 0)], p[(i1, 1)], p[(i1, 2)]);
                        let p2 = Vector3::new(p[(i2, 0)], p[(i2, 1)], p[(i2, 2)]);
                        let p3 = Vector3::new(p[(i3, 0)], p[(i3, 1)], p[(i3, 2)]);

                        let (f1, f2, f3) = (f[i1], f[i2], f[i3]);

                        let (pptl, arelm) = lslp_3d_integral_sing(nq, p1, p2, p3,
                                                                  f1, f2, f3,
                                                                  zz, ww);

                        ptl += pptl;
                        srf_area += arelm;
                    } else if i == i3 {
                        let (i1, i2, i3) = (i3, i1, i2);

                        let p1 = Vector3::new(p[(i1, 0)], p[(i1, 1)], p[(i1, 2)]);
                        let p2 = Vector3::new(p[(i2, 0)], p[(i2, 1)], p[(i2, 2)]);
                        let p3 = Vector3::new(p[(i3, 0)], p[(i3, 1)], p[(i3, 2)]);

                        let (f1, f2, f3) = (f[i1], f[i2], f[i3]);

                        let (pptl, arelm) = lslp_3d_integral_sing(nq, p1, p2, p3,
                                                                  f1, f2, f3,
                                                                  zz, ww);

                        ptl += pptl;
                        srf_area += arelm;
                    } //Check if the singular point is one of the edge nodes i4, i5, i6
                    else if i == i4 {
                        let (ia, ib, ic) = (i4, i6, i1);

                        let p1 = Vector3::new(p[(ia, 0)], p[(ia, 1)], p[(ia, 2)]);
                        let p2 = Vector3::new(p[(ib, 0)], p[(ib, 1)], p[(ib, 2)]);
                        let p3 = Vector3::new(p[(ic, 0)], p[(ic, 1)], p[(ic, 2)]);

                        let (f1, f2, f3) = (f[ia], f[ib], f[ic]);

                        let (pptl, arelm) = lslp_3d_integral_sing(nq, p1, p2, p3,
                                                                  f1, f2, f3,
                                                                  zz, ww);

                        ptl += pptl;
                        srf_area += arelm;

                        let (ia, ib, ic) = (i4, i3, i6);

                        let p1 = Vector3::new(p[(ia, 0)], p[(ia, 1)], p[(ia, 2)]);
                        let p2 = Vector3::new(p[(ib, 0)], p[(ib, 1)], p[(ib, 2)]);
                        let p3 = Vector3::new(p[(ic, 0)], p[(ic, 1)], p[(ic, 2)]);

                    let (f1, f2, f3) = (f[ia], f[ib], f[ic]);

                    let (pptl, arelm) = lslp_3d_integral_sing(nq, p1, p2, p3,
                                                              f1, f2, f3,
                                                              zz, ww);

                    ptl += pptl;
                    srf_area += arelm;

                    let (ia, ib, ic) = (i4, i5, i3);

                    let p1 = Vector3::new(p[(ia, 0)], p[(ia, 1)], p[(ia, 2)]);
                    let p2 = Vector3::new(p[(ib, 0)], p[(ib, 1)], p[(ib, 2)]);
                    let p3 = Vector3::new(p[(ic, 0)], p[(ic, 1)], p[(ic, 2)]);

                    let (f1, f2, f3) = (f[ia], f[ib], f[ic]);

                    let (pptl, arelm) = lslp_3d_integral_sing(nq, p1, p2, p3,
                                                              f1, f2, f3,
                                                              zz, ww);

                    ptl += pptl;
                    srf_area += arelm;

                    let (ia, ib, ic) = (i4, i2, i5);

                    let p1 = Vector3::new(p[(ia, 0)], p[(ia, 1)], p[(ia, 2)]);
                    let p2 = Vector3::new(p[(ib, 0)], p[(ib, 1)], p[(ib, 2)]);
                    let p3 = Vector3::new(p[(ic, 0)], p[(ic, 1)], p[(ic, 2)]);

                    let (f1, f2, f3) = (f[ia], f[ib], f[ic]);

                    let (pptl, arelm) = lslp_3d_integral_sing(nq, p1, p2, p3,
                                                              f1, f2, f3,
                                                              zz, ww);

                    ptl += pptl;
                    srf_area += arelm;

                } else if i == i5 {
                    let (ia, ib, ic) = (i5, i4, i2);

                    let p1 = Vector3::new(p[(ia, 0)], p[(ia, 1)], p[(ia, 2)]);
                    let p2 = Vector3::new(p[(ib, 0)], p[(ib, 1)], p[(ib, 2)]);
                    let p3 = Vector3::new(p[(ic, 0)], p[(ic, 1)], p[(ic, 2)]);

                    let (f1, f2, f3) = (f[ia], f[ib], f[ic]);

                    let (pptl, arelm) = lslp_3d_integral_sing(nq, p1, p2, p3,
                                                              f1, f2, f3,
                                                              zz, ww);

                    ptl += pptl;
                    srf_area += arelm;

                    let (ia, ib, ic) = (i5, i1, i4);

                    let p1 = Vector3::new(p[(ia, 0)], p[(ia, 1)], p[(ia, 2)]);
                    let p2 = Vector3::new(p[(ib, 0)], p[(ib, 1)], p[(ib, 2)]);
                    let p3 = Vector3::new(p[(ic, 0)], p[(ic, 1)], p[(ic, 2)]);

                    let (f1, f2, f3) = (f[ia], f[ib], f[ic]);

                    let (pptl, arelm) = lslp_3d_integral_sing(nq, p1, p2, p3,
                                                              f1, f2, f3,
                                                              zz, ww);

                    ptl += pptl;
                    srf_area += arelm;

                    let (ia, ib, ic) = (i5, i6, i1);

                    let p1 = Vector3::new(p[(ia, 0)], p[(ia, 1)], p[(ia, 2)]);
                    let p2 = Vector3::new(p[(ib, 0)], p[(ib, 1)], p[(ib, 2)]);
                    let p3 = Vector3::new(p[(ic, 0)], p[(ic, 1)], p[(ic, 2)]);

                    let (f1, f2, f3) = (f[ia], f[ib], f[ic]);

                    let (pptl, arelm) = lslp_3d_integral_sing(nq, p1, p2, p3,
                                                              f1, f2, f3,
                                                              zz, ww);

                    ptl += pptl;
                    srf_area += arelm;

                    let (ia, ib, ic) = (i5, i3, i6);

                    let p1 = Vector3::new(p[(ia, 0)], p[(ia, 1)], p[(ia, 2)]);
                    let p2 = Vector3::new(p[(ib, 0)], p[(ib, 1)], p[(ib, 2)]);
                    let p3 = Vector3::new(p[(ic, 0)], p[(ic, 1)], p[(ic, 2)]);

                    let (f1, f2, f3) = (f[ia], f[ib], f[ic]);

                    let (pptl, arelm) = lslp_3d_integral_sing(nq, p1, p2, p3,
                                                              f1, f2, f3,
                                                              zz, ww);

                    ptl += pptl;
                    srf_area += arelm;

                } else if i == i6 {
                    let (ia, ib, ic) = (i6, i1, i4);

                    let p1 = Vector3::new(p[(ia, 0)], p[(ia, 1)], p[(ia, 2)]);
                    let p2 = Vector3::new(p[(ib, 0)], p[(ib, 1)], p[(ib, 2)]);
                    let p3 = Vector3::new(p[(ic, 0)], p[(ic, 1)], p[(ic, 2)]);

                    let (f1, f2, f3) = (f[ia], f[ib], f[ic]);

                    let (pptl, arelm) = lslp_3d_integral_sing(nq, p1, p2, p3,
                                                              f1, f2, f3,
                                                              zz, ww);

                    ptl += pptl;
                    srf_area += arelm;

                    let (ia, ib, ic) = (i6, i4, i2);

                    let p1 = Vector3::new(p[(ia, 0)], p[(ia, 1)], p[(ia, 2)]);
                    let p2 = Vector3::new(p[(ib, 0)], p[(ib, 1)], p[(ib, 2)]);
                    let p3 = Vector3::new(p[(ic, 0)], p[(ic, 1)], p[(ic, 2)]);

                    let (f1, f2, f3) = (f[ia], f[ib], f[ic]);

                    let (pptl, arelm) = lslp_3d_integral_sing(nq, p1, p2, p3,
                                                              f1, f2, f3,
                                                              zz, ww);

                    ptl += pptl;
                    srf_area += arelm;

                    let (ia, ib, ic) = (i6, i2, i5);

                    let p1 = Vector3::new(p[(ia, 0)], p[(ia, 1)], p[(ia, 2)]);
                    let p2 = Vector3::new(p[(ib, 0)], p[(ib, 1)], p[(ib, 2)]);
                    let p3 = Vector3::new(p[(ic, 0)], p[(ic, 1)], p[(ic, 2)]);

                    let (f1, f2, f3) = (f[ia], f[ib], f[ic]);

                    let (pptl, arelm) = lslp_3d_integral_sing(nq, p1, p2, p3,
                                                              f1, f2, f3,
                                                              zz, ww);

                    ptl += pptl;
                    srf_area += arelm;

                    let (ia, ib, ic) = (i6, i5, i3);

                    let p1 = Vector3::new(p[(ia, 0)], p[(ia, 1)], p[(ia, 2)]);
                    let p2 = Vector3::new(p[(ib, 0)], p[(ib, 1)], p[(ib, 2)]);
                    let p3 = Vector3::new(p[(ic, 0)], p[(ic, 1)], p[(ic, 2)]);

                    let (f1, f2, f3) = (f[ia], f[ib], f[ic]);

                    let (pptl, arelm) = lslp_3d_integral_sing(nq, p1, p2, p3,
                                                              f1, f2, f3,
                                                              zz, ww);

                    ptl += pptl;
                    srf_area += arelm;
                } else //Do non-singular integral (precomputed element geometry)
                {
                    let eq = &eqs[k];
                    for q in 0..mint {
                        let (g, _dg) = lgf_3d_fs(&eq.x[q], &p0);
                        let ph = &eq.ph[q];
                        let f_int = f1 * ph[0] + f2 * ph[1] + f3 * ph[2]
                                  + f4 * ph[3] + f5 * ph[4] + f6 * ph[5];
                        ptl += g * f_int * eq.cf[q];
                    }
                }
            }
        }
        // println!("Surface area = {:?}", srf_area);
        ptl
        })
        .collect();

    DVector::from_vec(slp_vals)
}

///Interpolates all quantities over the surface of the triangle.
pub fn lsdlpp_3d_interp(p1 :Vector3<f64>,
                        p2 :Vector3<f64>,
                        p3 :Vector3<f64>,
                        p4 :Vector3<f64>,
                        p5 :Vector3<f64>,
                        p6 :Vector3<f64>,
                        v1 :Vector3<f64>,
                        v2 :Vector3<f64>,
                        v3 :Vector3<f64>,
                        v4 :Vector3<f64>,
                        v5 :Vector3<f64>,
                        v6 :Vector3<f64>,
                        f1 :f64, f2 :f64, f3 :f64, f4 :f64, f5 :f64, f6 :f64,
                        df1 :f64, df2 :f64, df3 :f64, df4 :f64, df5 :f64, df6 :f64,
                        al :f64, be :f64, ga :f64,
                        xi :f64, eta :f64) ->
                        (Vector3<f64>, Vector3<f64>, f64, f64, f64) {

    let (alc, bec, gac) = (1.0-al, 1.0-be, 1.0-ga);
    let (alalc, bebec, gagac) = (al * alc, be * bec, ga * gac);

    //Evaluate Basis functions
    let ph2 = xi * (xi - al + eta * (al- ga)/gac)/alc;
    let ph3 = eta * (eta - be + xi * (be + ga - 1.0)/ga)/bec;
    let ph4 = xi * (1.0 - xi - eta)/alalc;
    let ph5 = xi * eta / gagac;
    let ph6 = eta * (1.0 - xi -eta) / bebec;
    let ph1 = 1.0 - ph2 - ph3 - ph4 - ph5 - ph6;

    //Interpolate position vector
    let x = p1[0]*ph1 + p2[0]*ph2 + p3[0]*ph3 + p4[0]*ph4 + p5[0]*ph5 + p6[0]*ph6;
    let y = p1[1]*ph1 + p2[1]*ph2 + p3[1]*ph3 + p4[1]*ph4 + p5[1]*ph5 + p6[1]*ph6;
    let z = p1[2]*ph1 + p2[2]*ph2 + p3[2]*ph3 + p4[2]*ph4 + p5[2]*ph5 + p6[2]*ph6;
    let xvec = Vector3::new(x, y, z);

    //Interpolate the densities f and df
    let f = f1*ph1 + f2 * ph2 + f3 * ph3 + f4 * ph4 + f5 * ph5 + f6 * ph6;
    let df = df1 * ph1 + df2 * ph2 + df3 * ph3 + df4 * ph4 + df5 * ph5 + df6 * ph6;



    //Interpolate the normal vectors (vx, vy, vz)
    let vx = v1[0]*ph1 + v2[0]*ph2 + v3[0]*ph3 + v4[0]*ph4 + v5[0]*ph5 + v6[0]*ph6;
    let vy = v1[1]*ph1 + v2[1]*ph2 + v3[1]*ph3 + v4[1]*ph4 + v5[1]*ph5 + v6[1]*ph6;
    let vz = v1[2]*ph1 + v2[2]*ph2 + v3[2]*ph3 + v4[2]*ph4 + v5[2]*ph5 + v6[2]*ph6;
    let v = Vector3::new(vx, vy, vz);

    //Evaluate xi derivatives of basis functions
    let dph2 = (2.0 * xi - al + eta * (al - ga)/gac) / alc;
    let dph3 = eta * (be + ga - 1.0) / (ga * bec);
    let dph4 = (1.0 - 2.0 * xi - eta) / alalc;
    let dph5 = eta / gagac;
    let dph6 = -eta / bebec;
    let dph1 = -dph2 - dph3 - dph4 - dph5 - dph6;

    //Compute dx/dxi from xi derivatives of phi
    let dx_dxi = p1[0]*dph1 + p2[0]*dph2 + p3[0]*dph3 + p4[0]*dph4 + p5[0]*dph5 + p6[0]*dph6;
    let dy_dxi = p1[1]*dph1 + p2[1]*dph2 + p3[1]*dph3 + p4[1]*dph4 + p5[1]*dph5 + p6[1]*dph6;
    let dz_dxi = p1[2]*dph1 + p2[2]*dph2 + p3[2]*dph3 + p4[2]*dph4 + p5[2]*dph5 + p6[2]*dph6;
    let ddxi = Vector3::new(dx_dxi, dy_dxi, dz_dxi);

    //Evaluate eta derivatives of basis functions
    let pph2 = xi * (al - ga) / (alc * gac);
    let pph3 = (2.0 * eta - be + xi *(be + ga - 1.0) / ga) / bec;
    let pph4 = -xi / alalc;
    let pph5 = xi / gagac;
    let pph6 = (1.0 - xi - 2.0 * eta) / bebec;
    let pph1 = -pph2 - pph3 - pph4 - pph5 - pph6;

    //Compute Dx/Deta from eta derivatives of phi
    let dx_det = p1[0]*pph1 + p2[0]*pph2 + p3[0]*pph3 + p4[0]*pph4 + p5[0]*pph5 + p6[0]*pph6;
    let dy_det = p1[1]*pph1 + p2[1]*pph2 + p3[1]*pph3 + p4[1]*pph4 + p5[1]*pph5 + p6[1]*pph6;
    let dz_det = p1[2]*pph1 + p2[2]*pph2 + p3[2]*pph3 + p4[2]*pph4 + p5[2]*pph5 + p6[2]*pph6;
    let ddet = Vector3::new(dx_det, dy_det, dz_det);

    let mut vn = ddxi.cross(&ddet);
    let hs = vn.norm();

    // vn = vn.normalize();

    (xvec, v, hs, f, df)
}

pub fn gradient_interp(p1 :Vector3<f64>,
                        p2 :Vector3<f64>,
                        p3 :Vector3<f64>,
                        p4 :Vector3<f64>,
                        p5 :Vector3<f64>,
                        p6 :Vector3<f64>,
                        v1 :Vector3<f64>,
                        v2 :Vector3<f64>,
                        v3 :Vector3<f64>,
                        v4 :Vector3<f64>,
                        v5 :Vector3<f64>,
                        v6 :Vector3<f64>,
                        f1 :f64, f2 :f64, f3 :f64, f4 :f64, f5 :f64, f6 :f64,
                        df1 :f64, df2 :f64, df3 :f64, df4 :f64, df5 :f64, df6 :f64,
                        al :f64, be :f64, ga :f64,
                        xi :f64, eta :f64) ->
                        (Vector3<f64>, Vector3<f64>, f64, f64, f64, f64, f64, Vector3<f64>, Vector3<f64>) {

    let (alc, bec, gac) = (1.0-al, 1.0-be, 1.0-ga);
    let (alalc, bebec, gagac) = (al * alc, be * bec, ga * gac);

    //Evaluate Basis functions
    let ph2 = xi * (xi - al + eta * (al- ga)/gac)/alc;
    let ph3 = eta * (eta - be + xi * (be + ga - 1.0)/ga)/bec;
    let ph4 = xi * (1.0 - xi - eta)/alalc;
    let ph5 = xi * eta / gagac;
    let ph6 = eta * (1.0 - xi -eta) / bebec;
    let ph1 = 1.0 - ph2 - ph3 - ph4 - ph5 - ph6;

    //Interpolate position vector
    let x = p1[0]*ph1 + p2[0]*ph2 + p3[0]*ph3 + p4[0]*ph4 + p5[0]*ph5 + p6[0]*ph6;
    let y = p1[1]*ph1 + p2[1]*ph2 + p3[1]*ph3 + p4[1]*ph4 + p5[1]*ph5 + p6[1]*ph6;
    let z = p1[2]*ph1 + p2[2]*ph2 + p3[2]*ph3 + p4[2]*ph4 + p5[2]*ph5 + p6[2]*ph6;
    let xvec = Vector3::new(x, y, z);

    //Interpolate the densities f and df
    let f = f1*ph1 + f2 * ph2 + f3 * ph3 + f4 * ph4 + f5 * ph5 + f6 * ph6;
    let df = df1 * ph1 + df2 * ph2 + df3 * ph3 + df4 * ph4 + df5 * ph5 + df6 * ph6;



    //Interpolate the normal vectors (vx, vy, vz)
    let vx = v1[0]*ph1 + v2[0]*ph2 + v3[0]*ph3 + v4[0]*ph4 + v5[0]*ph5 + v6[0]*ph6;
    let vy = v1[1]*ph1 + v2[1]*ph2 + v3[1]*ph3 + v4[1]*ph4 + v5[1]*ph5 + v6[1]*ph6;
    let vz = v1[2]*ph1 + v2[2]*ph2 + v3[2]*ph3 + v4[2]*ph4 + v5[2]*ph5 + v6[2]*ph6;
    let v = Vector3::new(vx, vy, vz);

    //Evaluate xi derivatives of basis functions
    let dph2 = (2.0 * xi - al + eta * (al - ga)/gac) / alc;
    let dph3 = eta * (be + ga - 1.0) / (ga * bec);
    let dph4 = (1.0 - 2.0 * xi - eta) / alalc;
    let dph5 = eta / gagac;
    let dph6 = -eta / bebec;
    let dph1 = -dph2 - dph3 - dph4 - dph5 - dph6;

    //Compute dx/dxi from xi derivatives of phi
    let dx_dxi = p1[0]*dph1 + p2[0]*dph2 + p3[0]*dph3 + p4[0]*dph4 + p5[0]*dph5 + p6[0]*dph6;
    let dy_dxi = p1[1]*dph1 + p2[1]*dph2 + p3[1]*dph3 + p4[1]*dph4 + p5[1]*dph5 + p6[1]*dph6;
    let dz_dxi = p1[2]*dph1 + p2[2]*dph2 + p3[2]*dph3 + p4[2]*dph4 + p5[2]*dph5 + p6[2]*dph6;
    let ddxi = Vector3::new(dx_dxi, dy_dxi, dz_dxi);
    //Compute df/dxi
    let dfdxi = f1 * dph1 + f2 * dph2 + f3 * dph3 + f4 * dph4 + f5 * dph5 + f6 * dph6;


    //Evaluate eta derivatives of basis functions
    let pph2 = xi * (al - ga) / (alc * gac);
    let pph3 = (2.0 * eta - be + xi *(be + ga - 1.0) / ga) / bec;
    let pph4 = -xi / alalc;
    let pph5 = xi / gagac;
    let pph6 = (1.0 - xi - 2.0 * eta) / bebec;
    let pph1 = -pph2 - pph3 - pph4 - pph5 - pph6;

    //Compute Dx/Deta from eta derivatives of phi
    let dx_det = p1[0]*pph1 + p2[0]*pph2 + p3[0]*pph3 + p4[0]*pph4 + p5[0]*pph5 + p6[0]*pph6;
    let dy_det = p1[1]*pph1 + p2[1]*pph2 + p3[1]*pph3 + p4[1]*pph4 + p5[1]*pph5 + p6[1]*pph6;
    let dz_det = p1[2]*pph1 + p2[2]*pph2 + p3[2]*pph3 + p4[2]*pph4 + p5[2]*pph5 + p6[2]*pph6;
    let ddet = Vector3::new(dx_det, dy_det, dz_det);
    //Compute df/deta
    let dfdet = f1*pph1 + f2 * pph2 + f3 * pph3 + f4 * pph4 + f5 * pph5 + f6 * pph6;


    let mut vn = ddxi.cross(&ddet);
    let hs = vn.norm();

    // vn = vn.normalize();

    (xvec, v, hs, f, df, dfdxi, dfdet, ddxi, ddet)
}

fn line_interp_vector(p1 :Vector3<f64>, p2 :Vector3<f64>,
                      xi :f64) -> Vector3<f64> {
    let p = (1.0 - xi) * p1 + xi * p2;
    p
}

fn line_interp_scalar(f1 :f64, f2 :f64, xi :f64) -> f64 {
    let f = (1.0 - xi) * f1 + xi * f2;
    f
}

pub fn line_interp(p1 :Vector3<f64>,
                   p2 :Vector3<f64>,
                   p3 :Vector3<f64>,
                   f1 :f64, f2 :f64, f3 :f64,
                   df1 :f64, df2 :f64, df3 :f64,
                   xi :f64) -> (Vector3<f64>, f64, f64) {

    //Interpolate position vector

    let mut p = Vector3::zeros();
    let mut f = 0.0;
    let mut df = 0.0;
    if xi < 0.5 {
        p = line_interp_vector(p1, p2, xi * 2.0);
        f = line_interp_scalar(f1, f2, xi * 2.0);
        df = line_interp_scalar(df1, df2, xi * 2.0);
    } else {
        p = line_interp_vector(p2, p3, (xi - 0.5) * 2.0);
        f = line_interp_scalar(f2, f3, (xi - 0.5) * 2.0);
        df = line_interp_scalar(df2, df3, (xi - 0.5) * 2.0);
    }

    (p, f, df)
}

pub fn sine_find(a :&Vector3<f64>, b :&Vector3<f64>) -> f64 {
    (a.cross(b)).norm()/ (a.norm() * b.norm())
}

pub fn gradient_interp_3d_integral(k :usize,
                          mint :usize, f :&DVector<f64>, df :&DVector<f64>,
                          p :&DMatrix<f64>, n :&DMatrix<usize>, vna :&DMatrix<f64>,
                          alpha :&DVector<f64>, beta :&DVector<f64>, gamma :&DVector<f64>,
                          xiq :&DVector<f64>, etq :&DVector<f64>, wq :&DVector<f64>) -> (f64, f64) {

    let mut area = 0.0;
    let mut sdlp = 0.0;

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

    for i in 0..mint {

        let (xi, eta) = (xiq[i], etq[i]);

        let (_xvec, _v, hs, _f_int, dfdn_int, dfdxi, dfdet, ddxi, ddet) = gradient_interp(p1, p2, p3, p4, p5, p6,
                                                              vna1, vna2, vna3, vna4, vna5, vna6,
                                                              f1, f2, f3, f4, f5, f6,
                                                              df1, df2, df3, df4, df5, df6,
                                                              al, be, ga, xi, eta);

        // |∇_s φ|² = |dfdxi·ddet - dfdet·ddxi|² / hs² (surface metric correction)
        let tangential = dfdxi * ddet - dfdet * ddxi;
        let r_int = tangential.norm_squared() / (hs * hs) + dfdn_int * dfdn_int;

        let cf = 0.5 * hs * wq[i];

        area += cf;

        sdlp += r_int* cf;
    }

    (sdlp, area)
}

pub fn pressure_force_element(
    k: usize, mint: usize,
    body_centre: &Vector3<f64>,
    rho_f: f64,
    f: &DVector<f64>, dfdn: &DVector<f64>,
    p: &DMatrix<f64>, n: &DMatrix<usize>, vna: &DMatrix<f64>,
    alpha: &DVector<f64>, beta: &DVector<f64>, gamma: &DVector<f64>,
    xiq: &DVector<f64>, etq: &DVector<f64>, wq: &DVector<f64>,
) -> (Vector3<f64>, Vector3<f64>) {
    let mut force  = Vector3::zeros();
    let mut torque = Vector3::zeros();

    let i1 = n[(k,0)]; let i2 = n[(k,1)]; let i3 = n[(k,2)];
    let i4 = n[(k,3)]; let i5 = n[(k,4)]; let i6 = n[(k,5)];

    let p1 = Vector3::new(p[(i1,0)], p[(i1,1)], p[(i1,2)]);
    let p2 = Vector3::new(p[(i2,0)], p[(i2,1)], p[(i2,2)]);
    let p3 = Vector3::new(p[(i3,0)], p[(i3,1)], p[(i3,2)]);
    let p4 = Vector3::new(p[(i4,0)], p[(i4,1)], p[(i4,2)]);
    let p5 = Vector3::new(p[(i5,0)], p[(i5,1)], p[(i5,2)]);
    let p6 = Vector3::new(p[(i6,0)], p[(i6,1)], p[(i6,2)]);

    let vna1 = Vector3::new(vna[(i1,0)], vna[(i1,1)], vna[(i1,2)]);
    let vna2 = Vector3::new(vna[(i2,0)], vna[(i2,1)], vna[(i2,2)]);
    let vna3 = Vector3::new(vna[(i3,0)], vna[(i3,1)], vna[(i3,2)]);
    let vna4 = Vector3::new(vna[(i4,0)], vna[(i4,1)], vna[(i4,2)]);
    let vna5 = Vector3::new(vna[(i5,0)], vna[(i5,1)], vna[(i5,2)]);
    let vna6 = Vector3::new(vna[(i6,0)], vna[(i6,1)], vna[(i6,2)]);

    let (f1,f2,f3,f4,f5,f6) = (f[i1], f[i2], f[i3], f[i4], f[i5], f[i6]);
    let (df1,df2,df3,df4,df5,df6) = (dfdn[i1], dfdn[i2], dfdn[i3], dfdn[i4], dfdn[i5], dfdn[i6]);
    let (al, be, ga) = (alpha[k], beta[k], gamma[k]);

    for i in 0..mint {
        let (_xvec, _v, hs, _f_int, dfdn_int, dfdxi, dfdet, ddxi, ddet) =
            gradient_interp(p1, p2, p3, p4, p5, p6,
                            vna1, vna2, vna3, vna4, vna5, vna6,
                            f1, f2, f3, f4, f5, f6,
                            df1, df2, df3, df4, df5, df6,
                            al, be, ga, xiq[i], etq[i]);

        let vn = ddxi.cross(&ddet);
        let tangential = dfdxi * ddet - dfdet * ddxi;
        let grad_phi_sq = tangential.norm_squared() / (hs * hs) + dfdn_int * dfdn_int;

        // n̂ dA = vn * 0.5 * wq[i]  (n̂ = vn/hs, dA = 0.5*hs*wq[i])
        let n_hat_da = vn * (0.5 * wq[i]);
        let r = _xvec - body_centre;

        let dp = 0.5 * rho_f * grad_phi_sq;
        force  += dp * n_hat_da;
        torque += dp * r.cross(&n_hat_da);
    }

    (force, torque)
}

/// Computes the −ρ ∂φ/∂t surface-pressure force and torque on one element.
/// ∂φ/∂t is approximated as (f_current − f_prev) / dt.
/// Force = +ρ ∮ (∂φ/∂t) n̂ dA  (positive sign from unsteady Bernoulli).
pub fn dphi_dt_force_element(
    k: usize, mint: usize,
    body_centre: &Vector3<f64>,
    rho_f: f64,
    f: &DVector<f64>, f_prev: &DVector<f64>,
    p: &DMatrix<f64>, n: &DMatrix<usize>, vna: &DMatrix<f64>,
    alpha: &DVector<f64>, beta: &DVector<f64>, gamma: &DVector<f64>,
    xiq: &DVector<f64>, etq: &DVector<f64>, wq: &DVector<f64>,
    dt: f64,
) -> (Vector3<f64>, Vector3<f64>) {
    let mut force  = Vector3::zeros();
    let mut torque = Vector3::zeros();

    let i1 = n[(k,0)]; let i2 = n[(k,1)]; let i3 = n[(k,2)];
    let i4 = n[(k,3)]; let i5 = n[(k,4)]; let i6 = n[(k,5)];

    let p1 = Vector3::new(p[(i1,0)], p[(i1,1)], p[(i1,2)]);
    let p2 = Vector3::new(p[(i2,0)], p[(i2,1)], p[(i2,2)]);
    let p3 = Vector3::new(p[(i3,0)], p[(i3,1)], p[(i3,2)]);
    let p4 = Vector3::new(p[(i4,0)], p[(i4,1)], p[(i4,2)]);
    let p5 = Vector3::new(p[(i5,0)], p[(i5,1)], p[(i5,2)]);
    let p6 = Vector3::new(p[(i6,0)], p[(i6,1)], p[(i6,2)]);

    let vna1 = Vector3::new(vna[(i1,0)], vna[(i1,1)], vna[(i1,2)]);
    let vna2 = Vector3::new(vna[(i2,0)], vna[(i2,1)], vna[(i2,2)]);
    let vna3 = Vector3::new(vna[(i3,0)], vna[(i3,1)], vna[(i3,2)]);
    let vna4 = Vector3::new(vna[(i4,0)], vna[(i4,1)], vna[(i4,2)]);
    let vna5 = Vector3::new(vna[(i5,0)], vna[(i5,1)], vna[(i5,2)]);
    let vna6 = Vector3::new(vna[(i6,0)], vna[(i6,1)], vna[(i6,2)]);

    let (f1,f2,f3,f4,f5,f6) = (f[i1], f[i2], f[i3], f[i4], f[i5], f[i6]);
    // Pass f_prev as the "df" parameter; gradient_interp returns dfdn_int = interpolated f_prev
    let (fp1,fp2,fp3,fp4,fp5,fp6) = (f_prev[i1], f_prev[i2], f_prev[i3], f_prev[i4], f_prev[i5], f_prev[i6]);
    let (al, be, ga) = (alpha[k], beta[k], gamma[k]);

    for i in 0..mint {
        let (xvec, _v, _hs, f_int, fp_int, _dfdxi, _dfdet, ddxi, ddet) =
            gradient_interp(p1, p2, p3, p4, p5, p6,
                            vna1, vna2, vna3, vna4, vna5, vna6,
                            f1, f2, f3, f4, f5, f6,
                            fp1, fp2, fp3, fp4, fp5, fp6,
                            al, be, ga, xiq[i], etq[i]);

        let vn = ddxi.cross(&ddet);
        let dphi_dt_int = (f_int - fp_int) / dt;
        let n_hat_da = vn * (0.5 * wq[i]);
        let r = xvec - body_centre;

        force  += rho_f * dphi_dt_int * n_hat_da;
        torque += rho_f * dphi_dt_int * r.cross(&n_hat_da);
    }

    (force, torque)
}

pub fn gradient_interp_outer_3d(_npts :usize, nelm :usize, mint :usize,
                 f :&DVector<f64>, dfdn :&DVector<f64>,
                 p :&DMatrix<f64>, n :&DMatrix<usize>, vna :&DMatrix<f64>,
                 alpha :&DVector<f64>, beta :&DVector<f64>, gamma :&DVector<f64>,
                 xiq :&DVector<f64>, etq :&DVector<f64>, wq :&DVector<f64>,
                 p0 :&Vector3<f64>) -> (f64, f64) {
    //Integrates the surface velocity of the flow over the surface of the ellipsoids
    let mut f0 = 0.0;
    let mut srf_area = 0.0;

    for k in 0..nelm {


        let (sdlp, arelm) = lsdlpp_3d_integral(p0, k, mint,
                                               f, dfdn,
                                               p, n, vna,
                                               alpha, beta, gamma,
                                               xiq, etq, wq);
        srf_area += arelm;
        f0 += sdlp;
    }

    (srf_area, f0)
}


///Integrates single and double- layer potentials for given f and df/dn.
pub fn lsdlpp_3d_integral(x0 :&Vector3<f64>, k :usize,
                           mint :usize, f :&DVector<f64>, df :&DVector<f64>,
                           p :&DMatrix<f64>, n :&DMatrix<usize>, vna :&DMatrix<f64>,
                           alpha :&DVector<f64>, beta :&DVector<f64>, gamma :&DVector<f64>,
                           xiq :&DVector<f64>, etq :&DVector<f64>, wq :&DVector<f64>) -> (f64, f64) {

    let mut area = 0.0;
    let mut sdlp = 0.0;

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

    for i in 0..mint {

        let (xi, eta) = (xiq[i], etq[i]);

        let (xvec, v, hs, f_int, dfdn_int) = lsdlpp_3d_interp(p1, p2, p3, p4, p5, p6,
                                                 vna1, vna2, vna3, vna4, vna5, vna6,
                                                 f1, f2, f3, f4, f5, f6,
                                                 df1, df2, df3, df4, df5, df6,
                                                 al, be, ga, xi, eta);

        let (g, dg) = lgf_3d_fs(&xvec, &x0);

        let cf = 0.5 * hs * wq[i];

        area += cf;

        let r_int= -dfdn_int * g + f_int * v.dot(&dg);

        sdlp += r_int* cf;
    }

    (sdlp, area)
}

///Loops over all points and elements to calculate phi for given phi and d(phi)/dn on the surfaces.
pub fn lsdlpp_3d(_npts :usize, nelm :usize, mint :usize,
                 f :&DVector<f64>, dfdn :&DVector<f64>,
                 p :&DMatrix<f64>, n :&DMatrix<usize>, vna :&DMatrix<f64>,
                 alpha :&DVector<f64>, beta :&DVector<f64>, gamma :&DVector<f64>,
                 xiq :&DVector<f64>, etq :&DVector<f64>, wq :&DVector<f64>,
                 p0 :&Vector3<f64>) -> (f64, f64) {
    //Evaluates the value of the potential at given point p0 given distribution f, dfdn
    let mut f0 = 0.0;
    let mut srf_area = 0.0;

    for k in 0..nelm {


        let (sdlp, arelm) = lsdlpp_3d_integral(p0, k, mint,
                                               f, dfdn,
                                               p, n, vna,
                                               alpha, beta, gamma,
                                               xiq, etq, wq);
        srf_area += arelm;
        f0 += sdlp;
    }

    (srf_area, f0)
}

///Integrates the kinetic energy of the fluid over the surface of an element
pub fn ke_3d_integral(k :usize, mint :usize,
                      f :&DVector<f64>, df :&DVector<f64>,
                      p :&DMatrix<f64>, n :&DMatrix<usize>, vna :&DMatrix<f64>,
                      alpha :&DVector<f64>, beta :&DVector<f64>, gamma :&DVector<f64>,
                      xiq :&DVector<f64>, etq :&DVector<f64>, wq :&DVector<f64>) -> (f64, f64) {

    let mut area = 0.0;
    let mut sdlp = 0.0;

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

    for i in 0..mint {

        let (xi, eta) = (xiq[i], etq[i]);

        let (_xvec, _v, hs, f_int, dfdn_int) = lsdlpp_3d_interp(p1, p2, p3, p4, p5, p6,
                                                             vna1, vna2, vna3, vna4, vna5, vna6,
                                                             f1, f2, f3, f4, f5, f6,
                                                             df1, df2, df3, df4, df5, df6,
                                                             al, be, ga,
                                                             xi, eta);


        let cf = 0.5 * hs * wq[i];

        area += cf;

        let r_int= -dfdn_int * f_int;

        sdlp += r_int* cf;
    }

    (sdlp, area)
}

///Calculates the total kinetic energy of the fluid from a given phi, d(phi)/dn on the surface.(Assumes \rho_f = 1)
pub fn ke_3d(_npts :usize, nelm :usize, mint :usize,
                 f :&DVector<f64>, dfdn :&DVector<f64>,
                 p :&DMatrix<f64>, n :&DMatrix<usize>, vna :&DMatrix<f64>,
                 alpha :&DVector<f64>, beta :&DVector<f64>, gamma :&DVector<f64>,
                 xiq :&DVector<f64>, etq :&DVector<f64>, wq :&DVector<f64>) -> (f64, f64) {

    //Calculates the total kinetic energy of the fluid, doing a surface integral of phi*(dphi/dn)

    let mut srf_area = 0.0;
    let mut f0 = 0.0;

    for k in 0..nelm {


        let (sdlp, arelm) = ke_3d_integral(k, mint,
                                                f, dfdn,
                                                p, n, vna,
                                                alpha, beta, gamma,
                                                xiq, etq, wq);

        srf_area += arelm;
        f0 += sdlp;
    }

    (srf_area, f0)
}

pub fn grad_3d_integral(p0 :&Vector3<f64>,
                      k :usize, mint :usize,
                      f :&DVector<f64>, df :&DVector<f64>,
                      p :&DMatrix<f64>, n :&DMatrix<usize>, vna :&DMatrix<f64>,
                      alpha :&DVector<f64>, beta :&DVector<f64>, gamma :&DVector<f64>,
                      xiq :&DVector<f64>, etq :&DVector<f64>, wq :&DVector<f64>) -> (Vector3<f64>, f64) {

    let mut area = 0.0;
    let mut sdlp = Vector3::zeros();

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

    for i in 0..mint {

        let (xi, eta) = (xiq[i], etq[i]);

        let (xvec, vn, hs, f_int, dfdn_int) = lsdlpp_3d_interp(p1, p2, p3, p4, p5, p6,
                                                               vna1, vna2, vna3, vna4, vna5, vna6,
                                                               f1, f2, f3, f4, f5, f6,
                                                               df1, df2, df3, df4, df5, df6,
                                                               al, be, ga, xi, eta);

        // println!("p0 = {:?}, x = {:?}", p0, xvec);

        let (d_g, dd_g) = d_lgf_3d_fs_full(&xvec, p0);

        // println!("dg = {:?}", d_g);

        let cf = 0.5 * hs * wq[i];

        area += cf;

        let r_int= (-dfdn_int * d_g) + (f_int * dd_g * vn);

        sdlp += cf * r_int;
    }

    (sdlp, area)
}

pub fn grad_3d(nelm :usize, mint :usize,
             f :&DVector<f64>, dfdn :&DVector<f64>,
             p :&DMatrix<f64>, n :&DMatrix<usize>, vna :&DMatrix<f64>,
             alpha :&DVector<f64>, beta :&DVector<f64>, gamma :&DVector<f64>,
             xiq :&DVector<f64>, etq :&DVector<f64>, wq :&DVector<f64>,
             p0 :&Vector3<f64>) -> Vector3<f64> {

    //Calculates the gradient of phi in the given direction dxi at point p0.

    let mut f0 = Vector3::zeros();

    for k in 0..nelm {


        let (sdlp, _arelm) = grad_3d_integral(p0 ,
                                              k, mint,
                                              f, dfdn,
                                              p, n, vna,
                                              alpha, beta, gamma,
                                              xiq, etq, wq);

        f0 += sdlp;
    }
    f0
}

pub fn grad_3d_integral_l1(p0 :&Vector3<f64>,
                        k :usize, mint :usize,
                        f :&DVector<f64>, dfdn :&DVector<f64>,
                        p :&DMatrix<f64>, n :&DMatrix<usize>, vna :&DMatrix<f64>,
                        alpha :&DVector<f64>, beta :&DVector<f64>, gamma :&DVector<f64>,
                        xiq :&DVector<f64>, etq :&DVector<f64>, wq :&DVector<f64>)
    -> (Vector3<f64>, f64) {
    ///Does the integral on line 1 of eq(28). Carried out on outer subsurface.

    let (sdlp, area) = grad_3d_integral(p0,
                                                k, mint,
                                                f, dfdn,
                                                p, n ,vna,
                                                alpha, beta, gamma,
                                                xiq, etq, wq);

    (sdlp, area)

}

pub fn grad_3d_l1(nonsing_elms :&Vec<usize>, mint :usize,
                  f :&DVector<f64>, dfdn :&DVector<f64>,
                  p :&DMatrix<f64>, n :&DMatrix<usize>, vna :&DMatrix<f64>,
                  alpha :&DVector<f64>, beta :&DVector<f64>, gamma :&DVector<f64>,
                  xiq :&DVector<f64>, etq :&DVector<f64>, wq :&DVector<f64>,
                  p0 :&Vector3<f64>) -> Vector3<f64> {

    let mut f0 = Vector3::zeros();

    for &k in nonsing_elms {


        let (sdlp, area) =  grad_3d_integral_l1(p0,
                                         k, mint,
                                         f, dfdn,
                                         p, n ,vna,
                                         alpha, beta, gamma,
                                         xiq, etq, wq);
        // println!("k = {:?}, sdlp = {:?}", k, sdlp);


        f0 += sdlp
    }

    f0



}

pub fn grad_3d_integral_l2(p0 :&Vector3<f64>,
                        k :usize, mint :usize,
                        f :&DVector<f64>, df :&DVector<f64>,
                        p :&DMatrix<f64>, n :&DMatrix<usize>, vna :&DMatrix<f64>,
                        alpha :&DVector<f64>, beta :&DVector<f64>, gamma :&DVector<f64>,
                        xiq :&DVector<f64>, etq :&DVector<f64>, wq :&DVector<f64>,
                        dfdn_xi :f64) -> (Vector3<f64>, f64) {
    ///Does the integral on line 2 of eq(28) of Hyp-singular integrals paper. Carried out on the inner subsurface

    let mut area = 0.0;
    let mut sdlp = Vector3::zeros();

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

    for i in 0..mint {

        let (xi, eta) = (xiq[i], etq[i]);

        let (xvec, _vn, hs, _f_int, dfdn_int) = lsdlpp_3d_interp(p1, p2, p3, p4, p5, p6,
                                                                vna1, vna2, vna3, vna4, vna5, vna6,
                                                                f1, f2, f3, f4, f5, f6,
                                                                df1, df2, df3, df4, df5, df6,
                                                                al, be, ga, xi, eta);


        let (_g, mut dg) = lgf_3d_fs(&xvec, p0);

        if (xvec - p0).norm() < 1e-5 {
            dg = Vector3::zeros();
        }


        let cf = 0.5 * hs * wq[i];

        area += cf;

        let r_int = dg * (dfdn_int - dfdn_xi);

        sdlp += r_int * cf;
    }

    (sdlp, area)
}

pub fn grad_3d_l2(sing_elms :&Vec<usize>, mint :usize,
                  f :&DVector<f64>, dfdn :&DVector<f64>,
                  p :&DMatrix<f64>, n :&DMatrix<usize>, vna :&DMatrix<f64>,
                  alpha :&DVector<f64>, beta :&DVector<f64>, gamma :&DVector<f64>,
                  xiq :&DVector<f64>, etq :&DVector<f64>, wq :&DVector<f64>,
                  p0 :&Vector3<f64>, f_p0 :f64, dfdn_p0:f64) -> Vector3<f64> {

    let mut f0 = Vector3::zeros();

    for &k in sing_elms {

        let i1 = n[(k, 0)];
        let i2 = n[(k, 1)];
        let i3 = n[(k, 2)];
        let i4 = n[(k, 3)];
        let i5 = n[(k, 4)];
        let i6 = n[(k, 5)];

        let (f1, f2, f3, f4, f5, f6) = (f[i1], f[i2], f[i3], f[i4], f[i5], f[i6]);

        let test = Vector6::new(f1, f2, f3, f4, f5, f6).abs().sum();

        if test > 1e-8 {
            let (sdlp, area) =  grad_3d_integral_l2(p0,
                                                    k, mint,
                                                    f, dfdn,
                                                    p, n, vna,
                                                    alpha, beta, gamma,
                                                    xiq, etq, wq, dfdn_p0);
            f0 += sdlp;
        }



    }
    // f0 += - 0.5 * f_p0;

    f0
}

pub fn grad_3d_integral_l3_1(p0 :&Vector3<f64>,
                           k :usize, mint :usize,
                           f :&DVector<f64>, df :&DVector<f64>,
                           p :&DMatrix<f64>, n :&DMatrix<usize>, vna :&DMatrix<f64>,
                           alpha :&DVector<f64>, beta :&DVector<f64>, gamma :&DVector<f64>,
                           xiq :&DVector<f64>, etq :&DVector<f64>, wq :&DVector<f64>,
                           f_xi :f64) -> (Vector3<f64>, f64) {
    ///Does the integral on line 3 of eq(28) of Hyp-singular integrals paper. Carried out on the inner subsurface
    ///(Only the part that does not need to be rearranged) I have included the negative sign outside the integral.

    let mut area = 0.0;
    let mut sdlp = Vector3::zeros();

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

    for i in 0..mint {

        let (xi, eta) = (xiq[i], etq[i]);

        let (xvec, vn, hs, f_int, _dfdn_int) = lsdlpp_3d_interp(p1, p2, p3, p4, p5, p6,
                                                              vna1, vna2, vna3, vna4, vna5, vna6,
                                                              f1, f2, f3, f4, f5, f6,
                                                              df1, df2, df3, df4, df5, df6,
                                                              al, be, ga, xi, eta);


        let (_dg, mut dd_g) = d_lgf_3d_fs_full(&xvec, p0);

        if (xvec - p0).norm() < 1e-3 {
            dd_g = Matrix3::zeros()
        };

        let cf = 0.5 * hs * wq[i];

        area += cf;

        let r_int = Matrix3::from_diagonal_element(-(f_int - f_xi)) * dd_g * vn;

        sdlp += Matrix3::from_diagonal_element(cf) * r_int ;
    }

    (sdlp, area)
}

pub fn grad_3d_l3_1(sing_elms :&Vec<usize>, mint :usize,
                  f :&DVector<f64>, dfdn :&DVector<f64>,
                  p :&DMatrix<f64>, n :&DMatrix<usize>, vna :&DMatrix<f64>,
                  alpha :&DVector<f64>, beta :&DVector<f64>, gamma :&DVector<f64>,
                  xiq :&DVector<f64>, etq :&DVector<f64>, wq :&DVector<f64>,
                  p0 :&Vector3<f64>, f_xi :f64) -> Vector3<f64> {

    let mut f0 = Vector3::zeros();

    for &k in sing_elms {

        let (sdlp, area) =  grad_3d_integral_l3_1(p0,
                                                k, mint,
                                                f, dfdn,
                                                p, n ,vna,
                                                alpha, beta, gamma,
                                                xiq, etq, wq, f_xi);

        f0 += sdlp
    }
    f0
}

pub fn grad_3d_integral_l3_2(p0 :&Vector3<f64>,
                           k :usize, mint :usize,
                           f :&DVector<f64>, df :&DVector<f64>,
                           p :&DMatrix<f64>, n :&DMatrix<usize>, vna :&DMatrix<f64>,
                           alpha :&DVector<f64>, beta :&DVector<f64>, gamma :&DVector<f64>,
                           xiq :&DVector<f64>, etq :&DVector<f64>, wq :&DVector<f64>)
    -> (Matrix3<f64>, f64) {
    ///Does the integral on line 3 of eq(28) of Hyp-singular integrals paper. Carried out on the inner subsurface
    ///(Only the part that does need to be rearranged)
    ///
    let mut area = 0.0;
    let mut sdlp = Matrix3::zeros();

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
    // println!("Integrating on {:?}th element.", k);
    //should be 0..mint
    for i in 0..1 {

        let (xi, eta) = (xiq[i], etq[i]);

        let (xvec, vn, hs, f_int, dfdn_int) = lsdlpp_3d_interp(p1, p2, p3, p4, p5, p6,
                                                               vna1, vna2, vna3, vna4, vna5, vna6,
                                                               f1, f2, f3, f4, f5, f6,
                                                               df1, df2, df3, df4, df5, df6,
                                                               al, be, ga, xi, eta);

        // println!("xvec = {:?}, vn = {:?}", xvec, vn);
        let (dg, mut dd_g) = d_lgf_3d_fs_full(&xvec, p0);
        // println!("xvec - p0  = {:?}", (xvec - p0).norm());
        if (xvec - p0).norm() < 1e-5 {
            dd_g = Matrix3::zeros()
        }
        // println!("vn = {:?}", vn);
        let dd_g_dn = dd_g * vn;
        let cf = 0.5 * hs * wq[i];
        // println!("hs = {:?}", hs);
        // println!("elm area = {:?}", cf);
        area += cf;
        // println!("dd_g_dn = {:?}", dd_g_dn);
        // println!("x-p0 = {:?}", xvec-p0);
        let r_int = dd_g_dn * (xvec - p0).transpose() ;
        let cf_mat = Matrix3::from_diagonal_element(cf);
        sdlp += cf_mat * r_int;
    }
    // println!("Area = {:?}", area);
    (sdlp, area)
}

pub fn grad_3d_l3_2(sing_elms :&Vec<usize>, mint :usize,
                    f :&DVector<f64>, dfdn :&DVector<f64>,
                    p :&DMatrix<f64>, n :&DMatrix<usize>, vna :&DMatrix<f64>,
                    alpha :&DVector<f64>, beta :&DVector<f64>, gamma :&DVector<f64>,
                    xiq :&DVector<f64>, etq :&DVector<f64>, wq :&DVector<f64>,
                    p0 :&Vector3<f64>) -> Matrix3<f64> {

    let mut f0 = Matrix3::zeros();
    let mut area_tot = 0.0;


    for &k in sing_elms {

        let (sdlp, area) =  grad_3d_integral_l3_2(p0,
                                                  k, mint,
                                                  f, dfdn,
                                                  p, n ,vna,
                                                  alpha, beta, gamma,
                                                  xiq, etq, wq);

        f0 += sdlp;
        area_tot += area;
    }
    // println!("Area = {:?}", area_tot);
    f0
}


pub fn grad_3d_integral_l5(p0 :&Vector3<f64>,
                           k :usize, mint :usize,
                           f :&DVector<f64>, df :&DVector<f64>,
                           p :&DMatrix<f64>, n :&DMatrix<usize>, vna :&DMatrix<f64>,
                           alpha :&DVector<f64>, beta :&DVector<f64>, gamma :&DVector<f64>,
                           xiq :&DVector<f64>, etq :&DVector<f64>, wq :&DVector<f64>,
                           n_xi :&Vector3<f64>) -> (Matrix3<f64>, f64) {
    ///Does the integral on line 5 of eq(28) of Hyp-singular integrals paper. Carried out on the inner subsurface

    let mut area = 0.0;
    let mut sdlp = Matrix3::zeros();

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

    for i in 0..mint {

        let (xi, eta) = (xiq[i], etq[i]);

        let (xvec, vn, hs, f_int, dfdn_int) = lsdlpp_3d_interp(p1, p2, p3, p4, p5, p6,
                                                              vna1, vna2, vna3, vna4, vna5, vna6,
                                                              f1, f2, f3, f4, f5, f6,
                                                              df1, df2, df3, df4, df5, df6,
                                                              al, be, ga, xi, eta);


        let (g, mut d_g) = lgf_3d_fs(&xvec, p0);

        if (xvec - p0).norm() < 1e-3 {
            d_g = Vector3::zeros()
        }

        let cf = 0.5 * hs * wq[i];

        area += cf;

        let r_int = d_g * (vn - n_xi).transpose();


        sdlp += Matrix3::from_diagonal_element(cf) * r_int;
    }

    (sdlp, area)
}

pub fn grad_3d_l5(sing_elms :&Vec<usize>, mint :usize,
                    f :&DVector<f64>, dfdn :&DVector<f64>,
                    p :&DMatrix<f64>, n :&DMatrix<usize>, vna :&DMatrix<f64>,
                    alpha :&DVector<f64>, beta :&DVector<f64>, gamma :&DVector<f64>,
                    xiq :&DVector<f64>, etq :&DVector<f64>, wq :&DVector<f64>,
                    p0 :&Vector3<f64>, p0_n :&Vector3<f64>) -> Matrix3<f64> {

    let mut f0 = Matrix3::zeros();

    for &k in sing_elms {

        let (sdlp, area) =  grad_3d_integral_l5(p0,
                                                  k, mint,
                                                  f, dfdn,
                                                  p, n ,vna,
                                                  alpha, beta, gamma,
                                                  xiq, etq, wq, p0_n);

        f0 += sdlp
    }
    f0
}

fn grad_3d_integral_l4_1(p0 :&Vector3<f64>,
                         k :usize,
                         p :&DMatrix<f64>, n_line :&DMatrix<usize>) -> (Vector3<f64>, f64) {
    ///Does the first line integral on line 4.


    let i1 = n_line[(k, 0)];
    let i2 = n_line[(k, 1)];
    let i3 = n_line[(k, 2)];

    let p1 = Vector3::new(p[(i1, 0)], p[(i1, 1)], p[(i1, 2)]);
    let p2 = Vector3::new(p[(i2, 0)], p[(i2, 1)], p[(i2, 2)]);
    let p3 = Vector3::new(p[(i3, 0)], p[(i3, 1)], p[(i3, 2)]);


    let p12 = line_interp_vector(p1, p2, 0.5);
    let p23 = line_interp_vector(p2, p3, 0.5);


    let h1 = (p1 - p2);
    let h2 = (p2 - p3);

    let (g1, dg1) = lgf_3d_fs(&p12, p0);
    let (g2, dg2) = lgf_3d_fs(&p23, p0);

    let r1 = g1 * h1;
    let r2 = g2 * h2;

    let integral = r1 + r2;
    let length = h1.norm() + h2.norm();

    (integral, length)
}

pub fn grad_3d_l4_1(    p :&DMatrix<f64>, n_line :&DMatrix<usize>,
                        p0 :&Vector3<f64>) -> Vector3<f64> {

    let mut f0 = Vector3::zeros();

    for k in 0..n_line.shape().0 {

        let (sdlp, length) = grad_3d_integral_l4_1(p0,
                                                   k,
                                                   p, n_line);

        f0 += sdlp;
    }
    f0
}

fn grad_3d_integral_l4_2(p0 :&Vector3<f64>,
                         k :usize,
                         p :&DMatrix<f64>, n_line :&DMatrix<usize>)
    -> (Matrix3<Vector3<f64>>, f64) {
    ///Does the second line integral on line 4. Returns a 3x3x3 Tensor.


    let i1 = n_line[(k, 0)];
    let i2 = n_line[(k, 1)];
    let i3 = n_line[(k, 2)];

    let p1 = Vector3::new(p[(i1, 0)], p[(i1, 1)], p[(i1, 2)]);
    let p2 = Vector3::new(p[(i2, 0)], p[(i2, 1)], p[(i2, 2)]);
    let p3 = Vector3::new(p[(i3, 0)], p[(i3, 1)], p[(i3, 2)]);


    let p12 = line_interp_vector(p1, p2, 0.5);
    let p23 = line_interp_vector(p2, p3, 0.5);

    let h1 = (p1 - p2);
    let h2 = (p2 - p3);

    let (g1, dg1) = lgf_3d_fs(&p12, p0);
    let (g2, dg2) = lgf_3d_fs(&p23, p0);

    let r1 = p12 - p0;

    let mut outer_product1: Matrix3<Vector3<f64>> = Matrix3::zeros();
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                outer_product1[(i,j)][k] = h1[i] * dg1[j] * r1[k];
            }
        }
    };


    let r2 = p23 - p0;

    let mut outer_product2: Matrix3<Vector3<f64>> = Matrix3::zeros();
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                outer_product2[(i,j)][k] = h2[i] * dg2[j] * r2[k];
            }
        }
    };
    let integral = outer_product1 + outer_product2;
    let length = h1.norm() + h2.norm();

    (integral, length)
}

pub fn grad_3d_l4_2(p :&DMatrix<f64>, n_line :&DMatrix<usize>,
                    p0 :&Vector3<f64>) -> Matrix3<Vector3<f64>> {

    let mut f0 :Matrix3<Vector3<f64>> = Matrix3::zeros();


    for k in 0..n_line.shape().0 {

        let (sdlp, length) = grad_3d_integral_l4_2(p0,
                                                   k,
                                                   p, n_line);

        f0 += sdlp;
    }
    f0
}

pub fn grad_3d_integral_l6_1(p0 :&Vector3<f64>,
                             k :usize,
                             p :&DMatrix<f64>, n_line :&DMatrix<usize>) -> Vector3<f64> {

    let i1 = n_line[(k, 0)];
    let i2 = n_line[(k, 1)];
    let i3 = n_line[(k, 2)];

    let p1 = Vector3::new(p[(i1, 0)], p[(i1, 1)], p[(i1, 2)]);
    let p2 = Vector3::new(p[(i2, 0)], p[(i2, 1)], p[(i2, 2)]);
    let p3 = Vector3::new(p[(i3, 0)], p[(i3, 1)], p[(i3, 2)]);


    let p12 = line_interp_vector(p1, p2, 0.5);
    let p23 = line_interp_vector(p2, p3, 0.5);




    let h1 = (p1 - p2);
    let h2 = (p2 - p3);

    let (g1, dg1) = lgf_3d_fs(&p12, p0);
    let (g2, dg2) = lgf_3d_fs(&p23, p0);


    let integ1 = h1.cross(&dg1);
    let integ2 = h2.cross(&dg2);

    let integ = integ1 + integ2;

    integ
}

pub fn midpoint_gen(p1: &Vector3<f64>, p2: &Vector3<f64>, f1 :f64 ,f2:f64) -> (Vector3<f64>, f64) {

    let p_12 = (p1 + p2)/2.0;
    let f_12 = (f1 + f2)/2.0;

    (p_12, f_12)
}

pub fn grad_3d_l6_1(p :&DMatrix<f64>, n_line :&DMatrix<usize>,
                    p0 :&Vector3<f64>) -> Vector3<f64> {

    let mut f0 :Vector3<f64> = Vector3::zeros();


    for k in 0..n_line.shape().0 {

        let sdlp = grad_3d_integral_l6_1(p0,
                                                   k,
                                                   p, n_line);

        f0 += sdlp;
    }
    f0
}

pub fn grad_3d_all_rhs(sing_elms :&Vec<usize>, nonsing_elms :&Vec<usize>, mint :usize,
                       f :&DVector<f64>, dfdn :&DVector<f64>,
                       p :&DMatrix<f64>, n :&DMatrix<usize>, n_line :&DMatrix<usize>,  vna :&DMatrix<f64>,
                       alpha :&DVector<f64>, beta :&DVector<f64>, gamma :&DVector<f64>,
                       xiq :&DVector<f64>, etq :&DVector<f64>, wq :&DVector<f64>,
                       p0 :&Vector3<f64>,  f_p0 :f64, dfdn_p0 :f64) -> Vector3<f64> {

    let l1 = grad_3d_l1(nonsing_elms, mint,
                        f, dfdn,
                        p, n, vna,
                        alpha, beta, gamma,
                        xiq, etq, wq, p0);

    let l2 = grad_3d_l2(sing_elms, mint,
                        f, dfdn,
                        p, n, vna,
                        alpha, beta, gamma,
                        xiq, etq, wq,
                        p0, f_p0, dfdn_p0);

    let l3_1 = grad_3d_l3_1(sing_elms, mint,
                            f, dfdn,
                            p, n, vna,
                            alpha, beta, gamma,
                            xiq, etq, wq,
                            p0, f_p0);

    let l6_1 = grad_3d_l6_1(p, n_line,
                            p0) * f_p0;

    // println!("l1 = {:?}",l1);//    , l2 = {:?}, l3_1 = {:?}, l6_1 = {:?}", l1, l2, l3_1, l6_1);

    let rhs = l1 + l2 + l3_1 + l6_1;

    // println!("Rhs = {:?}",rhs);

    rhs
}

fn skew_symmetric_from_vec(v :&Vector3<f64>) -> Matrix3<f64> {
    let (v0, v1, v2) = (v[0], v[1], v[2]);

    let matrix = Matrix3::new(0.0, -v2, v1,
                              v2, 0.0, -v0,
                              -v1, v0, 0.0);

    matrix

}
pub fn grad_3d_all_lhs(sing_elms :&Vec<usize>, nonsing_elms :&Vec<usize>, mint :usize,
                       f :&DVector<f64>, dfdn :&DVector<f64>,
                       p :&DMatrix<f64>, n :&DMatrix<usize>, n_line :&DMatrix<usize>,  vna :&DMatrix<f64>,
                       alpha :&DVector<f64>, beta :&DVector<f64>, gamma :&DVector<f64>,
                       xiq :&DVector<f64>, etq :&DVector<f64>, wq :&DVector<f64>,
                       p0 :&Vector3<f64>, p0_n :&Vector3<f64>, f_p0 :f64, dfdn_p0 :f64) -> Matrix3<f64> {

    let l3_2 = grad_3d_l3_2(sing_elms, mint,
                            f, dfdn,
                            p, n, vna,
                            alpha, beta, gamma,
                            xiq, etq, wq,
                            p0);

    let l4_1_vec = -grad_3d_l4_1(p, n_line,
                                p0);
    //the full term on the rhs is u cross l4_1_vec, so -l4_1_vec cross u. (hence the -grad...())

    let mut l4_1 = skew_symmetric_from_vec(&l4_1_vec);

    let l4_2_tensor = grad_3d_l4_2(p, n_line,
                                   p0);

    let row0 = l4_2_tensor[(1,2)] - l4_2_tensor[(2,1)];
    let row1 = l4_2_tensor[(2,0)] - l4_2_tensor[(0,2)];
    let row2 = l4_2_tensor[(0,1)] - l4_2_tensor[(1,0)];

    let l4_2 = Matrix3::new(row0[0], row0[1], row0[2],
                            row1[0], row1[1], row1[2],
                            row2[0], row2[1], row2[2]);


    let l5 = grad_3d_l5( sing_elms, mint,
                        f, dfdn,
                        p, n, vna,
                        alpha, beta, gamma,
                        xiq, etq, wq,
                        p0, p0_n);

    let l6_2_scalar = 0.5; //Solid angle/4pi should be 0.5 in the limit as a point approaches the surface.

    let l6_2 = Matrix3::from_diagonal_element(l6_2_scalar);

    //When rearranging eq(28), these terms are all subtracted from I and multiplied by u to give the rhs.

    let id_mat = Matrix3::from_diagonal_element(1.0);
    //l3_2, l4_2, l5 are the assymetric ones
    // println!("l3_2 = {:?}", l3_2);

    let mat = id_mat - l3_2 - l4_1 - l4_2 - l5 - l6_2;

    mat

}

pub fn dphi(p1 :&Vector3<f64>, p2:&Vector3<f64>, f1:f64, f2:f64) -> Vector3<f64> {

    let dx = p2 - p1;
    let df = f2 - f1;

    // let mag_dx = dx.norm();
    //
    // let dphi = df/mag_dx;
    //
    // (dphi,dx.normalize())
    let mut dphi = Vector3::new(0.0,0.0,0.0);
    for i in 0..3 {
        dphi[i] = df/dx[i]
    }
    dphi
}
