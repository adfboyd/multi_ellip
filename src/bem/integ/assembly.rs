//! Green's-function kernels, element interpolation/quadrature, and BEM matrix/RHS assembly.
#![allow(
    unused_doc_comments,
    unused_variables,
    unused_assignments,
    unused_mut,
    unused_parens,
    unused_imports
)]
use super::*;
use nalgebra::{DMatrix, DVector, Matrix3, Vector3, Vector6};
use rayon::prelude::*;
use std::f64::consts::PI;

pub fn lgf_3d_fs(x: &Vector3<f64>, x0: &Vector3<f64>) -> (f64, Vector3<f64>) {
    let pi = PI;
    let pi4 = pi * 4.0;

    let dx = x - x0;
    let r = dx.norm();
    let g = 1.0 / (pi4 * r);

    let mut dg = Vector3::zeros();
    let den = pi4 * r * r * r;
    for i in 0..3 {
        dg[i] = -dx[i] / den;
    }

    (g, dg)
}

pub fn d_lgf_3d_fs(x: &Vector3<f64>, x0: &Vector3<f64>, xi: &Vector3<f64>) -> (f64, Vector3<f64>) {
    let (_, dg) = lgf_3d_fs(x, x0);

    let ri = x - x0;
    let r = ri.norm();
    let dgdx = dg.dot(&xi);

    let mut dd_g = Matrix3::zeros();

    for i in 0..3 {
        for j in 0..3 {
            let mut delta = 0f64;
            if i == j {
                delta = 1f64;
            }
            dd_g[(i, j)] = delta / r.powi(3) - (3.0 / (4.0 * PI * r.powi(5))) * ri[i] * ri[j];
        }
    }

    let dd_gdx2 = dd_g * xi;
    (dgdx, dd_gdx2)
}

pub fn d_lgf_3d_fs_full(x: &Vector3<f64>, x0: &Vector3<f64>) -> (Vector3<f64>, Matrix3<f64>) {
    let ri = x - x0;
    let r = ri.norm();
    let den = PI * 4.0 * r * r * r;

    let mut dg = Vector3::zeros();
    for i in 0..3 {
        dg[i] = -ri[i] / den;
    }

    let mut dd_g = Matrix3::zeros();

    for i in 0..3 {
        for j in 0..3 {
            let mut delta = 0f64;
            if i == j {
                delta = 1f64;
            }
            dd_g[(i, j)] =
                (-1.0 / (4.0 * PI)) * (delta / r.powi(3) - (3.0 / (r.powi(5))) * ri[i] * ri[j]);
        }
    }

    (dg, dd_g)
}

///Interpolates over the triangle, also interpolates the force f.
pub fn lslp_3d_interp(
    p1: Vector3<f64>,
    p2: Vector3<f64>,
    p3: Vector3<f64>,
    p4: Vector3<f64>,
    p5: Vector3<f64>,
    p6: Vector3<f64>,
    f1: f64,
    f2: f64,
    f3: f64,
    f4: f64,
    f5: f64,
    f6: f64,
    al: f64,
    be: f64,
    ga: f64,
    xi: f64,
    eta: f64,
) -> (Vector3<f64>, f64, f64) {
    let (alc, bec, gac) = (1.0 - al, 1.0 - be, 1.0 - ga);
    let (alalc, bebec, gagac) = (al * alc, be * bec, ga * gac);

    //Evaluate Basis functions
    let ph2 = xi * (xi - al + eta * (al - ga) / gac) / alc;
    let ph3 = eta * (eta - be + xi * (be + ga - 1.0) / ga) / bec;
    let ph4 = xi * (1.0 - xi - eta) / alalc;
    let ph5 = xi * eta / gagac;
    let ph6 = eta * (1.0 - xi - eta) / bebec;
    let ph1 = 1.0 - ph2 - ph3 - ph4 - ph5 - ph6;

    //Interpolate position vector
    let x = p1[0] * ph1 + p2[0] * ph2 + p3[0] * ph3 + p4[0] * ph4 + p5[0] * ph5 + p6[0] * ph6;
    let y = p1[1] * ph1 + p2[1] * ph2 + p3[1] * ph3 + p4[1] * ph4 + p5[1] * ph5 + p6[1] * ph6;
    let z = p1[2] * ph1 + p2[2] * ph2 + p3[2] * ph3 + p4[2] * ph4 + p5[2] * ph5 + p6[2] * ph6;
    let xvec = Vector3::new(x, y, z);

    //Interpolate the density f
    let f = f1 * ph1 + f2 * ph2 + f3 * ph3 + f4 * ph4 + f5 * ph5 + f6 * ph6;

    //Evaluate xi derivatives of basis functions
    let dph2 = (2.0 * xi - al + eta * (al - ga) / gac) / alc;
    let dph3 = eta * (be + ga - 1.0) / (ga * bec);
    let dph4 = (1.0 - 2.0 * xi - eta) / alalc;
    let dph5 = eta / gagac;
    let dph6 = -eta / bebec;
    let dph1 = -dph2 - dph3 - dph4 - dph5 - dph6;

    //Compute dx/dxi from xi derivatives of phi
    let dx_dxi =
        p1[0] * dph1 + p2[0] * dph2 + p3[0] * dph3 + p4[0] * dph4 + p5[0] * dph5 + p6[0] * dph6;
    let dy_dxi =
        p1[1] * dph1 + p2[1] * dph2 + p3[1] * dph3 + p4[1] * dph4 + p5[1] * dph5 + p6[1] * dph6;
    let dz_dxi =
        p1[2] * dph1 + p2[2] * dph2 + p3[2] * dph3 + p4[2] * dph4 + p5[2] * dph5 + p6[2] * dph6;
    let ddxi = Vector3::new(dx_dxi, dy_dxi, dz_dxi);

    //Evaluate eta derivatives of basis functions
    let pph2 = xi * (al - ga) / (alc * gac);
    let pph3 = (2.0 * eta - be + xi * (be + ga - 1.0) / ga) / bec;
    let pph4 = -xi / alalc;
    let pph5 = xi / gagac;
    let pph6 = (1.0 - xi - 2.0 * eta) / bebec;
    let pph1 = -pph2 - pph3 - pph4 - pph5 - pph6;

    //Compute Dx/Deta from eta derivatives of phi
    let dx_det =
        p1[0] * pph1 + p2[0] * pph2 + p3[0] * pph3 + p4[0] * pph4 + p5[0] * pph5 + p6[0] * pph6;
    let dy_det =
        p1[1] * pph1 + p2[1] * pph2 + p3[1] * pph3 + p4[1] * pph4 + p5[1] * pph5 + p6[1] * pph6;
    let dz_det =
        p1[2] * pph1 + p2[2] * pph2 + p3[2] * pph3 + p4[2] * pph4 + p5[2] * pph5 + p6[2] * pph6;
    let ddet = Vector3::new(dx_det, dy_det, dz_det);

    let vn = ddxi.cross(&ddet);
    let hs = vn.norm();

    // vn = vn.normalize();

    (xvec, hs, f)
}

///Interpolates over the triangle, including the force q and the normal vector.
pub fn ldlp_3d_interp(
    p1: Vector3<f64>,
    p2: Vector3<f64>,
    p3: Vector3<f64>,
    p4: Vector3<f64>,
    p5: Vector3<f64>,
    p6: Vector3<f64>,
    v1: Vector3<f64>,
    v2: Vector3<f64>,
    v3: Vector3<f64>,
    v4: Vector3<f64>,
    v5: Vector3<f64>,
    v6: Vector3<f64>,
    q1: f64,
    q2: f64,
    q3: f64,
    q4: f64,
    q5: f64,
    q6: f64,
    al: f64,
    be: f64,
    ga: f64,
    xi: f64,
    eta: f64,
) -> (Vector3<f64>, Vector3<f64>, f64, f64) {
    let (alc, bec, gac) = (1.0 - al, 1.0 - be, 1.0 - ga);
    let (alalc, bebec, gagac) = (al * alc, be * bec, ga * gac);

    //Evaluate Basis functions
    let ph2 = xi * (xi - al + eta * (al - ga) / gac) / alc;
    let ph3 = eta * (eta - be + xi * (be + ga - 1.0) / ga) / bec;
    let ph4 = xi * (1.0 - xi - eta) / alalc;
    let ph5 = xi * eta / gagac;
    let ph6 = eta * (1.0 - xi - eta) / bebec;
    let ph1 = 1.0 - ph2 - ph3 - ph4 - ph5 - ph6;

    //Interpolate position vector
    let x = p1[0] * ph1 + p2[0] * ph2 + p3[0] * ph3 + p4[0] * ph4 + p5[0] * ph5 + p6[0] * ph6;
    let y = p1[1] * ph1 + p2[1] * ph2 + p3[1] * ph3 + p4[1] * ph4 + p5[1] * ph5 + p6[1] * ph6;
    let z = p1[2] * ph1 + p2[2] * ph2 + p3[2] * ph3 + p4[2] * ph4 + p5[2] * ph5 + p6[2] * ph6;
    let xvec = Vector3::new(x, y, z);

    //Interpolate the density q
    let q = q1 * ph1 + q2 * ph2 + q3 * ph3 + q4 * ph4 + q5 * ph5 + q6 * ph6;

    //Interpolate the normal vectors (vx, vy, vz)
    let vx = v1[0] * ph1 + v2[0] * ph2 + v3[0] * ph3 + v4[0] * ph4 + v5[0] * ph5 + v6[0] * ph6;
    let vy = v1[1] * ph1 + v2[1] * ph2 + v3[1] * ph3 + v4[1] * ph4 + v5[1] * ph5 + v6[1] * ph6;
    let vz = v1[2] * ph1 + v2[2] * ph2 + v3[2] * ph3 + v4[2] * ph4 + v5[2] * ph5 + v6[2] * ph6;
    let v = Vector3::new(vx, vy, vz);

    //Evaluate xi derivatives of basis functions
    let dph2 = (2.0 * xi - al + eta * (al - ga) / gac) / alc;
    let dph3 = eta * (be + ga - 1.0) / (ga * bec);
    let dph4 = (1.0 - 2.0 * xi - eta) / alalc;
    let dph5 = eta / gagac;
    let dph6 = -eta / bebec;
    let dph1 = -dph2 - dph3 - dph4 - dph5 - dph6;

    //Compute dx/dxi from xi derivatives of phi
    let dx_dxi =
        p1[0] * dph1 + p2[0] * dph2 + p3[0] * dph3 + p4[0] * dph4 + p5[0] * dph5 + p6[0] * dph6;
    let dy_dxi =
        p1[1] * dph1 + p2[1] * dph2 + p3[1] * dph3 + p4[1] * dph4 + p5[1] * dph5 + p6[1] * dph6;
    let dz_dxi =
        p1[2] * dph1 + p2[2] * dph2 + p3[2] * dph3 + p4[2] * dph4 + p5[2] * dph5 + p6[2] * dph6;
    let ddxi = Vector3::new(dx_dxi, dy_dxi, dz_dxi);

    //Evaluate eta derivatives of basis functions
    let pph2 = xi * (al - ga) / (alc * gac);
    let pph3 = (2.0 * eta - be + xi * (be + ga - 1.0) / ga) / bec;
    let pph4 = -xi / alalc;
    let pph5 = xi / gagac;
    let pph6 = (1.0 - xi - 2.0 * eta) / bebec;
    let pph1 = -pph2 - pph3 - pph4 - pph5 - pph6;

    //Compute Dx/Deta from eta derivatives of phi
    let dx_det =
        p1[0] * pph1 + p2[0] * pph2 + p3[0] * pph3 + p4[0] * pph4 + p5[0] * pph5 + p6[0] * pph6;
    let dy_det =
        p1[1] * pph1 + p2[1] * pph2 + p3[1] * pph3 + p4[1] * pph4 + p5[1] * pph5 + p6[1] * pph6;
    let dz_det =
        p1[2] * pph1 + p2[2] * pph2 + p3[2] * pph3 + p4[2] * pph4 + p5[2] * pph5 + p6[2] * pph6;
    let ddet = Vector3::new(dx_det, dy_det, dz_det);

    let vn = ddxi.cross(&ddet);
    let hs = vn.norm();

    // vn = vn.normalize();

    (xvec, v, hs, q)
}

///Integrates the single-layer potential over element k from point x0.
pub fn lslp_3d_integral(
    x0: Vector3<f64>,
    k: usize,
    mint: usize,
    f: &DVector<f64>,
    p: &DMatrix<f64>,
    n: &DMatrix<usize>,
    alpha: &DVector<f64>,
    beta: &DVector<f64>,
    gamma: &DVector<f64>,
    xiq: &DVector<f64>,
    etq: &DVector<f64>,
    wq: &DVector<f64>,
) -> (f64, f64) {
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

        let (x, hs, f_int) = lslp_3d_interp(
            p1, p2, p3, p4, p5, p6, f1, f2, f3, f4, f5, f6, al, be, ga, xi, eta,
        );

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

pub fn ldlp_3d_integral(
    x0: Vector3<f64>,
    k: usize,
    mint: usize,
    q: &DVector<f64>,
    q0: f64,
    p: &DMatrix<f64>,
    n: &DMatrix<usize>,
    vna: &DMatrix<f64>,
    alpha: &DVector<f64>,
    beta: &DVector<f64>,
    gamma: &DVector<f64>,
    xiq: &DVector<f64>,
    etq: &DVector<f64>,
    wq: &DVector<f64>,
) -> (f64, f64) {
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

        let (xvec, v, hs, qint) = ldlp_3d_interp(
            p1, p2, p3, p4, p5, p6, vna1, vna2, vna3, vna4, vna5, vna6, q1, q2, q3, q4, q5, q6, al,
            be, ga, xi, eta,
        );

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
pub fn lslp_3d_integral_sing(
    ngl: usize,
    p1: Vector3<f64>,
    p2: Vector3<f64>,
    p3: Vector3<f64>,
    f1: f64,
    f2: f64,
    f3: f64,
    zz: &DVector<f64>,
    ww: &DVector<f64>,
) -> (f64, f64) {
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
pub fn ldlp_3d_integral_sing(
    ngl: usize,
    p1: Vector3<f64>,
    p2: Vector3<f64>,
    p3: Vector3<f64>,
    q1: f64,
    q2: f64,
    q3: f64,
    q0: f64,
    v1: Vector3<f64>,
    v2: Vector3<f64>,
    v3: Vector3<f64>,
    zz: &DVector<f64>,
    ww: &DVector<f64>,
) -> f64 {
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

            let x = p1[0] * zt + p2[0] * xi + p3[0] * et;
            let y = p1[1] * zt + p2[1] * xi + p3[1] * et;
            let z = p1[2] * zt + p2[2] * xi + p3[2] * et;
            let xvec = Vector3::new(x, y, z);

            let q_interp = q1 * zt + q2 * xi + q3 * et;
            let vx = v1[0] * zt + v2[0] * xi + v3[0] * et;
            let vy = v1[1] * zt + v2[1] * xi + v3[1] * et;
            let vz = v1[2] * zt + v2[2] * xi + v3[2] * et;
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

pub fn ldlp_3d(
    npts: usize,
    nelm: usize,
    mint: usize,
    nq: usize,
    q: &DVector<f64>,
    p: &DMatrix<f64>,
    n: &DMatrix<usize>,
    vna: &DMatrix<f64>,
    alpha: &DVector<f64>,
    beta: &DVector<f64>,
    gamma: &DVector<f64>,
    xiq: &DVector<f64>,
    etq: &DVector<f64>,
    wq: &DVector<f64>,
    zz: &DVector<f64>,
    ww: &DVector<f64>,
) -> DVector<f64> {
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

            let test =
                Vector6::new(q1.abs(), q2.abs(), q3.abs(), q4.abs(), q5.abs(), q6.abs()).sum();

            if test > tol {
                if i == i1 {
                    let pa = Vector3::new(p[(i1, 0)], p[(i1, 1)], p[(i1, 2)]);
                    let pb = Vector3::new(p[(i2, 0)], p[(i2, 1)], p[(i2, 2)]);
                    let pc = Vector3::new(p[(i3, 0)], p[(i3, 1)], p[(i3, 2)]);
                    let va = Vector3::new(vna[(i1, 0)], vna[(i1, 1)], vna[(i1, 2)]);
                    let vb = Vector3::new(vna[(i2, 0)], vna[(i2, 1)], vna[(i2, 2)]);
                    let vc = Vector3::new(vna[(i3, 0)], vna[(i3, 1)], vna[(i3, 2)]);
                    ptl +=
                        ldlp_3d_integral_sing(nq, pa, pb, pc, q1, q2, q3, q0, va, vb, vc, zz, ww);
                } else if i == i2 {
                    let (ia, ib, ic) = (i2, i3, i1);
                    let pa = Vector3::new(p[(ia, 0)], p[(ia, 1)], p[(ia, 2)]);
                    let pb = Vector3::new(p[(ib, 0)], p[(ib, 1)], p[(ib, 2)]);
                    let pc = Vector3::new(p[(ic, 0)], p[(ic, 1)], p[(ic, 2)]);
                    let va = Vector3::new(vna[(ia, 0)], vna[(ia, 1)], vna[(ia, 2)]);
                    let vb = Vector3::new(vna[(ib, 0)], vna[(ib, 1)], vna[(ib, 2)]);
                    let vc = Vector3::new(vna[(ic, 0)], vna[(ic, 1)], vna[(ic, 2)]);
                    ptl += ldlp_3d_integral_sing(
                        nq, pa, pb, pc, q[ia], q[ib], q[ic], q0, va, vb, vc, zz, ww,
                    );
                } else if i == i3 {
                    let (ia, ib, ic) = (i3, i1, i2);
                    let pa = Vector3::new(p[(ia, 0)], p[(ia, 1)], p[(ia, 2)]);
                    let pb = Vector3::new(p[(ib, 0)], p[(ib, 1)], p[(ib, 2)]);
                    let pc = Vector3::new(p[(ic, 0)], p[(ic, 1)], p[(ic, 2)]);
                    let va = Vector3::new(vna[(ia, 0)], vna[(ia, 1)], vna[(ia, 2)]);
                    let vb = Vector3::new(vna[(ib, 0)], vna[(ib, 1)], vna[(ib, 2)]);
                    let vc = Vector3::new(vna[(ic, 0)], vna[(ic, 1)], vna[(ic, 2)]);
                    ptl += ldlp_3d_integral_sing(
                        nq, pa, pb, pc, q[ia], q[ib], q[ic], q0, va, vb, vc, zz, ww,
                    );
                } else if i == i4 {
                    for &(ia, ib, ic) in &[(i4, i6, i1), (i4, i3, i6), (i4, i5, i3), (i4, i2, i5)] {
                        let pa = Vector3::new(p[(ia, 0)], p[(ia, 1)], p[(ia, 2)]);
                        let pb = Vector3::new(p[(ib, 0)], p[(ib, 1)], p[(ib, 2)]);
                        let pc = Vector3::new(p[(ic, 0)], p[(ic, 1)], p[(ic, 2)]);
                        let va = Vector3::new(vna[(ia, 0)], vna[(ia, 1)], vna[(ia, 2)]);
                        let vb = Vector3::new(vna[(ib, 0)], vna[(ib, 1)], vna[(ib, 2)]);
                        let vc = Vector3::new(vna[(ic, 0)], vna[(ic, 1)], vna[(ic, 2)]);
                        ptl += ldlp_3d_integral_sing(
                            nq, pa, pb, pc, q[ia], q[ib], q[ic], q0, va, vb, vc, zz, ww,
                        );
                    }
                } else if i == i5 {
                    for &(ia, ib, ic) in &[(i5, i4, i2), (i5, i1, i4), (i5, i6, i1), (i5, i3, i6)] {
                        let pa = Vector3::new(p[(ia, 0)], p[(ia, 1)], p[(ia, 2)]);
                        let pb = Vector3::new(p[(ib, 0)], p[(ib, 1)], p[(ib, 2)]);
                        let pc = Vector3::new(p[(ic, 0)], p[(ic, 1)], p[(ic, 2)]);
                        let va = Vector3::new(vna[(ia, 0)], vna[(ia, 1)], vna[(ia, 2)]);
                        let vb = Vector3::new(vna[(ib, 0)], vna[(ib, 1)], vna[(ib, 2)]);
                        let vc = Vector3::new(vna[(ic, 0)], vna[(ic, 1)], vna[(ic, 2)]);
                        ptl += ldlp_3d_integral_sing(
                            nq, pa, pb, pc, q[ia], q[ib], q[ic], q0, va, vb, vc, zz, ww,
                        );
                    }
                } else if i == i6 {
                    for &(ia, ib, ic) in &[(i6, i1, i4), (i6, i4, i2), (i6, i2, i5), (i6, i5, i3)] {
                        let pa = Vector3::new(p[(ia, 0)], p[(ia, 1)], p[(ia, 2)]);
                        let pb = Vector3::new(p[(ib, 0)], p[(ib, 1)], p[(ib, 2)]);
                        let pc = Vector3::new(p[(ic, 0)], p[(ic, 1)], p[(ic, 2)]);
                        let va = Vector3::new(vna[(ia, 0)], vna[(ia, 1)], vna[(ia, 2)]);
                        let vb = Vector3::new(vna[(ib, 0)], vna[(ib, 1)], vna[(ib, 2)]);
                        let vc = Vector3::new(vna[(ic, 0)], vna[(ic, 1)], vna[(ic, 2)]);
                        ptl += ldlp_3d_integral_sing(
                            nq, pa, pb, pc, q[ia], q[ib], q[ic], q0, va, vb, vc, zz, ww,
                        );
                    }
                } else {
                    let (pptl, _arelm) = ldlp_3d_integral(
                        p0, k, mint, q, q0, p, n, vna, alpha, beta, gamma, xiq, etq, wq,
                    );
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
pub fn ldlp_3d_assemble(
    npts: usize,
    nelm: usize,
    mint: usize,
    nq: usize,
    p: &DMatrix<f64>,
    n: &DMatrix<usize>,
    vna: &DMatrix<f64>,
    alpha: &DVector<f64>,
    beta: &DVector<f64>,
    gamma: &DVector<f64>,
    xiq: &DVector<f64>,
    etq: &DVector<f64>,
    wq: &DVector<f64>,
    zz: &DVector<f64>,
    ww: &DVector<f64>,
) -> DMatrix<f64> {
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
                        let pa = Vector3::new(p[(i1, 0)], p[(i1, 1)], p[(i1, 2)]);
                        let pb = Vector3::new(p[(i2, 0)], p[(i2, 1)], p[(i2, 2)]);
                        let pc = Vector3::new(p[(i3, 0)], p[(i3, 1)], p[(i3, 2)]);
                        let va = Vector3::new(vna[(i1, 0)], vna[(i1, 1)], vna[(i1, 2)]);
                        let vb = Vector3::new(vna[(i2, 0)], vna[(i2, 1)], vna[(i2, 2)]);
                        let vc = Vector3::new(vna[(i3, 0)], vna[(i3, 1)], vna[(i3, 2)]);
                        ptl += ldlp_3d_integral_sing(
                            nq, pa, pb, pc, q1, q2, q3, q0, va, vb, vc, zz, ww,
                        );
                    } else if i == i2 {
                        let (ia, ib, ic) = (i2, i3, i1);
                        let pa = Vector3::new(p[(ia, 0)], p[(ia, 1)], p[(ia, 2)]);
                        let pb = Vector3::new(p[(ib, 0)], p[(ib, 1)], p[(ib, 2)]);
                        let pc = Vector3::new(p[(ic, 0)], p[(ic, 1)], p[(ic, 2)]);
                        let va = Vector3::new(vna[(ia, 0)], vna[(ia, 1)], vna[(ia, 2)]);
                        let vb = Vector3::new(vna[(ib, 0)], vna[(ib, 1)], vna[(ib, 2)]);
                        let vc = Vector3::new(vna[(ic, 0)], vna[(ic, 1)], vna[(ic, 2)]);
                        ptl += ldlp_3d_integral_sing(
                            nq, pa, pb, pc, q[ia], q[ib], q[ic], q0, va, vb, vc, zz, ww,
                        );
                    } else if i == i3 {
                        let (ia, ib, ic) = (i3, i1, i2);
                        let pa = Vector3::new(p[(ia, 0)], p[(ia, 1)], p[(ia, 2)]);
                        let pb = Vector3::new(p[(ib, 0)], p[(ib, 1)], p[(ib, 2)]);
                        let pc = Vector3::new(p[(ic, 0)], p[(ic, 1)], p[(ic, 2)]);
                        let va = Vector3::new(vna[(ia, 0)], vna[(ia, 1)], vna[(ia, 2)]);
                        let vb = Vector3::new(vna[(ib, 0)], vna[(ib, 1)], vna[(ib, 2)]);
                        let vc = Vector3::new(vna[(ic, 0)], vna[(ic, 1)], vna[(ic, 2)]);
                        ptl += ldlp_3d_integral_sing(
                            nq, pa, pb, pc, q[ia], q[ib], q[ic], q0, va, vb, vc, zz, ww,
                        );
                    } else if i == i4 {
                        for &(ia, ib, ic) in
                            &[(i4, i6, i1), (i4, i3, i6), (i4, i5, i3), (i4, i2, i5)]
                        {
                            let pa = Vector3::new(p[(ia, 0)], p[(ia, 1)], p[(ia, 2)]);
                            let pb = Vector3::new(p[(ib, 0)], p[(ib, 1)], p[(ib, 2)]);
                            let pc = Vector3::new(p[(ic, 0)], p[(ic, 1)], p[(ic, 2)]);
                            let va = Vector3::new(vna[(ia, 0)], vna[(ia, 1)], vna[(ia, 2)]);
                            let vb = Vector3::new(vna[(ib, 0)], vna[(ib, 1)], vna[(ib, 2)]);
                            let vc = Vector3::new(vna[(ic, 0)], vna[(ic, 1)], vna[(ic, 2)]);
                            ptl += ldlp_3d_integral_sing(
                                nq, pa, pb, pc, q[ia], q[ib], q[ic], q0, va, vb, vc, zz, ww,
                            );
                        }
                    } else if i == i5 {
                        for &(ia, ib, ic) in
                            &[(i5, i4, i2), (i5, i1, i4), (i5, i6, i1), (i5, i3, i6)]
                        {
                            let pa = Vector3::new(p[(ia, 0)], p[(ia, 1)], p[(ia, 2)]);
                            let pb = Vector3::new(p[(ib, 0)], p[(ib, 1)], p[(ib, 2)]);
                            let pc = Vector3::new(p[(ic, 0)], p[(ic, 1)], p[(ic, 2)]);
                            let va = Vector3::new(vna[(ia, 0)], vna[(ia, 1)], vna[(ia, 2)]);
                            let vb = Vector3::new(vna[(ib, 0)], vna[(ib, 1)], vna[(ib, 2)]);
                            let vc = Vector3::new(vna[(ic, 0)], vna[(ic, 1)], vna[(ic, 2)]);
                            ptl += ldlp_3d_integral_sing(
                                nq, pa, pb, pc, q[ia], q[ib], q[ic], q0, va, vb, vc, zz, ww,
                            );
                        }
                    } else if i == i6 {
                        for &(ia, ib, ic) in
                            &[(i6, i1, i4), (i6, i4, i2), (i6, i2, i5), (i6, i5, i3)]
                        {
                            let pa = Vector3::new(p[(ia, 0)], p[(ia, 1)], p[(ia, 2)]);
                            let pb = Vector3::new(p[(ib, 0)], p[(ib, 1)], p[(ib, 2)]);
                            let pc = Vector3::new(p[(ic, 0)], p[(ic, 1)], p[(ic, 2)]);
                            let va = Vector3::new(vna[(ia, 0)], vna[(ia, 1)], vna[(ia, 2)]);
                            let vb = Vector3::new(vna[(ib, 0)], vna[(ib, 1)], vna[(ib, 2)]);
                            let vc = Vector3::new(vna[(ic, 0)], vna[(ic, 1)], vna[(ic, 2)]);
                            ptl += ldlp_3d_integral_sing(
                                nq, pa, pb, pc, q[ia], q[ib], q[ic], q0, va, vb, vc, zz, ww,
                            );
                        }
                    } else {
                        let (pptl, _arelm) = ldlp_3d_integral(
                            p0, k, mint, &q, q0, p, n, vna, alpha, beta, gamma, xiq, etq, wq,
                        );
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
fn ldlp_3d_shape(al: f64, be: f64, ga: f64, xi: f64, eta: f64) -> ([f64; 6], [f64; 6], [f64; 6]) {
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
fn elem_quad(
    k: usize,
    mint: usize,
    p: &DMatrix<f64>,
    n: &DMatrix<usize>,
    vna: &DMatrix<f64>,
    alpha: &DVector<f64>,
    beta: &DVector<f64>,
    gamma: &DVector<f64>,
    xiq: &DVector<f64>,
    etq: &DVector<f64>,
    wq: &DVector<f64>,
) -> ElemQuad {
    let nodes = [
        n[(k, 0)],
        n[(k, 1)],
        n[(k, 2)],
        n[(k, 3)],
        n[(k, 4)],
        n[(k, 5)],
    ];
    let pv: [Vector3<f64>; 6] =
        std::array::from_fn(|m| Vector3::new(p[(nodes[m], 0)], p[(nodes[m], 1)], p[(nodes[m], 2)]));
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

    ElemQuad {
        nodes,
        x,
        v,
        cf,
        ph: ph_all,
    }
}

/// The six per-basis double-layer contributions of one element to field point
/// `x0` (non-singular case). The kernel `v . grad G` is evaluated once per
/// quadrature point and shared across all six shape functions.
#[inline]
fn elem_basis_row(eq: &ElemQuad, x0: &Vector3<f64>) -> [f64; 6] {
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
pub fn ldlp_3d_assemble_interactions(
    npts: usize,
    nelm: usize,
    mint: usize,
    nptss: &[usize],
    p: &DMatrix<f64>,
    n: &DMatrix<usize>,
    vna: &DMatrix<f64>,
    alpha: &DVector<f64>,
    beta: &DVector<f64>,
    gamma: &DVector<f64>,
    xiq: &DVector<f64>,
    etq: &DVector<f64>,
    wq: &DVector<f64>,
) -> DMatrix<f64> {
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

pub fn lslp_3d(
    npts: usize,
    nelm: usize,
    mint: usize,
    nq: usize,
    f: &DVector<f64>,
    p: &DMatrix<f64>,
    n: &DMatrix<usize>,
    vna: &DMatrix<f64>,
    alpha: &DVector<f64>,
    beta: &DVector<f64>,
    gamma: &DVector<f64>,
    xiq: &DVector<f64>,
    etq: &DVector<f64>,
    wq: &DVector<f64>,
    zz: &DVector<f64>,
    ww: &DVector<f64>,
) -> DVector<f64> {
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

                let test =
                    Vector6::new(f1.abs(), f2.abs(), f3.abs(), f4.abs(), f5.abs(), f6.abs()).sum();

                if test > tol {
                    //Check if singular point is one of the corner nodes i1, i2, i3
                    if i == i1 {
                        let p1 = Vector3::new(p[(i1, 0)], p[(i1, 1)], p[(i1, 2)]);
                        let p2 = Vector3::new(p[(i2, 0)], p[(i2, 1)], p[(i2, 2)]);
                        let p3 = Vector3::new(p[(i3, 0)], p[(i3, 1)], p[(i3, 2)]);

                        let (f1, f2, f3) = (f[i1], f[i2], f[i3]);

                        let (pptl, arelm) =
                            lslp_3d_integral_sing(nq, p1, p2, p3, f1, f2, f3, zz, ww);

                        ptl += pptl;
                        srf_area = srf_area + arelm;
                    } else if i == i2 {
                        let (i1, i2, i3) = (i2, i3, i1);

                        let p1 = Vector3::new(p[(i1, 0)], p[(i1, 1)], p[(i1, 2)]);
                        let p2 = Vector3::new(p[(i2, 0)], p[(i2, 1)], p[(i2, 2)]);
                        let p3 = Vector3::new(p[(i3, 0)], p[(i3, 1)], p[(i3, 2)]);

                        let (f1, f2, f3) = (f[i1], f[i2], f[i3]);

                        let (pptl, arelm) =
                            lslp_3d_integral_sing(nq, p1, p2, p3, f1, f2, f3, zz, ww);

                        ptl += pptl;
                        srf_area += arelm;
                    } else if i == i3 {
                        let (i1, i2, i3) = (i3, i1, i2);

                        let p1 = Vector3::new(p[(i1, 0)], p[(i1, 1)], p[(i1, 2)]);
                        let p2 = Vector3::new(p[(i2, 0)], p[(i2, 1)], p[(i2, 2)]);
                        let p3 = Vector3::new(p[(i3, 0)], p[(i3, 1)], p[(i3, 2)]);

                        let (f1, f2, f3) = (f[i1], f[i2], f[i3]);

                        let (pptl, arelm) =
                            lslp_3d_integral_sing(nq, p1, p2, p3, f1, f2, f3, zz, ww);

                        ptl += pptl;
                        srf_area += arelm;
                    }
                    //Check if the singular point is one of the edge nodes i4, i5, i6
                    else if i == i4 {
                        let (ia, ib, ic) = (i4, i6, i1);

                        let p1 = Vector3::new(p[(ia, 0)], p[(ia, 1)], p[(ia, 2)]);
                        let p2 = Vector3::new(p[(ib, 0)], p[(ib, 1)], p[(ib, 2)]);
                        let p3 = Vector3::new(p[(ic, 0)], p[(ic, 1)], p[(ic, 2)]);

                        let (f1, f2, f3) = (f[ia], f[ib], f[ic]);

                        let (pptl, arelm) =
                            lslp_3d_integral_sing(nq, p1, p2, p3, f1, f2, f3, zz, ww);

                        ptl += pptl;
                        srf_area += arelm;

                        let (ia, ib, ic) = (i4, i3, i6);

                        let p1 = Vector3::new(p[(ia, 0)], p[(ia, 1)], p[(ia, 2)]);
                        let p2 = Vector3::new(p[(ib, 0)], p[(ib, 1)], p[(ib, 2)]);
                        let p3 = Vector3::new(p[(ic, 0)], p[(ic, 1)], p[(ic, 2)]);

                        let (f1, f2, f3) = (f[ia], f[ib], f[ic]);

                        let (pptl, arelm) =
                            lslp_3d_integral_sing(nq, p1, p2, p3, f1, f2, f3, zz, ww);

                        ptl += pptl;
                        srf_area += arelm;

                        let (ia, ib, ic) = (i4, i5, i3);

                        let p1 = Vector3::new(p[(ia, 0)], p[(ia, 1)], p[(ia, 2)]);
                        let p2 = Vector3::new(p[(ib, 0)], p[(ib, 1)], p[(ib, 2)]);
                        let p3 = Vector3::new(p[(ic, 0)], p[(ic, 1)], p[(ic, 2)]);

                        let (f1, f2, f3) = (f[ia], f[ib], f[ic]);

                        let (pptl, arelm) =
                            lslp_3d_integral_sing(nq, p1, p2, p3, f1, f2, f3, zz, ww);

                        ptl += pptl;
                        srf_area += arelm;

                        let (ia, ib, ic) = (i4, i2, i5);

                        let p1 = Vector3::new(p[(ia, 0)], p[(ia, 1)], p[(ia, 2)]);
                        let p2 = Vector3::new(p[(ib, 0)], p[(ib, 1)], p[(ib, 2)]);
                        let p3 = Vector3::new(p[(ic, 0)], p[(ic, 1)], p[(ic, 2)]);

                        let (f1, f2, f3) = (f[ia], f[ib], f[ic]);

                        let (pptl, arelm) =
                            lslp_3d_integral_sing(nq, p1, p2, p3, f1, f2, f3, zz, ww);

                        ptl += pptl;
                        srf_area += arelm;
                    } else if i == i5 {
                        let (ia, ib, ic) = (i5, i4, i2);

                        let p1 = Vector3::new(p[(ia, 0)], p[(ia, 1)], p[(ia, 2)]);
                        let p2 = Vector3::new(p[(ib, 0)], p[(ib, 1)], p[(ib, 2)]);
                        let p3 = Vector3::new(p[(ic, 0)], p[(ic, 1)], p[(ic, 2)]);

                        let (f1, f2, f3) = (f[ia], f[ib], f[ic]);

                        let (pptl, arelm) =
                            lslp_3d_integral_sing(nq, p1, p2, p3, f1, f2, f3, zz, ww);

                        ptl += pptl;
                        srf_area += arelm;

                        let (ia, ib, ic) = (i5, i1, i4);

                        let p1 = Vector3::new(p[(ia, 0)], p[(ia, 1)], p[(ia, 2)]);
                        let p2 = Vector3::new(p[(ib, 0)], p[(ib, 1)], p[(ib, 2)]);
                        let p3 = Vector3::new(p[(ic, 0)], p[(ic, 1)], p[(ic, 2)]);

                        let (f1, f2, f3) = (f[ia], f[ib], f[ic]);

                        let (pptl, arelm) =
                            lslp_3d_integral_sing(nq, p1, p2, p3, f1, f2, f3, zz, ww);

                        ptl += pptl;
                        srf_area += arelm;

                        let (ia, ib, ic) = (i5, i6, i1);

                        let p1 = Vector3::new(p[(ia, 0)], p[(ia, 1)], p[(ia, 2)]);
                        let p2 = Vector3::new(p[(ib, 0)], p[(ib, 1)], p[(ib, 2)]);
                        let p3 = Vector3::new(p[(ic, 0)], p[(ic, 1)], p[(ic, 2)]);

                        let (f1, f2, f3) = (f[ia], f[ib], f[ic]);

                        let (pptl, arelm) =
                            lslp_3d_integral_sing(nq, p1, p2, p3, f1, f2, f3, zz, ww);

                        ptl += pptl;
                        srf_area += arelm;

                        let (ia, ib, ic) = (i5, i3, i6);

                        let p1 = Vector3::new(p[(ia, 0)], p[(ia, 1)], p[(ia, 2)]);
                        let p2 = Vector3::new(p[(ib, 0)], p[(ib, 1)], p[(ib, 2)]);
                        let p3 = Vector3::new(p[(ic, 0)], p[(ic, 1)], p[(ic, 2)]);

                        let (f1, f2, f3) = (f[ia], f[ib], f[ic]);

                        let (pptl, arelm) =
                            lslp_3d_integral_sing(nq, p1, p2, p3, f1, f2, f3, zz, ww);

                        ptl += pptl;
                        srf_area += arelm;
                    } else if i == i6 {
                        let (ia, ib, ic) = (i6, i1, i4);

                        let p1 = Vector3::new(p[(ia, 0)], p[(ia, 1)], p[(ia, 2)]);
                        let p2 = Vector3::new(p[(ib, 0)], p[(ib, 1)], p[(ib, 2)]);
                        let p3 = Vector3::new(p[(ic, 0)], p[(ic, 1)], p[(ic, 2)]);

                        let (f1, f2, f3) = (f[ia], f[ib], f[ic]);

                        let (pptl, arelm) =
                            lslp_3d_integral_sing(nq, p1, p2, p3, f1, f2, f3, zz, ww);

                        ptl += pptl;
                        srf_area += arelm;

                        let (ia, ib, ic) = (i6, i4, i2);

                        let p1 = Vector3::new(p[(ia, 0)], p[(ia, 1)], p[(ia, 2)]);
                        let p2 = Vector3::new(p[(ib, 0)], p[(ib, 1)], p[(ib, 2)]);
                        let p3 = Vector3::new(p[(ic, 0)], p[(ic, 1)], p[(ic, 2)]);

                        let (f1, f2, f3) = (f[ia], f[ib], f[ic]);

                        let (pptl, arelm) =
                            lslp_3d_integral_sing(nq, p1, p2, p3, f1, f2, f3, zz, ww);

                        ptl += pptl;
                        srf_area += arelm;

                        let (ia, ib, ic) = (i6, i2, i5);

                        let p1 = Vector3::new(p[(ia, 0)], p[(ia, 1)], p[(ia, 2)]);
                        let p2 = Vector3::new(p[(ib, 0)], p[(ib, 1)], p[(ib, 2)]);
                        let p3 = Vector3::new(p[(ic, 0)], p[(ic, 1)], p[(ic, 2)]);

                        let (f1, f2, f3) = (f[ia], f[ib], f[ic]);

                        let (pptl, arelm) =
                            lslp_3d_integral_sing(nq, p1, p2, p3, f1, f2, f3, zz, ww);

                        ptl += pptl;
                        srf_area += arelm;

                        let (ia, ib, ic) = (i6, i5, i3);

                        let p1 = Vector3::new(p[(ia, 0)], p[(ia, 1)], p[(ia, 2)]);
                        let p2 = Vector3::new(p[(ib, 0)], p[(ib, 1)], p[(ib, 2)]);
                        let p3 = Vector3::new(p[(ic, 0)], p[(ic, 1)], p[(ic, 2)]);

                        let (f1, f2, f3) = (f[ia], f[ib], f[ic]);

                        let (pptl, arelm) =
                            lslp_3d_integral_sing(nq, p1, p2, p3, f1, f2, f3, zz, ww);

                        ptl += pptl;
                        srf_area += arelm;
                    } else
                    //Do non-singular integral (precomputed element geometry)
                    {
                        let eq = &eqs[k];
                        for q in 0..mint {
                            let (g, _dg) = lgf_3d_fs(&eq.x[q], &p0);
                            let ph = &eq.ph[q];
                            let f_int = f1 * ph[0]
                                + f2 * ph[1]
                                + f3 * ph[2]
                                + f4 * ph[3]
                                + f5 * ph[4]
                                + f6 * ph[5];
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
pub fn lsdlpp_3d_interp(
    p1: Vector3<f64>,
    p2: Vector3<f64>,
    p3: Vector3<f64>,
    p4: Vector3<f64>,
    p5: Vector3<f64>,
    p6: Vector3<f64>,
    v1: Vector3<f64>,
    v2: Vector3<f64>,
    v3: Vector3<f64>,
    v4: Vector3<f64>,
    v5: Vector3<f64>,
    v6: Vector3<f64>,
    f1: f64,
    f2: f64,
    f3: f64,
    f4: f64,
    f5: f64,
    f6: f64,
    df1: f64,
    df2: f64,
    df3: f64,
    df4: f64,
    df5: f64,
    df6: f64,
    al: f64,
    be: f64,
    ga: f64,
    xi: f64,
    eta: f64,
) -> (Vector3<f64>, Vector3<f64>, f64, f64, f64) {
    let (alc, bec, gac) = (1.0 - al, 1.0 - be, 1.0 - ga);
    let (alalc, bebec, gagac) = (al * alc, be * bec, ga * gac);

    //Evaluate Basis functions
    let ph2 = xi * (xi - al + eta * (al - ga) / gac) / alc;
    let ph3 = eta * (eta - be + xi * (be + ga - 1.0) / ga) / bec;
    let ph4 = xi * (1.0 - xi - eta) / alalc;
    let ph5 = xi * eta / gagac;
    let ph6 = eta * (1.0 - xi - eta) / bebec;
    let ph1 = 1.0 - ph2 - ph3 - ph4 - ph5 - ph6;

    //Interpolate position vector
    let x = p1[0] * ph1 + p2[0] * ph2 + p3[0] * ph3 + p4[0] * ph4 + p5[0] * ph5 + p6[0] * ph6;
    let y = p1[1] * ph1 + p2[1] * ph2 + p3[1] * ph3 + p4[1] * ph4 + p5[1] * ph5 + p6[1] * ph6;
    let z = p1[2] * ph1 + p2[2] * ph2 + p3[2] * ph3 + p4[2] * ph4 + p5[2] * ph5 + p6[2] * ph6;
    let xvec = Vector3::new(x, y, z);

    //Interpolate the densities f and df
    let f = f1 * ph1 + f2 * ph2 + f3 * ph3 + f4 * ph4 + f5 * ph5 + f6 * ph6;
    let df = df1 * ph1 + df2 * ph2 + df3 * ph3 + df4 * ph4 + df5 * ph5 + df6 * ph6;

    //Interpolate the normal vectors (vx, vy, vz)
    let vx = v1[0] * ph1 + v2[0] * ph2 + v3[0] * ph3 + v4[0] * ph4 + v5[0] * ph5 + v6[0] * ph6;
    let vy = v1[1] * ph1 + v2[1] * ph2 + v3[1] * ph3 + v4[1] * ph4 + v5[1] * ph5 + v6[1] * ph6;
    let vz = v1[2] * ph1 + v2[2] * ph2 + v3[2] * ph3 + v4[2] * ph4 + v5[2] * ph5 + v6[2] * ph6;
    let v = Vector3::new(vx, vy, vz);

    //Evaluate xi derivatives of basis functions
    let dph2 = (2.0 * xi - al + eta * (al - ga) / gac) / alc;
    let dph3 = eta * (be + ga - 1.0) / (ga * bec);
    let dph4 = (1.0 - 2.0 * xi - eta) / alalc;
    let dph5 = eta / gagac;
    let dph6 = -eta / bebec;
    let dph1 = -dph2 - dph3 - dph4 - dph5 - dph6;

    //Compute dx/dxi from xi derivatives of phi
    let dx_dxi =
        p1[0] * dph1 + p2[0] * dph2 + p3[0] * dph3 + p4[0] * dph4 + p5[0] * dph5 + p6[0] * dph6;
    let dy_dxi =
        p1[1] * dph1 + p2[1] * dph2 + p3[1] * dph3 + p4[1] * dph4 + p5[1] * dph5 + p6[1] * dph6;
    let dz_dxi =
        p1[2] * dph1 + p2[2] * dph2 + p3[2] * dph3 + p4[2] * dph4 + p5[2] * dph5 + p6[2] * dph6;
    let ddxi = Vector3::new(dx_dxi, dy_dxi, dz_dxi);

    //Evaluate eta derivatives of basis functions
    let pph2 = xi * (al - ga) / (alc * gac);
    let pph3 = (2.0 * eta - be + xi * (be + ga - 1.0) / ga) / bec;
    let pph4 = -xi / alalc;
    let pph5 = xi / gagac;
    let pph6 = (1.0 - xi - 2.0 * eta) / bebec;
    let pph1 = -pph2 - pph3 - pph4 - pph5 - pph6;

    //Compute Dx/Deta from eta derivatives of phi
    let dx_det =
        p1[0] * pph1 + p2[0] * pph2 + p3[0] * pph3 + p4[0] * pph4 + p5[0] * pph5 + p6[0] * pph6;
    let dy_det =
        p1[1] * pph1 + p2[1] * pph2 + p3[1] * pph3 + p4[1] * pph4 + p5[1] * pph5 + p6[1] * pph6;
    let dz_det =
        p1[2] * pph1 + p2[2] * pph2 + p3[2] * pph3 + p4[2] * pph4 + p5[2] * pph5 + p6[2] * pph6;
    let ddet = Vector3::new(dx_det, dy_det, dz_det);

    let mut vn = ddxi.cross(&ddet);
    let hs = vn.norm();

    // vn = vn.normalize();

    (xvec, v, hs, f, df)
}
