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
use crate::bem::exact_geometry::ExactEllipsoidSurface;
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

fn ldlp_3d_integral_with_geom(
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
    exact: Option<&ExactEllipsoidSurface>,
) -> (f64, f64) {
    if exact.is_none() {
        return ldlp_3d_integral(
            x0, k, mint, q, q0, p, n, vna, alpha, beta, gamma, xiq, etq, wq,
        );
    }

    let eq = elem_quad(k, mint, p, n, vna, alpha, beta, gamma, xiq, etq, wq, exact);
    let mut area = 0.0;
    let mut ptl = 0.0;

    for qi in 0..eq.x.len() {
        let ph = &eq.ph[qi];
        let qint = ph
            .iter()
            .enumerate()
            .map(|(m, &phi)| phi * q[eq.nodes[m]])
            .sum::<f64>();
        let (_g, dg) = lgf_3d_fs(&eq.x[qi], &x0);

        area += eq.cf[qi];
        ptl += (qint - q0) * eq.v[qi].dot(&dg) * eq.cf[qi];
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

    if std::env::var_os("MULTI_ELLIP_LEGACY_SINGULAR_SLP").is_some() {
        // Diagnostic-only compatibility path for reproducing pre-fix studies.
        // The correct return order is (slp, area); the legacy order is not
        // physically correct because callers interpret the first value as SLP.
        return (area, slp);
    }

    // If all works, asm = area. Return order matches `lslp_3d_integral`:
    // (single-layer potential contribution, area).
    (slp, area)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bem::geom::gauss_leg;

    #[test]
    fn singular_single_layer_returns_potential_then_area() {
        let p1 = Vector3::new(0.0, 0.0, 0.0);
        let p2 = Vector3::new(1.0, 0.0, 0.0);
        let p3 = Vector3::new(0.0, 1.0, 0.0);
        let (zz, ww) = gauss_leg(12);

        let (slp_pos, area_pos) = lslp_3d_integral_sing(12, p1, p2, p3, 1.0, -0.3, 0.7, &zz, &ww);
        let (slp_neg, area_neg) = lslp_3d_integral_sing(12, p1, p2, p3, -1.0, 0.3, -0.7, &zz, &ww);
        let (slp_zero, area_zero) = lslp_3d_integral_sing(12, p1, p2, p3, 0.0, 0.0, 0.0, &zz, &ww);

        assert!((area_pos - 0.5).abs() < 1.0e-14);
        assert!((area_neg - area_pos).abs() < 1.0e-14);
        assert!((area_zero - area_pos).abs() < 1.0e-14);
        assert!((slp_pos + slp_neg).abs() < 1.0e-13);
        assert!(slp_zero.abs() < 1.0e-14);
    }
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
    desingularize: bool,
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

            let density = if desingularize {
                q_interp - q0
            } else {
                q_interp
            };
            btl += density * v.dot(&dg) * cf;
        }

        dlp += btl * ww[i] * rmaxh;
    }

    dlp * (pi / 4.0) * hs
}

#[inline]
fn local_param_nodes(al: f64, be: f64, ga: f64) -> [(f64, f64); 6] {
    [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (al, 0.0),
        (ga, 1.0 - ga),
        (0.0, be),
    ]
}

#[inline]
fn param_det(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> f64 {
    ((b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)).abs()
}

fn singular_param_triangles(
    al: f64,
    be: f64,
    ga: f64,
    local_node: usize,
) -> ([[(f64, f64); 3]; 4], usize) {
    let p = local_param_nodes(al, be, ga);
    let zero = [(0.0, 0.0); 3];
    match local_node {
        0 => ([[p[0], p[1], p[2]], zero, zero, zero], 1),
        1 => ([[p[1], p[2], p[0]], zero, zero, zero], 1),
        2 => ([[p[2], p[0], p[1]], zero, zero, zero], 1),
        3 => (
            [
                [p[3], p[5], p[0]],
                [p[3], p[2], p[5]],
                [p[3], p[4], p[2]],
                [p[3], p[1], p[4]],
            ],
            4,
        ),
        4 => (
            [
                [p[4], p[3], p[1]],
                [p[4], p[0], p[3]],
                [p[4], p[5], p[0]],
                [p[4], p[2], p[5]],
            ],
            4,
        ),
        5 => (
            [
                [p[5], p[0], p[3]],
                [p[5], p[3], p[1]],
                [p[5], p[1], p[4]],
                [p[5], p[4], p[2]],
            ],
            4,
        ),
        _ => panic!("invalid local node for singular element: {local_node}"),
    }
}

fn exact_singular_single_layer(
    ngl: usize,
    element: usize,
    local_node: usize,
    f: &[f64; 6],
    exact: &ExactEllipsoidSurface,
    al: f64,
    be: f64,
    ga: f64,
    zz: &DVector<f64>,
    ww: &DVector<f64>,
) -> (f64, f64) {
    let nodes = local_param_nodes(al, be, ga);
    let (xi0, eta0) = nodes[local_node];
    let x0 = exact.evaluate(element, xi0, eta0).position;
    let (tris, ntri) = singular_param_triangles(al, be, ga, local_node);

    let mut area = 0.0;
    let mut slp = 0.0;

    for tri in tris.iter().take(ntri) {
        let a = tri[0];
        let b = tri[1];
        let c = tri[2];
        let det = param_det(a, b, c);
        if det <= f64::EPSILON {
            continue;
        }

        for i in 0..ngl {
            let theta = (PI / 4.0) * (1.0 + zz[i]);
            let ct = theta.cos();
            let st = theta.sin();
            let rmax = 1.0 / (ct + st);
            let rmaxh = 0.5 * rmax;

            let mut area_theta = 0.0;
            let mut slp_theta = 0.0;

            for j in 0..ngl {
                let r = rmaxh * (1.0 + zz[j]);
                let lam = r * ct;
                let mu = r * st;
                let xi = a.0 + lam * (b.0 - a.0) + mu * (c.0 - a.0);
                let eta = a.1 + lam * (b.1 - a.1) + mu * (c.1 - a.1);

                let point = exact.evaluate(element, xi, eta);
                let shape = exact.shape_values(element, xi, eta);
                let f_int = f[0] * shape[0]
                    + f[1] * shape[1]
                    + f[2] * shape[2]
                    + f[3] * shape[3]
                    + f[4] * shape[4]
                    + f[5] * shape[5];
                let (g, _dg) = lgf_3d_fs(&point.position, &x0);
                let weight = point.jacobian * det * r * ww[j];

                area_theta += weight;
                slp_theta += f_int * g * weight;
            }

            let theta_weight = ww[i] * rmaxh * (PI / 4.0);
            area += area_theta * theta_weight;
            slp += slp_theta * theta_weight;
        }
    }

    (slp, area)
}

fn exact_singular_double_layer(
    ngl: usize,
    element: usize,
    local_node: usize,
    q: &[f64; 6],
    _q0: f64,
    exact: &ExactEllipsoidSurface,
    al: f64,
    be: f64,
    ga: f64,
    zz: &DVector<f64>,
    ww: &DVector<f64>,
) -> f64 {
    // Matrix assembly represents the exterior boundary operator K - 1/2 I.
    // On singular self-elements we therefore integrate the raw basis density
    // over its support and let the caller apply the global jump term once.
    // Subtracting q0 here would remove an O(h) support contribution from the
    // diagonal and gives first-order sphere convergence.
    let nodes = local_param_nodes(al, be, ga);
    let (xi0, eta0) = nodes[local_node];
    let x0 = exact.evaluate(element, xi0, eta0).position;
    let (tris, ntri) = singular_param_triangles(al, be, ga, local_node);

    let mut dlp = 0.0;

    for tri in tris.iter().take(ntri) {
        let a = tri[0];
        let b = tri[1];
        let c = tri[2];
        let det = param_det(a, b, c);
        if det <= f64::EPSILON {
            continue;
        }

        for i in 0..ngl {
            let theta = (PI / 4.0) * (1.0 + zz[i]);
            let ct = theta.cos();
            let st = theta.sin();
            let rmax = 1.0 / (ct + st);
            let rmaxh = 0.5 * rmax;

            let mut dlp_theta = 0.0;

            for j in 0..ngl {
                let r = rmaxh * (1.0 + zz[j]);
                let lam = r * ct;
                let mu = r * st;
                let xi = a.0 + lam * (b.0 - a.0) + mu * (c.0 - a.0);
                let eta = a.1 + lam * (b.1 - a.1) + mu * (c.1 - a.1);

                let point = exact.evaluate(element, xi, eta);
                let shape = exact.shape_values(element, xi, eta);
                let q_int = q[0] * shape[0]
                    + q[1] * shape[1]
                    + q[2] * shape[2]
                    + q[3] * shape[3]
                    + q[4] * shape[4]
                    + q[5] * shape[5];
                let (_g, dg) = lgf_3d_fs(&point.position, &x0);
                let weight = point.jacobian * det * r * ww[j];

                dlp_theta += q_int * point.normal.dot(&dg) * weight;
            }

            dlp += dlp_theta * ww[i] * rmaxh * (PI / 4.0);
        }
    }

    dlp
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
                    ptl += ldlp_3d_integral_sing(
                        nq, pa, pb, pc, q1, q2, q3, q0, va, vb, vc, zz, ww, true,
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
                        nq, pa, pb, pc, q[ia], q[ib], q[ic], q0, va, vb, vc, zz, ww, true,
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
                        nq, pa, pb, pc, q[ia], q[ib], q[ic], q0, va, vb, vc, zz, ww, true,
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
                            nq, pa, pb, pc, q[ia], q[ib], q[ic], q0, va, vb, vc, zz, ww, true,
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
                            nq, pa, pb, pc, q[ia], q[ib], q[ic], q0, va, vb, vc, zz, ww, true,
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
                            nq, pa, pb, pc, q[ia], q[ib], q[ic], q0, va, vb, vc, zz, ww, true,
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
    ldlp_3d_assemble_with_geom(
        npts, nelm, mint, nq, p, n, vna, alpha, beta, gamma, xiq, etq, wq, zz, ww, None, false,
    )
}

pub fn ldlp_3d_assemble_with_geom(
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
    exact: Option<&ExactEllipsoidSurface>,
    exact_singular: bool,
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
                    let local_i = [i1, i2, i3, i4, i5, i6].iter().position(|&node| node == i);

                    if exact_singular {
                        if let (Some(local_node), Some(exact)) = (local_i, exact) {
                            let q_values = [q[i1], q[i2], q[i3], q[i4], q[i5], q[i6]];
                            ptl += exact_singular_double_layer(
                                nq, k, local_node, &q_values, q0, exact, alpha[k], beta[k],
                                gamma[k], zz, ww,
                            );
                            continue;
                        }
                    }

                    if i == i1 {
                        let pa = Vector3::new(p[(i1, 0)], p[(i1, 1)], p[(i1, 2)]);
                        let pb = Vector3::new(p[(i2, 0)], p[(i2, 1)], p[(i2, 2)]);
                        let pc = Vector3::new(p[(i3, 0)], p[(i3, 1)], p[(i3, 2)]);
                        let va = Vector3::new(vna[(i1, 0)], vna[(i1, 1)], vna[(i1, 2)]);
                        let vb = Vector3::new(vna[(i2, 0)], vna[(i2, 1)], vna[(i2, 2)]);
                        let vc = Vector3::new(vna[(i3, 0)], vna[(i3, 1)], vna[(i3, 2)]);
                        ptl += ldlp_3d_integral_sing(
                            nq, pa, pb, pc, q1, q2, q3, q0, va, vb, vc, zz, ww, true,
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
                            nq, pa, pb, pc, q[ia], q[ib], q[ic], q0, va, vb, vc, zz, ww, true,
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
                            nq, pa, pb, pc, q[ia], q[ib], q[ic], q0, va, vb, vc, zz, ww, true,
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
                                nq, pa, pb, pc, q[ia], q[ib], q[ic], q0, va, vb, vc, zz, ww, true,
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
                                nq, pa, pb, pc, q[ia], q[ib], q[ic], q0, va, vb, vc, zz, ww, true,
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
                                nq, pa, pb, pc, q[ia], q[ib], q[ic], q0, va, vb, vc, zz, ww, true,
                            );
                        }
                    } else {
                        let (pptl, _arelm) = ldlp_3d_integral_with_geom(
                            p0, k, mint, &q, q0, p, n, vna, alpha, beta, gamma, xiq, etq, wq, exact,
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
    exact: Option<&ExactEllipsoidSurface>,
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

        if let Some(exact) = exact {
            let point = exact.evaluate(k, xi, eta);
            x.push(point.position);
            v.push(point.normal);
            cf.push(0.5 * point.jacobian * wq[qi]);
        } else {
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
        }
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

/// Assemble exact-singular self single-layer blocks for each body.
///
/// These blocks map nodal Neumann data on a body to the same body's
/// single-layer RHS contribution. They are invariant under rigid-body motion,
/// so the O(N^2) singular self work can be cached while cross-body interactions
/// are still recomputed every solve.
pub fn lslp_3d_assemble_self_blocks_exact_singular(
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
    zz: &DVector<f64>,
    ww: &DVector<f64>,
    exact: &ExactEllipsoidSurface,
) -> Vec<DMatrix<f64>> {
    let mut node_body = vec![0_usize; npts];
    let mut offsets = Vec::with_capacity(nptss.len());
    let mut off = 0;
    for (b, &np) in nptss.iter().enumerate() {
        offsets.push(off);
        for t in 0..np {
            node_body[off + t] = b;
        }
        off += np;
    }

    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); npts];
    for k in 0..nelm {
        for c in 0..6 {
            adj[n[(k, c)]].push(k);
        }
    }

    let eqs: Vec<ElemQuad> = (0..nelm)
        .into_par_iter()
        .map(|k| {
            elem_quad(
                k,
                mint,
                p,
                n,
                vna,
                alpha,
                beta,
                gamma,
                xiq,
                etq,
                wq,
                Some(exact),
            )
        })
        .collect();

    (0..nptss.len())
        .into_par_iter()
        .map(|b| {
            let np = nptss[b];
            let off = offsets[b];
            let mut block = DMatrix::zeros(np, np);

            block
                .as_mut_slice()
                .par_chunks_mut(np)
                .enumerate()
                .for_each(|(j_local, col)| {
                    let j = off + j_local;
                    for i_local in 0..np {
                        let i = off + i_local;
                        let x0 = Vector3::new(p[(i, 0)], p[(i, 1)], p[(i, 2)]);
                        let mut ptl = 0.0;

                        for &k in &adj[j] {
                            debug_assert_eq!(node_body[n[(k, 0)]], b);
                            let eq = &eqs[k];
                            let source_local = eq.nodes.iter().position(|&node| node == j).unwrap();
                            let field_local = eq.nodes.iter().position(|&node| node == i);

                            if let Some(local_node) = field_local {
                                let mut f_values = [0.0_f64; 6];
                                f_values[source_local] = 1.0;
                                let (pptl, _arelm) = exact_singular_single_layer(
                                    ww.len(),
                                    k,
                                    local_node,
                                    &f_values,
                                    exact,
                                    alpha[k],
                                    beta[k],
                                    gamma[k],
                                    zz,
                                    ww,
                                );
                                ptl += pptl;
                            } else {
                                for q in 0..eq.x.len() {
                                    let (g, _dg) = lgf_3d_fs(&eq.x[q], &x0);
                                    ptl += g * eq.ph[q][source_local] * eq.cf[q];
                                }
                            }
                        }

                        col[i_local] = ptl;
                    }
                });

            block
        })
        .collect()
}

/// Compute only cross-body single-layer RHS contributions.
///
/// Same-body blocks are omitted so callers can add cached self blocks. Cross
/// terms are smooth (no singular collocation), but change with relative body
/// motion and therefore remain freshly evaluated.
pub fn lslp_3d_interactions_with_geom(
    npts: usize,
    nelm: usize,
    mint: usize,
    nptss: &[usize],
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
    exact: Option<&ExactEllipsoidSurface>,
) -> DVector<f64> {
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
    let elem_body: Vec<usize> = (0..nelm).map(|k| node_body[n[(k, 0)]]).collect();

    let eqs: Vec<ElemQuad> = (0..nelm)
        .into_par_iter()
        .map(|k| elem_quad(k, mint, p, n, vna, alpha, beta, gamma, xiq, etq, wq, exact))
        .collect();

    let rhs_vals: Vec<f64> = (0..npts)
        .into_par_iter()
        .map(|i| {
            let ibody = node_body[i];
            let x0 = Vector3::new(p[(i, 0)], p[(i, 1)], p[(i, 2)]);
            let mut ptl = 0.0;

            for k in 0..nelm {
                if elem_body[k] == ibody {
                    continue;
                }
                let eq = &eqs[k];
                for q in 0..mint {
                    let (g, _dg) = lgf_3d_fs(&eq.x[q], &x0);
                    let ph = &eq.ph[q];
                    let f_int = f[eq.nodes[0]] * ph[0]
                        + f[eq.nodes[1]] * ph[1]
                        + f[eq.nodes[2]] * ph[2]
                        + f[eq.nodes[3]] * ph[3]
                        + f[eq.nodes[4]] * ph[4]
                        + f[eq.nodes[5]] * ph[5];
                    ptl += g * f_int * eq.cf[q];
                }
            }

            ptl
        })
        .collect();

    DVector::from_vec(rhs_vals)
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
    ldlp_3d_assemble_interactions_with_geom(
        npts, nelm, mint, nptss, p, n, vna, alpha, beta, gamma, xiq, etq, wq, None,
    )
}

pub fn ldlp_3d_assemble_interactions_with_geom(
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
    exact: Option<&ExactEllipsoidSurface>,
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
        .map(|k| elem_quad(k, mint, p, n, vna, alpha, beta, gamma, xiq, etq, wq, exact))
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
    lslp_3d_with_geom(
        npts, nelm, mint, nq, f, p, n, vna, alpha, beta, gamma, xiq, etq, wq, zz, ww, None, false,
    )
}

pub fn lslp_3d_with_geom(
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
    exact: Option<&ExactEllipsoidSurface>,
    exact_singular: bool,
) -> DVector<f64> {
    let tol = 1e-8;

    // Precompute each element's quadrature geometry once, reused across all
    // (non-singular) field points (the single layer needs position, Jacobian
    // and shape functions; the cached normal is simply unused here).
    let eqs: Vec<ElemQuad> = (0..nelm)
        .into_par_iter()
        .map(|k| elem_quad(k, mint, p, n, vna, alpha, beta, gamma, xiq, etq, wq, exact))
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
                    let local_i = [i1, i2, i3, i4, i5, i6].iter().position(|&node| node == i);

                    if exact_singular {
                        if let (Some(local_node), Some(exact)) = (local_i, exact) {
                            let f_values = [f1, f2, f3, f4, f5, f6];
                            let (pptl, arelm) = exact_singular_single_layer(
                                nq, k, local_node, &f_values, exact, alpha[k], beta[k], gamma[k],
                                zz, ww,
                            );
                            ptl += pptl;
                            srf_area += arelm;
                            continue;
                        }
                    }

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
