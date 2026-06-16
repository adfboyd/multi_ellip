//! Surface-gradient reconstruction (grad_3d_* family).
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

pub fn grad_3d_integral(
    p0: &Vector3<f64>,
    k: usize,
    mint: usize,
    f: &DVector<f64>,
    df: &DVector<f64>,
    p: &DMatrix<f64>,
    n: &DMatrix<usize>,
    vna: &DMatrix<f64>,
    alpha: &DVector<f64>,
    beta: &DVector<f64>,
    gamma: &DVector<f64>,
    xiq: &DVector<f64>,
    etq: &DVector<f64>,
    wq: &DVector<f64>,
) -> (Vector3<f64>, f64) {
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

        let (xvec, vn, hs, f_int, dfdn_int) = lsdlpp_3d_interp(
            p1, p2, p3, p4, p5, p6, vna1, vna2, vna3, vna4, vna5, vna6, f1, f2, f3, f4, f5, f6,
            df1, df2, df3, df4, df5, df6, al, be, ga, xi, eta,
        );

        // println!("p0 = {:?}, x = {:?}", p0, xvec);

        let (d_g, dd_g) = d_lgf_3d_fs_full(&xvec, p0);

        // println!("dg = {:?}", d_g);

        let cf = 0.5 * hs * wq[i];

        area += cf;

        let r_int = (-dfdn_int * d_g) + (f_int * dd_g * vn);

        sdlp += cf * r_int;
    }

    (sdlp, area)
}

pub fn grad_3d(
    nelm: usize,
    mint: usize,
    f: &DVector<f64>,
    dfdn: &DVector<f64>,
    p: &DMatrix<f64>,
    n: &DMatrix<usize>,
    vna: &DMatrix<f64>,
    alpha: &DVector<f64>,
    beta: &DVector<f64>,
    gamma: &DVector<f64>,
    xiq: &DVector<f64>,
    etq: &DVector<f64>,
    wq: &DVector<f64>,
    p0: &Vector3<f64>,
) -> Vector3<f64> {
    //Calculates the gradient of phi in the given direction dxi at point p0.

    let mut f0 = Vector3::zeros();

    for k in 0..nelm {
        let (sdlp, _arelm) = grad_3d_integral(
            p0, k, mint, f, dfdn, p, n, vna, alpha, beta, gamma, xiq, etq, wq,
        );

        f0 += sdlp;
    }
    f0
}

pub fn grad_3d_integral_l1(
    p0: &Vector3<f64>,
    k: usize,
    mint: usize,
    f: &DVector<f64>,
    dfdn: &DVector<f64>,
    p: &DMatrix<f64>,
    n: &DMatrix<usize>,
    vna: &DMatrix<f64>,
    alpha: &DVector<f64>,
    beta: &DVector<f64>,
    gamma: &DVector<f64>,
    xiq: &DVector<f64>,
    etq: &DVector<f64>,
    wq: &DVector<f64>,
) -> (Vector3<f64>, f64) {
    ///Does the integral on line 1 of eq(28). Carried out on outer subsurface.
    let (sdlp, area) = grad_3d_integral(
        p0, k, mint, f, dfdn, p, n, vna, alpha, beta, gamma, xiq, etq, wq,
    );

    (sdlp, area)
}

pub fn grad_3d_l1(
    nonsing_elms: &Vec<usize>,
    mint: usize,
    f: &DVector<f64>,
    dfdn: &DVector<f64>,
    p: &DMatrix<f64>,
    n: &DMatrix<usize>,
    vna: &DMatrix<f64>,
    alpha: &DVector<f64>,
    beta: &DVector<f64>,
    gamma: &DVector<f64>,
    xiq: &DVector<f64>,
    etq: &DVector<f64>,
    wq: &DVector<f64>,
    p0: &Vector3<f64>,
) -> Vector3<f64> {
    let mut f0 = Vector3::zeros();

    for &k in nonsing_elms {
        let (sdlp, area) = grad_3d_integral_l1(
            p0, k, mint, f, dfdn, p, n, vna, alpha, beta, gamma, xiq, etq, wq,
        );
        // println!("k = {:?}, sdlp = {:?}", k, sdlp);

        f0 += sdlp
    }

    f0
}

pub fn grad_3d_integral_l2(
    p0: &Vector3<f64>,
    k: usize,
    mint: usize,
    f: &DVector<f64>,
    df: &DVector<f64>,
    p: &DMatrix<f64>,
    n: &DMatrix<usize>,
    vna: &DMatrix<f64>,
    alpha: &DVector<f64>,
    beta: &DVector<f64>,
    gamma: &DVector<f64>,
    xiq: &DVector<f64>,
    etq: &DVector<f64>,
    wq: &DVector<f64>,
    dfdn_xi: f64,
) -> (Vector3<f64>, f64) {
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

        let (xvec, _vn, hs, _f_int, dfdn_int) = lsdlpp_3d_interp(
            p1, p2, p3, p4, p5, p6, vna1, vna2, vna3, vna4, vna5, vna6, f1, f2, f3, f4, f5, f6,
            df1, df2, df3, df4, df5, df6, al, be, ga, xi, eta,
        );

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

pub fn grad_3d_l2(
    sing_elms: &Vec<usize>,
    mint: usize,
    f: &DVector<f64>,
    dfdn: &DVector<f64>,
    p: &DMatrix<f64>,
    n: &DMatrix<usize>,
    vna: &DMatrix<f64>,
    alpha: &DVector<f64>,
    beta: &DVector<f64>,
    gamma: &DVector<f64>,
    xiq: &DVector<f64>,
    etq: &DVector<f64>,
    wq: &DVector<f64>,
    p0: &Vector3<f64>,
    f_p0: f64,
    dfdn_p0: f64,
) -> Vector3<f64> {
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
            let (sdlp, area) = grad_3d_integral_l2(
                p0, k, mint, f, dfdn, p, n, vna, alpha, beta, gamma, xiq, etq, wq, dfdn_p0,
            );
            f0 += sdlp;
        }
    }
    // f0 += - 0.5 * f_p0;

    f0
}

pub fn grad_3d_integral_l3_1(
    p0: &Vector3<f64>,
    k: usize,
    mint: usize,
    f: &DVector<f64>,
    df: &DVector<f64>,
    p: &DMatrix<f64>,
    n: &DMatrix<usize>,
    vna: &DMatrix<f64>,
    alpha: &DVector<f64>,
    beta: &DVector<f64>,
    gamma: &DVector<f64>,
    xiq: &DVector<f64>,
    etq: &DVector<f64>,
    wq: &DVector<f64>,
    f_xi: f64,
) -> (Vector3<f64>, f64) {
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

        let (xvec, vn, hs, f_int, _dfdn_int) = lsdlpp_3d_interp(
            p1, p2, p3, p4, p5, p6, vna1, vna2, vna3, vna4, vna5, vna6, f1, f2, f3, f4, f5, f6,
            df1, df2, df3, df4, df5, df6, al, be, ga, xi, eta,
        );

        let (_dg, mut dd_g) = d_lgf_3d_fs_full(&xvec, p0);

        if (xvec - p0).norm() < 1e-3 {
            dd_g = Matrix3::zeros()
        };

        let cf = 0.5 * hs * wq[i];

        area += cf;

        let r_int = Matrix3::from_diagonal_element(-(f_int - f_xi)) * dd_g * vn;

        sdlp += Matrix3::from_diagonal_element(cf) * r_int;
    }

    (sdlp, area)
}

pub fn grad_3d_l3_1(
    sing_elms: &Vec<usize>,
    mint: usize,
    f: &DVector<f64>,
    dfdn: &DVector<f64>,
    p: &DMatrix<f64>,
    n: &DMatrix<usize>,
    vna: &DMatrix<f64>,
    alpha: &DVector<f64>,
    beta: &DVector<f64>,
    gamma: &DVector<f64>,
    xiq: &DVector<f64>,
    etq: &DVector<f64>,
    wq: &DVector<f64>,
    p0: &Vector3<f64>,
    f_xi: f64,
) -> Vector3<f64> {
    let mut f0 = Vector3::zeros();

    for &k in sing_elms {
        let (sdlp, area) = grad_3d_integral_l3_1(
            p0, k, mint, f, dfdn, p, n, vna, alpha, beta, gamma, xiq, etq, wq, f_xi,
        );

        f0 += sdlp
    }
    f0
}

pub fn grad_3d_integral_l3_2(
    p0: &Vector3<f64>,
    k: usize,
    mint: usize,
    f: &DVector<f64>,
    df: &DVector<f64>,
    p: &DMatrix<f64>,
    n: &DMatrix<usize>,
    vna: &DMatrix<f64>,
    alpha: &DVector<f64>,
    beta: &DVector<f64>,
    gamma: &DVector<f64>,
    xiq: &DVector<f64>,
    etq: &DVector<f64>,
    wq: &DVector<f64>,
) -> (Matrix3<f64>, f64) {
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

        let (xvec, vn, hs, f_int, dfdn_int) = lsdlpp_3d_interp(
            p1, p2, p3, p4, p5, p6, vna1, vna2, vna3, vna4, vna5, vna6, f1, f2, f3, f4, f5, f6,
            df1, df2, df3, df4, df5, df6, al, be, ga, xi, eta,
        );

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
        let r_int = dd_g_dn * (xvec - p0).transpose();
        let cf_mat = Matrix3::from_diagonal_element(cf);
        sdlp += cf_mat * r_int;
    }
    // println!("Area = {:?}", area);
    (sdlp, area)
}

pub fn grad_3d_l3_2(
    sing_elms: &Vec<usize>,
    mint: usize,
    f: &DVector<f64>,
    dfdn: &DVector<f64>,
    p: &DMatrix<f64>,
    n: &DMatrix<usize>,
    vna: &DMatrix<f64>,
    alpha: &DVector<f64>,
    beta: &DVector<f64>,
    gamma: &DVector<f64>,
    xiq: &DVector<f64>,
    etq: &DVector<f64>,
    wq: &DVector<f64>,
    p0: &Vector3<f64>,
) -> Matrix3<f64> {
    let mut f0 = Matrix3::zeros();
    let mut area_tot = 0.0;

    for &k in sing_elms {
        let (sdlp, area) = grad_3d_integral_l3_2(
            p0, k, mint, f, dfdn, p, n, vna, alpha, beta, gamma, xiq, etq, wq,
        );

        f0 += sdlp;
        area_tot += area;
    }
    // println!("Area = {:?}", area_tot);
    f0
}

pub fn grad_3d_integral_l5(
    p0: &Vector3<f64>,
    k: usize,
    mint: usize,
    f: &DVector<f64>,
    df: &DVector<f64>,
    p: &DMatrix<f64>,
    n: &DMatrix<usize>,
    vna: &DMatrix<f64>,
    alpha: &DVector<f64>,
    beta: &DVector<f64>,
    gamma: &DVector<f64>,
    xiq: &DVector<f64>,
    etq: &DVector<f64>,
    wq: &DVector<f64>,
    n_xi: &Vector3<f64>,
) -> (Matrix3<f64>, f64) {
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

        let (xvec, vn, hs, f_int, dfdn_int) = lsdlpp_3d_interp(
            p1, p2, p3, p4, p5, p6, vna1, vna2, vna3, vna4, vna5, vna6, f1, f2, f3, f4, f5, f6,
            df1, df2, df3, df4, df5, df6, al, be, ga, xi, eta,
        );

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

pub fn grad_3d_l5(
    sing_elms: &Vec<usize>,
    mint: usize,
    f: &DVector<f64>,
    dfdn: &DVector<f64>,
    p: &DMatrix<f64>,
    n: &DMatrix<usize>,
    vna: &DMatrix<f64>,
    alpha: &DVector<f64>,
    beta: &DVector<f64>,
    gamma: &DVector<f64>,
    xiq: &DVector<f64>,
    etq: &DVector<f64>,
    wq: &DVector<f64>,
    p0: &Vector3<f64>,
    p0_n: &Vector3<f64>,
) -> Matrix3<f64> {
    let mut f0 = Matrix3::zeros();

    for &k in sing_elms {
        let (sdlp, area) = grad_3d_integral_l5(
            p0, k, mint, f, dfdn, p, n, vna, alpha, beta, gamma, xiq, etq, wq, p0_n,
        );

        f0 += sdlp
    }
    f0
}

fn grad_3d_integral_l4_1(
    p0: &Vector3<f64>,
    k: usize,
    p: &DMatrix<f64>,
    n_line: &DMatrix<usize>,
) -> (Vector3<f64>, f64) {
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

pub fn grad_3d_l4_1(p: &DMatrix<f64>, n_line: &DMatrix<usize>, p0: &Vector3<f64>) -> Vector3<f64> {
    let mut f0 = Vector3::zeros();

    for k in 0..n_line.shape().0 {
        let (sdlp, length) = grad_3d_integral_l4_1(p0, k, p, n_line);

        f0 += sdlp;
    }
    f0
}

fn grad_3d_integral_l4_2(
    p0: &Vector3<f64>,
    k: usize,
    p: &DMatrix<f64>,
    n_line: &DMatrix<usize>,
) -> (Matrix3<Vector3<f64>>, f64) {
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
                outer_product1[(i, j)][k] = h1[i] * dg1[j] * r1[k];
            }
        }
    }

    let r2 = p23 - p0;

    let mut outer_product2: Matrix3<Vector3<f64>> = Matrix3::zeros();
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                outer_product2[(i, j)][k] = h2[i] * dg2[j] * r2[k];
            }
        }
    }
    let integral = outer_product1 + outer_product2;
    let length = h1.norm() + h2.norm();

    (integral, length)
}

pub fn grad_3d_l4_2(
    p: &DMatrix<f64>,
    n_line: &DMatrix<usize>,
    p0: &Vector3<f64>,
) -> Matrix3<Vector3<f64>> {
    let mut f0: Matrix3<Vector3<f64>> = Matrix3::zeros();

    for k in 0..n_line.shape().0 {
        let (sdlp, length) = grad_3d_integral_l4_2(p0, k, p, n_line);

        f0 += sdlp;
    }
    f0
}

pub fn grad_3d_integral_l6_1(
    p0: &Vector3<f64>,
    k: usize,
    p: &DMatrix<f64>,
    n_line: &DMatrix<usize>,
) -> Vector3<f64> {
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

pub fn midpoint_gen(p1: &Vector3<f64>, p2: &Vector3<f64>, f1: f64, f2: f64) -> (Vector3<f64>, f64) {
    let p_12 = (p1 + p2) / 2.0;
    let f_12 = (f1 + f2) / 2.0;

    (p_12, f_12)
}

pub fn grad_3d_l6_1(p: &DMatrix<f64>, n_line: &DMatrix<usize>, p0: &Vector3<f64>) -> Vector3<f64> {
    let mut f0: Vector3<f64> = Vector3::zeros();

    for k in 0..n_line.shape().0 {
        let sdlp = grad_3d_integral_l6_1(p0, k, p, n_line);

        f0 += sdlp;
    }
    f0
}

pub fn grad_3d_all_rhs(
    sing_elms: &Vec<usize>,
    nonsing_elms: &Vec<usize>,
    mint: usize,
    f: &DVector<f64>,
    dfdn: &DVector<f64>,
    p: &DMatrix<f64>,
    n: &DMatrix<usize>,
    n_line: &DMatrix<usize>,
    vna: &DMatrix<f64>,
    alpha: &DVector<f64>,
    beta: &DVector<f64>,
    gamma: &DVector<f64>,
    xiq: &DVector<f64>,
    etq: &DVector<f64>,
    wq: &DVector<f64>,
    p0: &Vector3<f64>,
    f_p0: f64,
    dfdn_p0: f64,
) -> Vector3<f64> {
    let l1 = grad_3d_l1(
        nonsing_elms,
        mint,
        f,
        dfdn,
        p,
        n,
        vna,
        alpha,
        beta,
        gamma,
        xiq,
        etq,
        wq,
        p0,
    );

    let l2 = grad_3d_l2(
        sing_elms, mint, f, dfdn, p, n, vna, alpha, beta, gamma, xiq, etq, wq, p0, f_p0, dfdn_p0,
    );

    let l3_1 = grad_3d_l3_1(
        sing_elms, mint, f, dfdn, p, n, vna, alpha, beta, gamma, xiq, etq, wq, p0, f_p0,
    );

    let l6_1 = grad_3d_l6_1(p, n_line, p0) * f_p0;

    // println!("l1 = {:?}",l1);//    , l2 = {:?}, l3_1 = {:?}, l6_1 = {:?}", l1, l2, l3_1, l6_1);

    let rhs = l1 + l2 + l3_1 + l6_1;

    // println!("Rhs = {:?}",rhs);

    rhs
}

fn skew_symmetric_from_vec(v: &Vector3<f64>) -> Matrix3<f64> {
    let (v0, v1, v2) = (v[0], v[1], v[2]);

    let matrix = Matrix3::new(0.0, -v2, v1, v2, 0.0, -v0, -v1, v0, 0.0);

    matrix
}
pub fn grad_3d_all_lhs(
    sing_elms: &Vec<usize>,
    nonsing_elms: &Vec<usize>,
    mint: usize,
    f: &DVector<f64>,
    dfdn: &DVector<f64>,
    p: &DMatrix<f64>,
    n: &DMatrix<usize>,
    n_line: &DMatrix<usize>,
    vna: &DMatrix<f64>,
    alpha: &DVector<f64>,
    beta: &DVector<f64>,
    gamma: &DVector<f64>,
    xiq: &DVector<f64>,
    etq: &DVector<f64>,
    wq: &DVector<f64>,
    p0: &Vector3<f64>,
    p0_n: &Vector3<f64>,
    f_p0: f64,
    dfdn_p0: f64,
) -> Matrix3<f64> {
    let l3_2 = grad_3d_l3_2(
        sing_elms, mint, f, dfdn, p, n, vna, alpha, beta, gamma, xiq, etq, wq, p0,
    );

    let l4_1_vec = -grad_3d_l4_1(p, n_line, p0);
    //the full term on the rhs is u cross l4_1_vec, so -l4_1_vec cross u. (hence the -grad...())

    let mut l4_1 = skew_symmetric_from_vec(&l4_1_vec);

    let l4_2_tensor = grad_3d_l4_2(p, n_line, p0);

    let row0 = l4_2_tensor[(1, 2)] - l4_2_tensor[(2, 1)];
    let row1 = l4_2_tensor[(2, 0)] - l4_2_tensor[(0, 2)];
    let row2 = l4_2_tensor[(0, 1)] - l4_2_tensor[(1, 0)];

    let l4_2 = Matrix3::new(
        row0[0], row0[1], row0[2], row1[0], row1[1], row1[2], row2[0], row2[1], row2[2],
    );

    let l5 = grad_3d_l5(
        sing_elms, mint, f, dfdn, p, n, vna, alpha, beta, gamma, xiq, etq, wq, p0, p0_n,
    );

    let l6_2_scalar = 0.5; //Solid angle/4pi should be 0.5 in the limit as a point approaches the surface.

    let l6_2 = Matrix3::from_diagonal_element(l6_2_scalar);

    //When rearranging eq(28), these terms are all subtracted from I and multiplied by u to give the rhs.

    let id_mat = Matrix3::from_diagonal_element(1.0);
    //l3_2, l4_2, l5 are the assymetric ones
    // println!("l3_2 = {:?}", l3_2);

    let mat = id_mat - l3_2 - l4_1 - l4_2 - l5 - l6_2;

    mat
}

pub fn dphi(p1: &Vector3<f64>, p2: &Vector3<f64>, f1: f64, f2: f64) -> Vector3<f64> {
    let dx = p2 - p1;
    let df = f2 - f1;

    // let mag_dx = dx.norm();
    //
    // let dphi = df/mag_dx;
    //
    // (dphi,dx.normalize())
    let mut dphi = Vector3::new(0.0, 0.0, 0.0);
    for i in 0..3 {
        dphi[i] = df / dx[i]
    }
    dphi
}
