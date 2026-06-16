//! Post-solve surface integrals: gradient interpolation, hydrodynamic forces, Lamb impulse, fluid KE.
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

pub fn gradient_interp(
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
) -> (
    Vector3<f64>,
    Vector3<f64>,
    f64,
    f64,
    f64,
    f64,
    f64,
    Vector3<f64>,
    Vector3<f64>,
) {
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
    //Compute df/dxi
    let dfdxi = f1 * dph1 + f2 * dph2 + f3 * dph3 + f4 * dph4 + f5 * dph5 + f6 * dph6;

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
    //Compute df/deta
    let dfdet = f1 * pph1 + f2 * pph2 + f3 * pph3 + f4 * pph4 + f5 * pph5 + f6 * pph6;

    let mut vn = ddxi.cross(&ddet);
    let hs = vn.norm();

    // vn = vn.normalize();

    (xvec, v, hs, f, df, dfdxi, dfdet, ddxi, ddet)
}

pub fn line_interp_vector(p1: Vector3<f64>, p2: Vector3<f64>, xi: f64) -> Vector3<f64> {
    let p = (1.0 - xi) * p1 + xi * p2;
    p
}

fn line_interp_scalar(f1: f64, f2: f64, xi: f64) -> f64 {
    let f = (1.0 - xi) * f1 + xi * f2;
    f
}

pub fn line_interp(
    p1: Vector3<f64>,
    p2: Vector3<f64>,
    p3: Vector3<f64>,
    f1: f64,
    f2: f64,
    f3: f64,
    df1: f64,
    df2: f64,
    df3: f64,
    xi: f64,
) -> (Vector3<f64>, f64, f64) {
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

pub fn sine_find(a: &Vector3<f64>, b: &Vector3<f64>) -> f64 {
    (a.cross(b)).norm() / (a.norm() * b.norm())
}

pub fn gradient_interp_3d_integral(
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
) -> (f64, f64) {
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

        let (_xvec, _v, hs, _f_int, dfdn_int, dfdxi, dfdet, ddxi, ddet) = gradient_interp(
            p1, p2, p3, p4, p5, p6, vna1, vna2, vna3, vna4, vna5, vna6, f1, f2, f3, f4, f5, f6,
            df1, df2, df3, df4, df5, df6, al, be, ga, xi, eta,
        );

        // |∇_s φ|² = |dfdxi·ddet - dfdet·ddxi|² / hs² (surface metric correction)
        let tangential = dfdxi * ddet - dfdet * ddxi;
        let r_int = tangential.norm_squared() / (hs * hs) + dfdn_int * dfdn_int;

        let cf = 0.5 * hs * wq[i];

        area += cf;

        sdlp += r_int * cf;
    }

    (sdlp, area)
}

pub fn pressure_force_element(
    k: usize,
    mint: usize,
    body_centre: &Vector3<f64>,
    rho_f: f64,
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
) -> (Vector3<f64>, Vector3<f64>) {
    let mut force = Vector3::zeros();
    let mut torque = Vector3::zeros();

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
    let (df1, df2, df3, df4, df5, df6) =
        (dfdn[i1], dfdn[i2], dfdn[i3], dfdn[i4], dfdn[i5], dfdn[i6]);
    let (al, be, ga) = (alpha[k], beta[k], gamma[k]);

    for i in 0..mint {
        let (_xvec, _v, hs, _f_int, dfdn_int, dfdxi, dfdet, ddxi, ddet) = gradient_interp(
            p1, p2, p3, p4, p5, p6, vna1, vna2, vna3, vna4, vna5, vna6, f1, f2, f3, f4, f5, f6,
            df1, df2, df3, df4, df5, df6, al, be, ga, xiq[i], etq[i],
        );

        let vn = ddxi.cross(&ddet);
        let tangential = dfdxi * ddet - dfdet * ddxi;
        let grad_phi_sq = tangential.norm_squared() / (hs * hs) + dfdn_int * dfdn_int;

        // n̂ dA = vn * 0.5 * wq[i]  (n̂ = vn/hs, dA = 0.5*hs*wq[i])
        let n_hat_da = vn * (0.5 * wq[i]);
        let r = _xvec - body_centre;

        let dp = 0.5 * rho_f * grad_phi_sq;
        force += dp * n_hat_da;
        torque += dp * r.cross(&n_hat_da);
    }

    (force, torque)
}

/// Computes the −ρ ∂φ/∂t surface-pressure force and torque on one element.
/// ∂φ/∂t is approximated as (f_current − f_prev) / dt.
/// Force = +ρ ∮ (∂φ/∂t) n̂ dA  (positive sign from unsteady Bernoulli).
pub fn dphi_dt_force_element(
    k: usize,
    mint: usize,
    body_centre: &Vector3<f64>,
    rho_f: f64,
    f: &DVector<f64>,
    phi_dot: &DVector<f64>,
    p: &DMatrix<f64>,
    n: &DMatrix<usize>,
    vna: &DMatrix<f64>,
    alpha: &DVector<f64>,
    beta: &DVector<f64>,
    gamma: &DVector<f64>,
    xiq: &DVector<f64>,
    etq: &DVector<f64>,
    wq: &DVector<f64>,
) -> (Vector3<f64>, Vector3<f64>) {
    let mut force = Vector3::zeros();
    let mut torque = Vector3::zeros();

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
    // Pass the nodal ∂φ/∂t (phi_dot) as the "df" parameter; gradient_interp
    // returns dfdn_int = phi_dot interpolated to the quadrature point.
    let (pd1, pd2, pd3, pd4, pd5, pd6) = (
        phi_dot[i1],
        phi_dot[i2],
        phi_dot[i3],
        phi_dot[i4],
        phi_dot[i5],
        phi_dot[i6],
    );
    let (al, be, ga) = (alpha[k], beta[k], gamma[k]);

    for i in 0..mint {
        let (xvec, _v, _hs, _f_int, phi_dot_int, _dfdxi, _dfdet, ddxi, ddet) = gradient_interp(
            p1, p2, p3, p4, p5, p6, vna1, vna2, vna3, vna4, vna5, vna6, f1, f2, f3, f4, f5, f6,
            pd1, pd2, pd3, pd4, pd5, pd6, al, be, ga, xiq[i], etq[i],
        );

        let vn = ddxi.cross(&ddet);
        let dphi_dt_int = phi_dot_int;
        let n_hat_da = vn * (0.5 * wq[i]);
        let r = xvec - body_centre;

        force += rho_f * dphi_dt_int * n_hat_da;
        torque += rho_f * dphi_dt_int * r.cross(&n_hat_da);
    }

    (force, torque)
}

/// Lamb hydrodynamic impulse of one element:
///   L_lin = rho ∮ φ n̂ dA      L_ang = rho ∮ φ (r × n̂) dA      (r = x − body_centre)
/// Used to form the rotating-frame transport terms ω×L_lin (force) and
/// ω×L_ang (torque) that the exact impulse rate dL/dt carries for a rotating
/// body, but the per-element ∂φ/∂t integral in `dphi_dt_force_element` omits.
pub fn lamb_impulse_element(
    k: usize,
    mint: usize,
    body_centre: &Vector3<f64>,
    rho_f: f64,
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
) -> (Vector3<f64>, Vector3<f64>) {
    let mut l_lin = Vector3::zeros();
    let mut l_ang = Vector3::zeros();

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
    let (al, be, ga) = (alpha[k], beta[k], gamma[k]);

    for i in 0..mint {
        // Reuse gradient_interp for geometry + interpolated φ (f_int); the "df"
        // slot is unused here so f is passed again as a dummy.
        let (xvec, _v, _hs, f_int, _df_int, _dfdxi, _dfdet, ddxi, ddet) = gradient_interp(
            p1, p2, p3, p4, p5, p6, vna1, vna2, vna3, vna4, vna5, vna6, f1, f2, f3, f4, f5, f6, f1,
            f2, f3, f4, f5, f6, al, be, ga, xiq[i], etq[i],
        );

        let vn = ddxi.cross(&ddet);
        let n_hat_da = vn * (0.5 * wq[i]);
        let r = xvec - body_centre;

        l_lin += rho_f * f_int * n_hat_da;
        l_ang += rho_f * f_int * r.cross(&n_hat_da);
    }

    (l_lin, l_ang)
}

pub fn gradient_interp_outer_3d(
    _npts: usize,
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
) -> (f64, f64) {
    //Integrates the surface velocity of the flow over the surface of the ellipsoids
    let mut f0 = 0.0;
    let mut srf_area = 0.0;

    for k in 0..nelm {
        let (sdlp, arelm) = lsdlpp_3d_integral(
            p0, k, mint, f, dfdn, p, n, vna, alpha, beta, gamma, xiq, etq, wq,
        );
        srf_area += arelm;
        f0 += sdlp;
    }

    (srf_area, f0)
}

///Integrates single and double- layer potentials for given f and df/dn.
pub fn lsdlpp_3d_integral(
    x0: &Vector3<f64>,
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
) -> (f64, f64) {
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

        let (xvec, v, hs, f_int, dfdn_int) = lsdlpp_3d_interp(
            p1, p2, p3, p4, p5, p6, vna1, vna2, vna3, vna4, vna5, vna6, f1, f2, f3, f4, f5, f6,
            df1, df2, df3, df4, df5, df6, al, be, ga, xi, eta,
        );

        let (g, dg) = lgf_3d_fs(&xvec, &x0);

        let cf = 0.5 * hs * wq[i];

        area += cf;

        let r_int = -dfdn_int * g + f_int * v.dot(&dg);

        sdlp += r_int * cf;
    }

    (sdlp, area)
}

///Loops over all points and elements to calculate phi for given phi and d(phi)/dn on the surfaces.
pub fn lsdlpp_3d(
    _npts: usize,
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
) -> (f64, f64) {
    //Evaluates the value of the potential at given point p0 given distribution f, dfdn
    let mut f0 = 0.0;
    let mut srf_area = 0.0;

    for k in 0..nelm {
        let (sdlp, arelm) = lsdlpp_3d_integral(
            p0, k, mint, f, dfdn, p, n, vna, alpha, beta, gamma, xiq, etq, wq,
        );
        srf_area += arelm;
        f0 += sdlp;
    }

    (srf_area, f0)
}

///Integrates the kinetic energy of the fluid over the surface of an element
pub fn ke_3d_integral(
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
) -> (f64, f64) {
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

        let (_xvec, _v, hs, f_int, dfdn_int) = lsdlpp_3d_interp(
            p1, p2, p3, p4, p5, p6, vna1, vna2, vna3, vna4, vna5, vna6, f1, f2, f3, f4, f5, f6,
            df1, df2, df3, df4, df5, df6, al, be, ga, xi, eta,
        );

        let cf = 0.5 * hs * wq[i];

        area += cf;

        let r_int = -dfdn_int * f_int;

        sdlp += r_int * cf;
    }

    (sdlp, area)
}

///Calculates the total kinetic energy of the fluid from a given phi, d(phi)/dn on the surface.(Assumes \rho_f = 1)
pub fn ke_3d(
    _npts: usize,
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
) -> (f64, f64) {
    //Calculates the total kinetic energy of the fluid, doing a surface integral of phi*(dphi/dn)

    let mut srf_area = 0.0;
    let mut f0 = 0.0;

    for k in 0..nelm {
        let (sdlp, arelm) = ke_3d_integral(
            k, mint, f, dfdn, p, n, vna, alpha, beta, gamma, xiq, etq, wq,
        );

        srf_area += arelm;
        f0 += sdlp;
    }

    (srf_area, f0)
}
