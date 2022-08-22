use nalgebra::{DMatrix, DVector, Vector3, Vector6};


pub fn lgf_3d_fs(x :Vector3<f64>, x0 :Vector3<f64>) -> (f64, Vector3<f64>) {

    let pi = std::f64::consts::PI;
    let pi4 = pi * 4.0;

    let dx = Vector3::new(x[0] - x0[0], x[1] - x0[1], x[2] - x0[2]);
    let r = dx.norm();
    let g = 1.0 / (pi4 * r);

    let mut dg = Vector3::zeros();
    for i in 0..3 {
        dg[i] = - dx[i] / r;
    }

    (g, dg)
}
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

pub fn lslp_3d_integral(x0 :Vector3<f64>, k :usize,
                        mint :usize, f :&DVector<f64>,
                        p :&DMatrix<f64>, n :&DMatrix<usize>,
                        alpha :&DVector<f64>, beta :&DVector<f64>, gamma :&DVector<f64>,
                        xiq :&DVector<f64>, etq :&DVector<f64>, wq :&DVector<f64>) -> (f64, f64) {

    let mut area = 0.0;
    let mut slp = 0.0;

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

    let (f1, f2, f3, f4, f5, f6) = (f[i1], f[i2], f[i3], f[i4], f[i5], f[i6]);

    let (al, be, ga) = (alpha[k], beta[k], gamma[k]);

    //Loop over integration points
    for i in 0..mint {

        let (xi, eta) = (xiq[i], etq[i]);

        let (x, hs, fint) = lslp_3d_interp(p1, p2, p3, p4, p5, p6,
                                           f1, f2, f3, f4, f5, f6,
                                           al, be, ga, xi, eta);

        //Compute Greens fn
        let (g, _dg) = lgf_3d_fs(x, x0);

        //Apply triangle quadrature
        let cf = 0.5 * hs * wq[i];

        area += cf;
        slp += fint * g * cf;
    }
    (slp, area)
}

pub fn ldlp_3d_integral (x0 :Vector3<f64>, k :usize,
                         mint :usize, q :&DVector<f64>, q0 :f64,
                         p :&DMatrix<f64>, n :&DMatrix<usize>,
                         vna :&DMatrix<f64>, alpha :&DVector<f64>, beta :&DVector<f64>, gamma :&DVector<f64>,
                         xiq :&DVector<f64>, etq :&DVector<f64>, wq :&DVector<f64>) -> (f64, f64) {

    let mut area = 0.0;
    let mut ptl = 0.0;

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

    let (q1, q2, q3, q4, q5, q6) = (q[i1], q[i2], q[i3], q[i4], q[i5], q[i6]);

    let (al, be, ga) = (alpha[k], beta[k], gamma[k]);

    for i in 0..mint {

        let (xi, eta) = (xiq[i], etq[i]);

        let (xvec, v, hs, qint) = ldlp_3d_interp(p1, p2, p3, p4, p5, p6,
                                                 vna1, vna2, vna3, vna4, vna5, vna6,
                                                    q1, q2, q3, q4, q5, q6,
                                                    al, be, ga, xi, eta);

        let (_g, dg) = lgf_3d_fs(xvec, x0);

        let cf = 0.5 * hs * wq[i];

        area += cf;
        ptl += (qint - q0) * v.dot(&dg) * cf;
    }

    (ptl, area)
}

pub fn lslp_3d_integral_sing(ngl :usize,
                             p1 :Vector3<f64>, p2 :Vector3<f64>, p3 :Vector3<f64>,
                             f1 :f64, f2 :f64, f3 :f64,
                             zz :&DVector<f64>, ww :&DVector<f64>) -> (f64, f64) {
    /// Compute the laplace single-layer potential over a flat triangle with points p1-3
    /// Integrate in local polar coordinates with origin at p1

    //Compute triangle area and surface metric

    let mut vn = (p2 - p1).cross(&(p3 - p1));

    let area = 0.5 * vn.norm();
    let hs = 2.0 * area;

    let mut asm = 0.0;
    let mut slp = 0.0;
    let pi = std::f64::consts::PI;

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

            let (g, dg) = lgf_3d_fs(xvec, p1);

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

pub fn ldlp_3d(npts :usize, nelm :usize,
               mint :usize, q :&DVector<f64>,
               p :&DMatrix<f64>, n :&DMatrix<usize>, vna :&DMatrix<f64>,
               alpha :&DVector<f64>, beta :&DVector<f64>, gamma :&DVector<f64>,
               xiq :&DVector<f64>, etq :&DVector<f64>, wq :&DVector<f64>) -> DVector<f64> {

    let mut dlp = DVector::zeros(npts);
    let tol = 1e-8;

    for i in 0..npts {

        let p0 = Vector3::new(p[(i, 0)], p[(i, 1)], p[(i, 2)]);
        let q0 = q[i];


        let mut ptl = 0.0;

        //Compile the integrals over triangles

        for k in 0..nelm {

            let i1 = n[(k, 0)] - 1;
            let i2 = n[(k, 1)] - 1;
            let i3 = n[(k, 2)] - 1;
            let i4 = n[(k, 3)] - 1;
            let i5 = n[(k, 4)] - 1;
            let i6 = n[(k, 5)] - 1;

            let (q1, q2, q3, q4, q5, q6) = (q[i1], q[i2], q[i3], q[i4], q[i5], q[i6]);

            let test = Vector6::new(q1.abs(), q2.abs(), q3.abs(), q4.abs(), q5.abs(), q6.abs()).sum();

            if test > tol {

                let (pptl, _arelm) = ldlp_3d_integral(p0, k, mint, q, q0,
                                                     p, n, vna,
                                                     alpha, beta, gamma,
                                                     xiq, etq, wq);

                ptl += pptl;
            }

        }
        dlp[i] += ptl - 0.5 * q0;
    }
    dlp

}

pub fn lslp_3d(npts :usize, nelm :usize,
               mint :usize, nq :usize, f :&DVector<f64>,
               p :&DMatrix<f64>, n :&DMatrix<usize>, _vna :&DMatrix<f64>,
               alpha :&DVector<f64>, beta :&DVector<f64>, gamma :&DVector<f64>,
               xiq :&DVector<f64>, etq :&DVector<f64>, wq :&DVector<f64>,
               zz :&DVector<f64>, ww :&DVector<f64>) -> DVector<f64> {
    let mut slp = DVector::zeros(npts);
    let tol = 1e-8;

    for i in 0..npts {

        let p0 = Vector3::new(p[(i, 0)], p[(i, 1)], p[(i, 2)]);

        let mut srf_area = 0.0;
        let mut ptl = 0.0;

        for k in 0..nelm {

            let i1 = n[(k, 0)] - 1;
            let i2 = n[(k, 1)] - 1;
            let i3 = n[(k, 2)] - 1;
            let i4 = n[(k, 3)] - 1;
            let i5 = n[(k, 4)] - 1;
            let i6 = n[(k, 5)] - 1;

            let (f1, f2, f3, f4, f5, f6) = (f[i1], f[i2], f[i3], f[i4], f[i5], f[i6]);

            let test = Vector6::new(f1.abs(), f2.abs(), f3.abs(), f4.abs(), f5.abs(), f6.abs()).sum();

            if test > tol {
                if i == i1 {

                    let p1 = Vector3::new(p[(i1, 0)], p[(i1, 1)], p[(i1, 2)]);
                    let p2 = Vector3::new(p[(i2, 0)], p[(i2, 1)], p[(i2, 2)]);
                    let p3 = Vector3::new(p[(i3, 0)], p[(i3, 1)], p[(i3, 2)]);

                    let (f1, f2, f3) = (f[i1], f[i2], f[i3]);

                    let (pptl, arelm) = lslp_3d_integral_sing(nq, p1, p2, p3,
                                                              f1, f2, f3,
                                                              zz, ww);

                    ptl += pptl;
                    srf_area += arelm;

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
                } else if i == i4 {
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
                } else {

                    let (pptl, arelm) = lslp_3d_integral(p0, k, mint,
                                                         f, p, n,
                                                         alpha, beta, gamma,
                                                         xiq, etq, wq);

                    ptl += pptl;
                    srf_area += arelm;
                }
            }
        }
        slp[i] = ptl;
    }

    slp
}

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

    let vn = ddxi.cross(&ddet);
    let hs = vn.norm();

    // vn = vn.normalize();

    (xvec, v, hs, f, df)
}

pub fn lsdlpp_3d_integral(x0 :Vector3<f64>, k :usize,
                           mint :usize, f :&DVector<f64>, df :&DVector<f64>,
                           p :&DMatrix<f64>, n :&DMatrix<usize>, vna :&DMatrix<f64>,
                           alpha :&DVector<f64>, beta :&DVector<f64>, gamma :&DVector<f64>,
                           xiq :&DVector<f64>, etq :&DVector<f64>, wq :&DVector<f64>) -> (f64, f64) {

    let mut area = 0.0;
    let mut sdlp = 0.0;

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

    for i in 0..mint {

        let (xi, eta) = (xiq[i], etq[i]);

        let (xvec, v, hs, fint, dfdn_int) = lsdlpp_3d_interp(p1, p2, p3, p4, p5, p6,
                                                 vna1, vna2, vna3, vna4, vna5, vna6,
                                                 f1, f2, f3, f4, f5, f6,
                                                 df1, df2, df3, df4, df5, df6,
                                                 al, be, ga, xi, eta);

        let (g, dg) = lgf_3d_fs(xvec, x0);

        let cf = 0.5 * hs * wq[i];

        area += cf;

        let rint = -dfdn_int * g + fint * v.dot(&dg);

        sdlp += rint * cf;
    }

    (sdlp, area)
}

pub fn lsdlpp_3d(_npts :usize, nelm :usize, mint :usize,
                 f :&DVector<f64>, dfdn :&DVector<f64>,
                 p :&DMatrix<f64>, n :&DMatrix<usize>, vna :&DMatrix<f64>,
                 alpha :&DVector<f64>, beta :&DVector<f64>, gamma :&DVector<f64>,
                 xiq :&DVector<f64>, etq :&DVector<f64>, wq :&DVector<f64>,
                 p0 :Vector3<f64>) -> f64 {
    ///Evaluates the value of the potential at given point p0 given distribution f, dfdn
    let mut f0 = 0.0;

    for k in 0..nelm {


        let (sdlp, _arelm) = lsdlpp_3d_integral(p0, k, mint,
                                               f, dfdn,
                                               p, n, vna,
                                               alpha, beta, gamma,
                                               xiq, etq, wq);

        f0 += sdlp;
    }

    f0
}

