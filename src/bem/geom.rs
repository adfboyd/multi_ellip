use nalgebra as na;
use nalgebra::{DMatrix, DVector, Dynamic, OMatrix, Vector3, Vector6};
use nalgebra::{U1};

pub fn ellip_gridder(ndiv : u32, req :f64,
                     shape :Vector3<f64>, centre :Vector3<f64>, orient : na::UnitQuaternion<f64>)
    -> (usize, usize, DMatrix<f64>, DMatrix<usize>)
    {

        let mut nelm :usize = 8;
        let eps = 1e-8;
        let nelm_end = nelm * 4_usize.pow(ndiv);
        let npts_end  = nelm_end * 2 + 2;



        let (a, b, c) = (shape[0], shape[1], shape[2]);
        let (boa, coa) = (b/a, c/a);

        type MatrixU3 = DMatrix<usize>;
        type MatrixF6 = DMatrix<f64>;

        let mut x = MatrixF6::zeros(nelm_end, 6);
        let mut y = MatrixF6::zeros(nelm_end, 6);
        let mut z = MatrixF6::zeros(nelm_end, 6);

        let mut xn = MatrixF6::zeros(nelm_end, 6);
        let mut yn = MatrixF6::zeros(nelm_end, 6);
        let mut zn = MatrixF6::zeros(nelm_end, 6);

        let mut p = MatrixF6::zeros(npts_end, 3);
        let mut n = MatrixU3::zeros(nelm_end, 6);


    //     Set initial 8 elements (ie faces)

        //Element 1
        (x[(0,0)], y[(0,0)], z[(0,0)]) = (0.0, 0.0, 1.0);
        (x[(0,1)], y[(0,1)], z[(0,1)]) = (1.0, 0.0, 0.0);
        (x[(0,2)], y[(0,2)], z[(0,2)]) = (0.0, 1.0, 0.0);
        //Element 5
        (x[(4,0)], y[(4,0)], z[(4,0)]) = (1.0, 0.0, 0.0);
        (x[(4,1)], y[(4,1)], z[(4,1)]) = (0.0, 0.0, -1.0);
        (x[(4,2)], y[(4,2)], z[(4,2)]) = (0.0, 1.0, 0.0);
        //Element 6
        (x[(5,0)], y[(5,0)], z[(5,0)]) = (0.0, 0.0, -1.0);
        (x[(5,1)], y[(5,1)], z[(5,1)]) = (-1.0, 0.0, 0.0);
        (x[(5,2)], y[(5,2)], z[(5,2)]) = (0.0, 1.0, 0.0);
        //Element 2
        (x[(1,0)], y[(1,0)], z[(1,0)]) = (-1.0, 0.0, 0.0);
        (x[(1,1)], y[(1,1)], z[(1,1)]) = (0.0, 0.0, 1.0);
        (x[(1,2)], y[(1,2)], z[(1,2)]) = (0.0, 1.0, 0.0);
        //Corner points - lower half of xz plane
        //Element 4
        (x[(3,0)], y[(3,0)], z[(3,0)]) = (0.0, 0.0, 1.0);
        (x[(3,1)], y[(3,1)], z[(3,1)]) = (0.0, -1.0, 0.0);
        (x[(3,2)], y[(3,2)], z[(3,2)]) = (1.0, 0.0, 0.0);
        //Element 8
        (x[(7,0)], y[(7,0)], z[(7,0)]) = (1.0, 0.0, 0.0);
        (x[(7,1)], y[(7,1)], z[(7,1)]) = (0.0, -1.0, 0.0);
        (x[(7,2)], y[(7,2)], z[(7,2)]) = (0.0, 0.0, -1.0);
        //Element 7
        (x[(6,0)], y[(6,0)], z[(6,0)]) = (0.0, 0.0, -1.0);
        (x[(6,1)], y[(6,1)], z[(6,1)]) = (0.0, -1.0, 0.0);
        (x[(6,2)], y[(6,2)], z[(6,2)]) = (-1.0, 0.0, 0.0);
        //Element 3
        (x[(2,0)], y[(2,0)], z[(2,0)]) = (-1.0, 0.0, 0.0);
        (x[(2,1)], y[(2,1)], z[(2,1)]) = (0.0, -1.0, 0.0);
        (x[(2,2)], y[(2,2)], z[(2,2)]) = (0.0, 0.0, 1.0);

        //Compute midpoints of sides
        for i in 0..nelm as usize {

            x[(i, 3)] = 0.5 * (x[(i, 0)] + x[(i, 1)]);
            y[(i, 3)] = 0.5 * (y[(i, 0)] + y[(i, 1)]);
            z[(i, 3)] = 0.5 * (z[(i, 0)] + z[(i, 1)]);

            x[(i, 4)] = 0.5 * (x[(i, 1)] + x[(i, 2)]);
            y[(i, 4)] = 0.5 * (y[(i, 1)] + y[(i, 2)]);
            z[(i, 4)] = 0.5 * (z[(i, 1)] + z[(i, 2)]);

            x[(i, 5)] = 0.5 * (x[(i, 2)] + x[(i, 0)]);
            y[(i, 5)] = 0.5 * (y[(i, 2)] + y[(i, 0)]);
            z[(i, 5)] = 0.5 * (z[(i, 2)] + z[(i, 0)]);

        }

        //Compute node coords on each element for discretization
        // levels 1 through ndiv

        for _ in 0..ndiv {

            let mut num :usize = 0;

            for j in 0..nelm {

                //Assign corner points to sub-elements
                //1st sub-element
                (xn[(num, 0)], yn[(num, 0)], zn[(num, 0)]) = (x[(j, 0)], y[(j, 0)], z[(j, 0)]);
                (xn[(num, 1)], yn[(num, 1)], zn[(num, 1)]) = (x[(j, 3)], y[(j, 3)], z[(j, 3)]);
                (xn[(num, 2)], yn[(num, 2)], zn[(num, 2)]) = (x[(j, 5)], y[(j, 5)], z[(j, 5)]);

                xn[(num, 3)] = 0.5 * (xn[(num, 0)] + xn[(num, 1)]);
                yn[(num, 3)] = 0.5 * (yn[(num, 0)] + yn[(num, 1)]);
                zn[(num, 3)] = 0.5 * (zn[(num, 0)] + zn[(num, 1)]);

                xn[(num, 4)] = 0.5 * (xn[(num, 1)] + xn[(num, 2)]);
                yn[(num, 4)] = 0.5 * (yn[(num, 1)] + yn[(num, 2)]);
                zn[(num, 4)] = 0.5 * (zn[(num, 1)] + zn[(num, 2)]);

                xn[(num, 5)] = 0.5 * (xn[(num, 2)] + xn[(num, 0)]);
                yn[(num, 5)] = 0.5 * (yn[(num, 2)] + yn[(num, 0)]);
                zn[(num, 5)] = 0.5 * (zn[(num, 2)] + zn[(num, 0)]);

                num += 1;

                (xn[(num, 0)], yn[(num, 0)], zn[(num, 0)]) = (x[(j, 3)], y[(j, 3)], z[(j, 3)]);
                (xn[(num, 1)], yn[(num, 1)], zn[(num, 1)]) = (x[(j, 1)], y[(j, 1)], z[(j, 1)]);
                (xn[(num, 2)], yn[(num, 2)], zn[(num, 2)]) = (x[(j, 4)], y[(j, 4)], z[(j, 4)]);

                xn[(num, 3)] = 0.5 * (xn[(num, 0)] + xn[(num, 1)]);
                yn[(num, 3)] = 0.5 * (yn[(num, 0)] + yn[(num, 1)]);
                zn[(num, 3)] = 0.5 * (zn[(num, 0)] + zn[(num, 1)]);

                xn[(num, 4)] = 0.5 * (xn[(num, 1)] + xn[(num, 2)]);
                yn[(num, 4)] = 0.5 * (yn[(num, 1)] + yn[(num, 2)]);
                zn[(num, 4)] = 0.5 * (zn[(num, 1)] + zn[(num, 2)]);

                xn[(num, 5)] = 0.5 * (xn[(num, 2)] + xn[(num, 0)]);
                yn[(num, 5)] = 0.5 * (yn[(num, 2)] + yn[(num, 0)]);
                zn[(num, 5)] = 0.5 * (zn[(num, 2)] + zn[(num, 0)]);

                num += 1;

                (xn[(num, 0)], yn[(num, 0)], zn[(num, 0)]) = (x[(j, 5)], y[(j, 5)], z[(j, 5)]);
                (xn[(num, 1)], yn[(num, 1)], zn[(num, 1)]) = (x[(j, 4)], y[(j, 4)], z[(j, 4)]);
                (xn[(num, 2)], yn[(num, 2)], zn[(num, 2)]) = (x[(j, 2)], y[(j, 2)], z[(j, 2)]);

                xn[(num, 3)] = 0.5 * (xn[(num, 0)] + xn[(num, 1)]);
                yn[(num, 3)] = 0.5 * (yn[(num, 0)] + yn[(num, 1)]);
                zn[(num, 3)] = 0.5 * (zn[(num, 0)] + zn[(num, 1)]);

                xn[(num, 4)] = 0.5 * (xn[(num, 1)] + xn[(num, 2)]);
                yn[(num, 4)] = 0.5 * (yn[(num, 1)] + yn[(num, 2)]);
                zn[(num, 4)] = 0.5 * (zn[(num, 1)] + zn[(num, 2)]);

                xn[(num, 5)] = 0.5 * (xn[(num, 2)] + xn[(num, 0)]);
                yn[(num, 5)] = 0.5 * (yn[(num, 2)] + yn[(num, 0)]);
                zn[(num, 5)] = 0.5 * (zn[(num, 2)] + zn[(num, 0)]);

                num += 1;

                (xn[(num, 0)], yn[(num, 0)], zn[(num, 0)]) = (x[(j, 3)], y[(j, 3)], z[(j, 3)]);
                (xn[(num, 1)], yn[(num, 1)], zn[(num, 1)]) = (x[(j, 4)], y[(j, 4)], z[(j, 4)]);
                (xn[(num, 2)], yn[(num, 2)], zn[(num, 2)]) = (x[(j, 5)], y[(j, 5)], z[(j, 5)]);

                xn[(num, 3)] = 0.5 * (xn[(num, 0)] + xn[(num, 1)]);
                yn[(num, 3)] = 0.5 * (yn[(num, 0)] + yn[(num, 1)]);
                zn[(num, 3)] = 0.5 * (zn[(num, 0)] + zn[(num, 1)]);

                xn[(num, 4)] = 0.5 * (xn[(num, 1)] + xn[(num, 2)]);
                yn[(num, 4)] = 0.5 * (yn[(num, 1)] + yn[(num, 2)]);
                zn[(num, 4)] = 0.5 * (zn[(num, 1)] + zn[(num, 2)]);

                xn[(num, 5)] = 0.5 * (xn[(num, 2)] + xn[(num, 0)]);
                yn[(num, 5)] = 0.5 * (yn[(num, 2)] + yn[(num, 0)]);
                zn[(num, 5)] = 0.5 * (zn[(num, 2)] + zn[(num, 0)]);

                num += 1;

            }
            nelm *= 4;
            //Rename new points and place them in master list
            for k in 0..nelm {
                for l in 0..6 {

                    (x[(k, l)], y[(k, l)], z[(k, l)]) = (xn[(k, l)], yn[(k, l)], zn[(k, l)]);
                    (xn[(k, l)], yn[(k, l)], zn[(k, l)]) = (0.0, 0.0, 0.0); //Just in case

                    let rad = (x[(k, l)].powi(2) + y[(k, l)].powi(2) + z[(k, l)].powi(2)).sqrt();

                    (x[(k, l)], y[(k, l)], z[(k, l)]) = (x[(k, l)] / rad, y[(k, l)] / rad, z[(k, l)] / rad)
                }
            }
        }

        ///Generate a list of global nodes by looping over all elements
        /// and adding in new ones
        /// n[(i,j)] is jth node of i+1th element
        /// First size nodes of first element are entered manually

        for i in 0..6 {
            (p[(i, 0)], p[(i, 1)], p[(i, 2)]) = (x[(0, i)], y[(0, i)], z[(0, i)]);

            n[(0, i)] = i + 1;
        }

        let mut npts = 6;

        for i in 0..nelm {
            for j in 0..6 {
                let mut iflag = true; //Is current point new?

                for k in 0..npts {
                    if (x[(i,j)] - p[(k,0)]).abs() < eps {
                        if (y[(i,j)] - p[(k,1)]).abs() < eps {
                            if (z[(i,j)] - p[(k,2)]).abs() < eps {
                                iflag = false;
                                n[(i, j)] = k + 1;
                            }
                        }
                    }
                }
                if iflag {
                    (p[(npts, 0)], p[(npts, 1)], p[(npts, 2)]) = (x[(i, j)], y[(i, j)], z[(i, j)]);
                    n[(i, j)] = npts + 1;
                    npts += 1;

                }
            }
        }

        let scale = req / (boa * coa).powf(1.0/3.0);

        for i in 0..npts {
            //Reshape to ellipsoid
            p[(i, 0)] = scale * p[(i, 0)];
            p[(i, 1)] = scale * p[(i, 1)] * boa;
            p[(i, 2)] = scale * p[(i, 2)] * coa;

            // p.set_row(i, )
            let prow = na::Vector3::new(p.row(i)[0], p.row(i)[1], p.row(i)[2]);
            let p1 = orient.transform_vector(&prow);
            for j in 0..3 {
                p[(i, j)] = p1[j]
            }
            //Translate from centre
            for j in 0..3 {
                p[(i, j)] += centre[j];
            }
        }
        (nelm, npts, p, n)
    }

pub fn combiner(nelm1 :usize, nelm2 :usize, npts1 :usize, npts2 :usize,
                p1 :&DMatrix<f64>, p2 :&DMatrix<f64>,
                n1 :&DMatrix<usize>, n2 :&DMatrix<usize>) -> (
        usize, usize, DMatrix<f64>, DMatrix<usize>
) {
    let nelm = nelm1 + nelm2;
    let npts = npts1 + npts2;


    let mut n = DMatrix::zeros(nelm, 6);
    let mut p = DMatrix::zeros(npts, 3);

    for i in 0..nelm1 {
        for j in 0..6 {
            n[(i,j)] = n1[(i, j)];
        }
    }

    for i in 0..nelm2 {
        for j in 0..6 {
            n[(i + nelm1, j)] = n2[(i, j)] + npts1;
        }
    }

    for i in 0..npts1 {
        for j in 0..3 {
            p[(i, j)] = p1[(i, j)];
        }
    }

    for i in 0..npts2 {
        for j in 0..3 {
            p[(i + npts1, j)] = p2[(i, j)];
        }
    }

    (nelm, npts, p, n)
}

pub fn gauss_leg(nq:usize) -> (DVector<f64>, DVector<f64>) {

    let nq_a = if (nq != 3) & (nq != 6) {
        println!("Unsupported quadrature points number, taking nq = 6");
        6
    }
    else {
        nq
    };


    let mut z = DVector::zeros(nq_a);
    let mut w = DVector::zeros(nq_a);

    if nq_a == 3 {
        (z[0], z[1], z[2]) = (-0.77459666924148337703, 0.0, -0.77459666924148337703);
        (w[0], w[1], w[2]) = (5.0/9.0, 8.0/9.0, 5.0/9.0);
    }
    else if nq_a == 6 {
        (z[0], z[1], z[2]) = (-0.932469514203152, -0.661209386466265, -0.238619186083197);
        (z[3], z[4], z[5]) = (-z[2], -z[1], -z[0]);

        (w[0], w[1], w[2]) = (0.171324492379170, 0.360761573048139, 0.467913934572691);
        (w[3], w[4], w[5]) = (w[2], w[1], w[0]);
    }

    (z, w)
}

pub fn gauss_trgl(mint :usize) -> (DVector<f64>, DVector<f64>, DVector<f64>) {

    let mint_a:usize = if mint == 7 {
        mint
    }
    else {
        println!("Only implemented for mint = 7 at the moment, proceeding as such");
        7};

    let mut xi = DVector::zeros(mint_a);
    let mut eta = DVector::zeros(mint_a);
    let mut w = DVector::zeros(mint_a);

    if mint == 7 {

        let al = 0.797426958353087;
        let be = 0.470142064105115;
        let ga = 0.059715871789770;
        let de = 0.101286507323456;
        let o1 = 0.125939180544827;
        let o2 = 0.132394152788506;

        (xi[0], xi[1], xi[2], xi[3], xi[4], xi[5], xi[6]) = (de, al, de, be, ga, be, 1.0/3.0);
        (eta[0], eta[1], eta[2], eta[3], eta[4], eta[5], eta[6]) = (de, de, al, be, be, ga, 1.0/3.0);
        (w[0], w[1], w[2], w[3], w[4], w[5], w[6]) = (o1, o1, o1, o2, o2, o2, 0.225);
    }

    (xi, eta, w)
}

pub fn abc(p1 :Vector3<f64>,
           p2 :Vector3<f64>,
           p3 :Vector3<f64>,
           p4 :Vector3<f64>,
           p5 :Vector3<f64>,
           p6 :Vector3<f64>) -> (f64, f64, f64) {

    let d42 = (p4 - p2).norm();
    let d41 = (p4 - p1).norm();
    let d63 = (p6 - p3).norm();
    let d61 = (p6 - p1).norm();
    let d52 = (p5 - p2).norm();
    let d53 = (p5 - p3).norm();

    let al = 1.0 / (1.0 + d42 / d41);
    let be = 1.0 / (1.0 + d63 / d61);
    let ga = 1.0 / (1.0 + d52 / d53);

    (al, be, ga)
}
pub fn abc_vec(nelm :usize,
               p :&DMatrix<f64>,
               n :&DMatrix<usize>) ->
                    (DVector<f64>,
                     DVector<f64>,
                     DVector<f64>)
{
    let mut alpha = DVector::zeros(nelm);
    let mut beta = DVector::zeros(nelm);
    let mut gamma = DVector::zeros(nelm);

    for k in 0..nelm {

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

        let (al, be, ga) = abc(p1, p2, p3, p4, p5, p6);

        (alpha[k], beta[k], gamma[k]) = (al, be, ga);
    }
    (alpha, beta, gamma)
}

pub fn interp_p(p1 :Vector3<f64>,
                p2 :Vector3<f64>,
                p3 :Vector3<f64>,
                p4 :Vector3<f64>,
                p5 :Vector3<f64>,
                p6 :Vector3<f64>,
                al :f64, be :f64, ga :f64,
                xi :f64, eta :f64) ->
                                                  (Vector3<f64>, Vector3<f64>, f64) {

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

    vn = vn.normalize();

    (xvec, vn, hs)
}

pub fn elm_geom(npts :usize, nelm :usize, mint :usize,
                p :&DMatrix<f64>, n :&DMatrix<usize>,
                alpha :&DVector<f64>, beta :&DVector<f64>, gamma :&DVector<f64>,
                xiq :&DVector<f64>, etq :&DVector<f64>, wq :&DVector<f64>) ->
                       (DMatrix<f64>, f64, f64) {

    let mut area = 0.0;
    let mut vlm = 0.0;
    // let mut cx = 0.0;
    // let mut cy = 0.0;
    // let mut cz = 0.0;

    let mut vna = DMatrix::zeros(npts, 3);
    // let mut v = na::DMatrix::zeros(6, 3);
    let mut itally = DVector::zeros(npts);

    let mut arel :OMatrix<f64, Dynamic, U1> = DVector::zeros(nelm);
    // let mut xmom :OMatrix<f64, Dynamic, U1> = DVector::zeros(nelm);
    // let mut ymom :OMatrix<f64, Dynamic, U1> = DVector::zeros(nelm);
    // let mut zmom :OMatrix<f64, Dynamic, U1> = DVector::zeros(nelm);

    for k in 0..nelm {

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

        let (al, be, ga) = (alpha[k], beta[k], gamma[k]);
        let (_alc, _bec, gac) = (1.0 - al, 1.0 - be, 1.0 - ga);

        for i in 0..mint {

            let xi = xiq[i];
            let eta = etq[i];

            let (xvec, vn, hs) = interp_p(p1, p2, p3, p4, p5, p6,
                                                                     al, be, ga, xi, eta);

            let cf = hs * wq[i];

            arel[k] += cf;

            // xmom[k] += cf * xvec[0];
            // ymom[k] += cf * xvec[1];
            // zmom[k] += cf * xvec[2];

            vlm += xvec.dot(&vn) * cf;
        }

        arel[k] *= 0.5;
        // xmom[k] *= 0.5;
        // ymom[k] *= 0.5;
        // zmom[k] *= 0.5;

        area += arel[k];

        // cx += xmom[k];
        // cy += ymom[k];
        // cz += zmom[k];

        //Node triangle coordinates
        let xxi = Vector6::new(0.0, 1.0, 0.0, al, ga, 0.0);
        let eet = Vector6::new(0.0, 0.0, 1.0, 0.0, gac, be);

        //Loop over triangle coordinates of the nodes
        for i in 0..6 {
            let (xi, eta) = (xxi[i], eet[i]);

            let (_xvec, vn, _hs) = interp_p(p1, p2, p3, p4, p5, p6,
                                          al, be, ga,
                                          xi, eta);

            let m = n[(k, i)] - 1;

            for j in 0..3 {
                vna[(m, j)] = vn[j];
            }

            itally[m] += 1.0;
        }
    }

    //Average normal vector at the nodes and then normalize to make its length equal to unity
    for i in 0..npts {

        let par :f64= itally[i];

        for j in 0..3 {
            vna[(i, j)] = vna[(i, j)] / par;
        }

        let par = vna.row(i).norm();

        for j in 0..3 {
            vna[(i, j)] = vna[(i, j)] / par;
        }
    }

    //Finally compute surface area and volume

    // (cx, cy, cz) = (cx / area, cy / area, cx / area);
    vlm = vlm / 6.0;

    (vna, vlm, area)


}