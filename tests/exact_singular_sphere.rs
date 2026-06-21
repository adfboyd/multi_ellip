use multi_ellip::bem::exact_geometry::{
    ellipsoid_normal_body, ExactEllipsoidPatch, ExactEllipsoidSurface,
};
use multi_ellip::bem::geom::{
    abc, ellip_gridder, ellip_gridder_no_rotation, gauss_leg, gauss_trgl,
};
use multi_ellip::bem::integ::{ldlp_3d_assemble_with_geom, lslp_3d_with_geom};
use multi_ellip::bem::potentials::dfdn_single;
use nalgebra::{DMatrix, DVector, UnitQuaternion, Vector3};

fn rel_norm(a: &DVector<f64>, b: &DVector<f64>) -> f64 {
    (a - b).norm() / b.norm().max(1.0e-300)
}

fn centered_rel_norm(a: &DVector<f64>, b: &DVector<f64>) -> f64 {
    let mut ac = a.clone();
    let mut bc = b.clone();
    let am = ac.iter().sum::<f64>() / ac.len() as f64;
    let bm = bc.iter().sum::<f64>() / bc.len() as f64;
    for v in ac.iter_mut() {
        *v -= am;
    }
    for v in bc.iter_mut() {
        *v -= bm;
    }
    rel_norm(&ac, &bc)
}

#[test]
fn exact_singular_sphere_translation_matches_analytic_solution() {
    let ndiv = 1;
    let axes = Vector3::new(1.0, 1.0, 1.0);
    let centre = Vector3::zeros();
    let orient = UnitQuaternion::identity();
    let (nelm, npts, p, n) = ellip_gridder(ndiv, 1.0, &axes, &centre, &orient);
    let (unit_nelm, unit_npts, unit_p, unit_n) = ellip_gridder_no_rotation(ndiv);
    assert_eq!(unit_nelm, nelm);
    assert_eq!(unit_npts, npts);

    let mut node_normals = DMatrix::zeros(npts, 3);
    for i in 0..npts {
        let body_point = Vector3::new(p[(i, 0)], p[(i, 1)], p[(i, 2)]);
        let normal = ellipsoid_normal_body(&body_point, &axes);
        for c in 0..3 {
            node_normals[(i, c)] = normal[c];
        }
    }

    let mut patches = Vec::with_capacity(nelm);
    for k in 0..nelm {
        let unit_nodes: [Vector3<f64>; 6] = std::array::from_fn(|m| {
            let idx = unit_n[(k, m)];
            Vector3::new(unit_p[(idx, 0)], unit_p[(idx, 1)], unit_p[(idx, 2)])
        });
        let (alpha, beta, gamma) = abc(
            unit_nodes[0],
            unit_nodes[1],
            unit_nodes[2],
            unit_nodes[3],
            unit_nodes[4],
            unit_nodes[5],
        );
        patches.push(ExactEllipsoidPatch::new(
            unit_nodes, axes, centre, orient, alpha, beta, gamma,
        ));
    }

    let exact_surface = ExactEllipsoidSurface::new(patches, node_normals.clone());
    let (alpha, beta, gamma) = exact_surface.abc_vectors();
    let (xiq, etq, wq) = gauss_trgl(13);
    let (zz, ww) = gauss_leg(24);
    let dfdn = dfdn_single(
        &Vector3::zeros(),
        &Vector3::new(1.0, 0.0, 0.0),
        &Vector3::zeros(),
        npts,
        &p,
        &node_normals,
    );

    let rhs = lslp_3d_with_geom(
        npts,
        nelm,
        13,
        24,
        &dfdn,
        &p,
        &n,
        &node_normals,
        &alpha,
        &beta,
        &gamma,
        &xiq,
        &etq,
        &wq,
        &zz,
        &ww,
        Some(&exact_surface),
        true,
    );
    let amat = ldlp_3d_assemble_with_geom(
        npts,
        nelm,
        13,
        24,
        &p,
        &n,
        &node_normals,
        &alpha,
        &beta,
        &gamma,
        &xiq,
        &etq,
        &wq,
        &zz,
        &ww,
        Some(&exact_surface),
        true,
    );

    let mut analytic_phi = DVector::zeros(npts);
    let mut analytic_rhs = DVector::zeros(npts);
    for i in 0..npts {
        let x = p[(i, 0)];
        analytic_phi[i] = -0.5 * x;
        analytic_rhs[i] = x / 3.0;
    }

    let a_phi = &amat * &analytic_phi;
    let solved_phi = amat.lu().solve(&rhs).expect("BEM solve failed");

    assert!(rel_norm(&rhs, &analytic_rhs) < 5.0e-3);
    assert!(rel_norm(&a_phi, &analytic_rhs) < 2.0e-3);
    assert!(centered_rel_norm(&solved_phi, &analytic_phi) < 5.0e-3);
}
