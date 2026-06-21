use multi_ellip::bem::exact_geometry::{
    ellipsoid_normal_body, ExactEllipsoidPatch, ExactEllipsoidSurface,
};
use multi_ellip::bem::geom::*;
use multi_ellip::bem::integ::*;
use multi_ellip::bem::potentials::dfdn_single;
use nalgebra::{DMatrix, DVector, UnitQuaternion, Vector3};
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

const REQ: f64 = 1.0;

#[derive(Clone, Copy)]
struct Mode {
    name: &'static str,
    exact: bool,
    exact_singular: bool,
}

struct Geometry {
    nelm: usize,
    npts: usize,
    p: DMatrix<f64>,
    n: DMatrix<usize>,
    vna: DMatrix<f64>,
    alpha: DVector<f64>,
    beta: DVector<f64>,
    gamma: DVector<f64>,
    exact_surface: Option<ExactEllipsoidSurface>,
}

struct TriQuad {
    mint: usize,
    xiq: DVector<f64>,
    etq: DVector<f64>,
    wq: DVector<f64>,
}

fn tri_quadrature() -> TriQuad {
    let (xiq, etq, wq) = gauss_trgl(13);
    TriQuad {
        mint: 13,
        xiq,
        etq,
        wq,
    }
}

fn build_geometry(ndiv: u32, mode: Mode, axes: &Vector3<f64>, quad: &TriQuad) -> Geometry {
    let centre = Vector3::zeros();
    let orient = UnitQuaternion::identity();
    let (nelm, npts, p, n) = ellip_gridder(ndiv, REQ, axes, &centre, &orient);

    if mode.exact {
        let (unit_nelm, unit_npts, unit_p, unit_n) = ellip_gridder_no_rotation(ndiv);
        debug_assert_eq!(unit_nelm, nelm);
        debug_assert_eq!(unit_npts, npts);

        let mut node_normals = DMatrix::zeros(npts, 3);
        for i in 0..npts {
            let body_point = Vector3::new(p[(i, 0)], p[(i, 1)], p[(i, 2)]);
            let normal = ellipsoid_normal_body(&body_point, axes);
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
            let (al, be, ga) = abc(
                unit_nodes[0],
                unit_nodes[1],
                unit_nodes[2],
                unit_nodes[3],
                unit_nodes[4],
                unit_nodes[5],
            );
            patches.push(ExactEllipsoidPatch::new(
                unit_nodes, *axes, centre, orient, al, be, ga,
            ));
        }

        let exact_surface = ExactEllipsoidSurface::new(patches, node_normals.clone());
        let (alpha, beta, gamma) = exact_surface.abc_vectors();
        Geometry {
            nelm,
            npts,
            p,
            n,
            vna: node_normals,
            alpha,
            beta,
            gamma,
            exact_surface: Some(exact_surface),
        }
    } else {
        let (alpha, beta, gamma) = abc_vec(nelm, &p, &n);
        let (vna, _vlm, _sa) = elm_geom(
            npts, nelm, quad.mint, &p, &n, &alpha, &beta, &gamma, &quad.xiq, &quad.etq, &quad.wq,
        );
        Geometry {
            nelm,
            npts,
            p,
            n,
            vna,
            alpha,
            beta,
            gamma,
            exact_surface: None,
        }
    }
}

fn rel_norm(a: &DVector<f64>, b: &DVector<f64>) -> f64 {
    (a - b).norm() / b.norm().max(1.0e-300)
}

fn centered_rel_norm(a: &DVector<f64>, b: &DVector<f64>) -> f64 {
    let mut da = a.clone();
    let mut db = b.clone();
    let ma = da.iter().sum::<f64>() / da.len() as f64;
    let mb = db.iter().sum::<f64>() / db.len() as f64;
    for v in da.iter_mut() {
        *v -= ma;
    }
    for v in db.iter_mut() {
        *v -= mb;
    }
    rel_norm(&da, &db)
}

fn direct_matrix_diff(
    geom: &Geometry,
    quad: &TriQuad,
    zz: &DVector<f64>,
    ww: &DVector<f64>,
) -> f64 {
    let fast = ldlp_3d_assemble(
        geom.npts,
        geom.nelm,
        quad.mint,
        zz.len(),
        &geom.p,
        &geom.n,
        &geom.vna,
        &geom.alpha,
        &geom.beta,
        &geom.gamma,
        &quad.xiq,
        &quad.etq,
        &quad.wq,
        zz,
        ww,
    );

    let mut slow = DMatrix::zeros(geom.npts, geom.npts);
    for j in 0..geom.npts {
        let mut q = DVector::zeros(geom.npts);
        q[j] = 1.0;
        let col = ldlp_3d(
            geom.npts,
            geom.nelm,
            quad.mint,
            zz.len(),
            &q,
            &geom.p,
            &geom.n,
            &geom.vna,
            &geom.alpha,
            &geom.beta,
            &geom.gamma,
            &quad.xiq,
            &quad.etq,
            &quad.wq,
            zz,
            ww,
        );
        for i in 0..geom.npts {
            slow[(i, j)] = col[i];
        }
    }

    (&fast - &slow).norm() / slow.norm().max(1.0e-300)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let out_path = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("scratch_bem_operator_probe.csv"));
    let max_ndiv: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);

    let quad = tri_quadrature();
    let axes = Vector3::new(1.0, 1.0, 1.0);
    let modes = [
        Mode {
            name: "default",
            exact: false,
            exact_singular: false,
        },
        Mode {
            name: "exact",
            exact: true,
            exact_singular: false,
        },
        Mode {
            name: "exact_singular",
            exact: true,
            exact_singular: true,
        },
    ];

    let file = File::create(&out_path).expect("failed to create output csv");
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "mode,ndiv,npts,nelm,matrix_diff_rel,rhs_rel_error,a_phi_rel_error,solved_phi_rel_error"
    )
    .unwrap();

    for ndiv in 1..=max_ndiv {
        for mode in modes {
            let nq = if mode.exact_singular { 24 } else { 12 };
            let (zz, ww) = gauss_leg(nq);
            let geom = build_geometry(ndiv, mode, &axes, &quad);

            let u = Vector3::new(1.0, 0.0, 0.0);
            let omega = Vector3::zeros();
            let dfdn = dfdn_single(&Vector3::zeros(), &u, &omega, geom.npts, &geom.p, &geom.vna);
            let rhs = lslp_3d_with_geom(
                geom.npts,
                geom.nelm,
                quad.mint,
                nq,
                &dfdn,
                &geom.p,
                &geom.n,
                &geom.vna,
                &geom.alpha,
                &geom.beta,
                &geom.gamma,
                &quad.xiq,
                &quad.etq,
                &quad.wq,
                &zz,
                &ww,
                geom.exact_surface.as_ref(),
                mode.exact_singular,
            );
            let amat = ldlp_3d_assemble_with_geom(
                geom.npts,
                geom.nelm,
                quad.mint,
                nq,
                &geom.p,
                &geom.n,
                &geom.vna,
                &geom.alpha,
                &geom.beta,
                &geom.gamma,
                &quad.xiq,
                &quad.etq,
                &quad.wq,
                &zz,
                &ww,
                geom.exact_surface.as_ref(),
                mode.exact_singular,
            );

            let mut analytic_phi = DVector::zeros(geom.npts);
            let mut analytic_rhs = DVector::zeros(geom.npts);
            for i in 0..geom.npts {
                let x = geom.p[(i, 0)];
                analytic_phi[i] = -0.5 * x;
                analytic_rhs[i] = x / 3.0;
            }

            let a_phi = &amat * &analytic_phi;
            let solved_phi = amat.lu().solve(&rhs).expect("BEM solve failed");
            let matrix_diff_rel = if mode.name == "default" && ndiv <= 2 {
                direct_matrix_diff(&geom, &quad, &zz, &ww)
            } else {
                f64::NAN
            };

            writeln!(
                writer,
                "{},{},{},{},{:.16e},{:.16e},{:.16e},{:.16e}",
                mode.name,
                ndiv,
                geom.npts,
                geom.nelm,
                matrix_diff_rel,
                rel_norm(&rhs, &analytic_rhs),
                rel_norm(&a_phi, &analytic_rhs),
                centered_rel_norm(&solved_phi, &analytic_phi),
            )
            .unwrap();
        }
    }

    println!("wrote {}", out_path.display());
}
