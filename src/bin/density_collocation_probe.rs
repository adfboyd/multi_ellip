use multi_ellip::bem::exact_geometry::{
    ellipsoid_normal_body, semi_axes_from_solver_shape, ExactEllipsoidPatch, ExactEllipsoidSurface,
};
use multi_ellip::bem::geom::*;
use multi_ellip::bem::integ::*;
use multi_ellip::bem::potentials::dfdn_single;
use nalgebra::{DMatrix, DVector, UnitQuaternion, Vector3};
use std::env;
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

const RHO_F: f64 = 1.0;
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

fn shape(kind: &str) -> Vector3<f64> {
    match kind {
        "sphere" => Vector3::new(REQ, REQ, REQ),
        "ellipsoid" => {
            let ratios = Vector3::new(1.0, 0.8, 0.6);
            semi_axes_from_solver_shape(REQ, &ratios)
        }
        other => panic!("unknown shape kind '{other}', expected 'ellipsoid' or 'sphere'"),
    }
}

fn analytic_added_inertia(axes: &Vector3<f64>) -> [f64; 6] {
    let v_s = 4.0 / 3.0 * PI * axes[0] * axes[1] * axes[2];
    let (alpha, beta, gamma) = analytic_shape_factors(axes);
    let mf = Vector3::new(
        v_s * RHO_F * alpha / (2.0 - alpha),
        v_s * RHO_F * beta / (2.0 - beta),
        v_s * RHO_F * gamma / (2.0 - gamma),
    );

    let jf = if (axes[0] - axes[1]).abs() < 1.0e-12 && (axes[0] - axes[2]).abs() < 1.0e-12 {
        Vector3::zeros()
    } else {
        let (a, b, c) = (axes[0], axes[1], axes[2]);
        let e1 = ((b.powi(2) - c.powi(2)).powi(2) * (gamma - beta))
            / (2.0 * (b.powi(2) - c.powi(2)) + (beta - gamma) * (b.powi(2) + c.powi(2)));
        let e2 = ((a.powi(2) - c.powi(2)).powi(2) * (gamma - alpha))
            / (2.0 * (a.powi(2) - c.powi(2)) + (alpha - gamma) * (a.powi(2) + c.powi(2)));
        let e3 = ((a.powi(2) - b.powi(2)).powi(2) * (beta - alpha))
            / (2.0 * (a.powi(2) - b.powi(2)) + (alpha - beta) * (a.powi(2) + b.powi(2)));
        0.2 * v_s * RHO_F * Vector3::new(e1, e2, e3)
    };

    [mf[0], mf[1], mf[2], jf[0], jf[1], jf[2]]
}

fn analytic_shape_factors(axes: &Vector3<f64>) -> (f64, f64, f64) {
    let n = 160;
    let (z, w) = gauss_leg(n);
    let abc = axes[0] * axes[1] * axes[2];
    let mut out = Vector3::zeros();
    for i in 0..n {
        let t = 0.5 * (z[i] + 1.0);
        let jac = 0.5 / (1.0 - t).powi(2);
        let s = t / (1.0 - t);
        let delta =
            ((axes[0] * axes[0] + s) * (axes[1] * axes[1] + s) * (axes[2] * axes[2] + s)).sqrt();
        for c in 0..3 {
            out[c] += w[i] * jac * abc / ((axes[c] * axes[c] + s) * delta);
        }
    }
    (out[0], out[1], out[2])
}

fn tri_quadrature(duffy_order: usize) -> TriQuad {
    if duffy_order == 0 {
        let (xiq, etq, wq) = gauss_trgl(13);
        return TriQuad {
            mint: 13,
            xiq,
            etq,
            wq,
        };
    }

    let (z, w) = gauss_leg(duffy_order);
    let mut xiq = DVector::zeros(duffy_order * duffy_order);
    let mut etq = DVector::zeros(duffy_order * duffy_order);
    let mut wq = DVector::zeros(duffy_order * duffy_order);
    let mut k = 0;
    for i in 0..duffy_order {
        let u = 0.5 * (z[i] + 1.0);
        let wu = 0.5 * w[i];
        for j in 0..duffy_order {
            let v = 0.5 * (z[j] + 1.0);
            let wv = 0.5 * w[j];
            xiq[k] = u;
            etq[k] = (1.0 - u) * v;
            wq[k] = 2.0 * wu * wv * (1.0 - u);
            k += 1;
        }
    }
    TriQuad {
        mint: duffy_order * duffy_order,
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

fn basis_velocity(dof: usize) -> (Vector3<f64>, Vector3<f64>) {
    let mut u = Vector3::zeros();
    let mut omega = Vector3::zeros();
    if dof < 3 {
        u[dof] = 1.0;
    } else {
        omega[dof - 3] = 1.0;
    }
    (u, omega)
}

fn analytic_phi_vector(
    geom: &Geometry,
    axes: &Vector3<f64>,
    added: &[f64; 6],
    dof: usize,
) -> DVector<f64> {
    let volume = 4.0 / 3.0 * PI * axes[0] * axes[1] * axes[2];
    let mut out = DVector::zeros(geom.npts);

    for i in 0..geom.npts {
        let x = geom.p[(i, 0)];
        let y = geom.p[(i, 1)];
        let z = geom.p[(i, 2)];
        out[i] = match dof {
            0 => -(added[0] / (RHO_F * volume)) * x,
            1 => -(added[1] / (RHO_F * volume)) * y,
            2 => -(added[2] / (RHO_F * volume)) * z,
            3 => {
                let denom = RHO_F * volume * (axes[1] * axes[1] - axes[2] * axes[2]) / 5.0;
                -(added[3] / denom) * y * z
            }
            4 => {
                let denom = RHO_F * volume * (axes[2] * axes[2] - axes[0] * axes[0]) / 5.0;
                -(added[4] / denom) * x * z
            }
            5 => {
                let denom = RHO_F * volume * (axes[0] * axes[0] - axes[1] * axes[1]) / 5.0;
                -(added[5] / denom) * x * y
            }
            _ => unreachable!(),
        };
    }
    out
}

fn nodal_mean(v: &DVector<f64>) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn rel_phi_error(f: &DVector<f64>, analytic: &DVector<f64>) -> (f64, f64) {
    let mut diff = f - analytic;
    let offset = nodal_mean(&diff);
    for v in diff.iter_mut() {
        *v -= offset;
    }
    let denom = {
        let mut centered = analytic.clone();
        let mean = nodal_mean(&centered);
        for v in centered.iter_mut() {
            *v -= mean;
        }
        centered.norm()
    };
    (diff.norm() / denom.max(1.0e-300), offset)
}

fn best_scale(f: &DVector<f64>, analytic: &DVector<f64>) -> f64 {
    let mut fc = f.clone();
    let mut ac = analytic.clone();
    let fm = nodal_mean(&fc);
    let am = nodal_mean(&ac);
    for v in fc.iter_mut() {
        *v -= fm;
    }
    for v in ac.iter_mut() {
        *v -= am;
    }
    fc.dot(&ac) / ac.dot(&ac).max(1.0e-300)
}

fn impulse_from_phi(geom: &Geometry, f: &DVector<f64>, dof: usize, quad: &TriQuad) -> f64 {
    let mut l_lin = Vector3::zeros();
    let mut l_ang = Vector3::zeros();
    let centre = Vector3::zeros();
    for k in 0..geom.nelm {
        let (ll, la) = lamb_impulse_element_with_geom(
            k,
            quad.mint,
            &centre,
            RHO_F,
            f,
            &geom.p,
            &geom.n,
            &geom.vna,
            &geom.alpha,
            &geom.beta,
            &geom.gamma,
            &quad.xiq,
            &quad.etq,
            &quad.wq,
            geom.exact_surface.as_ref(),
        );
        l_lin += ll;
        l_ang += la;
    }
    if dof < 3 {
        -l_lin[dof]
    } else {
        -l_ang[dof - 3]
    }
}

fn rhs_for_dof(
    geom: &Geometry,
    mode: Mode,
    dof: usize,
    quad: &TriQuad,
    zz: &DVector<f64>,
    ww: &DVector<f64>,
) -> DVector<f64> {
    let (u, omega) = basis_velocity(dof);
    let dfdn = dfdn_single(&Vector3::zeros(), &u, &omega, geom.npts, &geom.p, &geom.vna);
    lslp_3d_with_geom(
        geom.npts,
        geom.nelm,
        quad.mint,
        zz.len(),
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
    )
}

fn influence_matrix(
    geom: &Geometry,
    mode: Mode,
    quad: &TriQuad,
    zz: &DVector<f64>,
    ww: &DVector<f64>,
) -> DMatrix<f64> {
    ldlp_3d_assemble_with_geom(
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
        &zz,
        &ww,
        geom.exact_surface.as_ref(),
        mode.exact_singular,
    )
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let out_path = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("scratch_density_collocation_probe.csv"));
    let max_ndiv = args.get(2).and_then(|s| s.parse::<u32>().ok()).unwrap_or(4);
    let duffy_order = args
        .get(3)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);
    let shape_kind = args.get(4).map(String::as_str).unwrap_or("ellipsoid");
    let mode_filter = args.get(5).map(String::as_str).unwrap_or("all");
    let quad = tri_quadrature(duffy_order);

    let axes = shape(shape_kind);
    let added = analytic_added_inertia(&axes);
    let dof_count = if shape_kind == "sphere" { 3 } else { 6 };
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

    let mut out = BufWriter::new(File::create(&out_path).expect("create output csv"));
    writeln!(
        out,
        "shape,mode,ndiv,dof,tri_quad,analytic_added,solved_added,analytic_phi_added,solved_added_rel_error,analytic_phi_added_rel_error,phi_rel_error,phi_offset,best_scale,analytic_residual_rel"
    )
    .unwrap();

    for ndiv in 1..=max_ndiv {
        for mode in modes {
            if mode_filter != "all" && mode.name != mode_filter {
                continue;
            }
            let geom = build_geometry(ndiv, mode, &axes, &quad);
            let nq = if mode.exact_singular { 24 } else { 12 };
            let (zz, ww) = gauss_leg(nq);
            eprintln!("mode={} ndiv={} assembling/factoring", mode.name, ndiv);
            let amat = influence_matrix(&geom, mode, &quad, &zz, &ww);
            let lu = amat.clone().lu();
            for dof in 0..dof_count {
                eprintln!("mode={} ndiv={} dof={}", mode.name, ndiv, dof);
                let f_analytic = analytic_phi_vector(&geom, &axes, &added, dof);
                let analytic_phi_added = impulse_from_phi(&geom, &f_analytic, dof, &quad);
                let rhs = rhs_for_dof(&geom, mode, dof, &quad, &zz, &ww);
                let f_solved = lu.solve(&rhs).expect("BEM solve failed");
                let solved_added = impulse_from_phi(&geom, &f_solved, dof, &quad);
                let (phi_rel_error, phi_offset) = rel_phi_error(&f_solved, &f_analytic);
                let scale = best_scale(&f_solved, &f_analytic);
                let residual = &amat * &f_analytic - &rhs;
                let residual_rel = residual.norm() / rhs.norm().max(1.0e-300);
                let target = added[dof];

                writeln!(
                    out,
                    "{},{},{},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}",
                    shape_kind,
                    mode.name,
                    ndiv,
                    dof,
                    if duffy_order == 0 {
                        "trgl13".to_string()
                    } else {
                        format!("duffy{}", duffy_order)
                    },
                    target,
                    solved_added,
                    analytic_phi_added,
                    (solved_added - target).abs() / target.abs().max(1.0e-300),
                    (analytic_phi_added - target).abs() / target.abs().max(1.0e-300),
                    phi_rel_error,
                    phi_offset,
                    scale,
                    residual_rel,
                )
                .unwrap();
            }
        }
    }
    eprintln!("wrote {}", out_path.display());
}
