use std::sync::{Arc, Mutex};
#[cfg(feature = "timing")]
use std::time::Instant;
use nalgebra::{DMatrix, DVector, Dyn, Quaternion, UnitQuaternion, Vector3};
use rayon::prelude::*;
#[cfg(feature = "lapack")]
use nalgebra_lapack::LU;
use crate::bem::geom::*;
use crate::bem::integ::*;
use crate::bem::potentials::dfdn_single;
use crate::system::system::Simulation;

/// Type of the cached LU factorisation, matching whichever backend is active.
#[cfg(feature = "lapack")]
type CachedLu = LU<f64, Dyn, Dyn>;
#[cfg(not(feature = "lapack"))]
type CachedLu = nalgebra::linalg::LU<f64, Dyn, Dyn>;

/// Linear state for N bodies: (positions, velocities), each a stacked 3*N vector
/// where body `i` occupies rows `3*i .. 3*i+3`.
pub type LinearState = (DVector<f64>, DVector<f64>);

/// Angular state for N bodies: (orientations, angular velocities), one quaternion
/// per body (index `i` <-> body `i`).
pub type AngularState = (Vec<Quaternion<f64>>, Vec<Quaternion<f64>>);

pub struct ForceCalculate {
    pub system: Arc<Mutex<Simulation>>,
    /// Cached LU factorisation of the time-invariant influence matrix. The
    /// double-layer operator is invariant under rigid-body motion, so for a
    /// single body it is constant for all time: build + factor once, reuse
    /// forever. Multi-body rebuilds each step (until block-structured caching
    /// of the per-body self-blocks lands).
    cache: Mutex<Option<CachedLu>>,
}

impl ForceCalculate {
    pub fn new(system: Arc<Mutex<Simulation>>) -> Self {
        Self {
            system,
            cache: Mutex::new(None),
        }
    }
}

pub struct LinearUpdate {
    pub system: Arc<Mutex<Simulation>>,
}

pub struct AngularUpdate {
    pub system: Arc<Mutex<Simulation>>,
}

/// Grid every body and fold the individual meshes into a single combined surface
/// mesh. Returns the combined `(nelm, npts, p, n)` plus the per-body element counts,
/// node counts, and individual node-coordinate matrices (needed for RHS assembly).
fn grid_all_bodies(
    sys: &Simulation,
) -> (
    usize,
    usize,
    DMatrix<f64>,
    DMatrix<usize>,
    Vec<usize>,
    Vec<usize>,
    Vec<DMatrix<f64>>,
) {
    let ndiv = sys.ndiv;

    let mut nelms = Vec::with_capacity(sys.nbody);
    let mut nptss = Vec::with_capacity(sys.nbody);
    let mut ps = Vec::with_capacity(sys.nbody);
    let mut ns = Vec::with_capacity(sys.nbody);

    for b in &sys.bodies {
        let s = b.shape;
        // Equivalent radius of the (already shape-normalised) body; reproduces the
        // body's ellipsoid exactly in ellip_gridder.
        let req = (s[0] * s[1] * s[2]).powf(1.0 / 3.0);
        let orient = UnitQuaternion::from_quaternion(b.orientation);
        let (nelm_i, npts_i, p_i, n_i) =
            ellip_gridder(ndiv, req, &b.shape, &b.position, &orient);
        nelms.push(nelm_i);
        nptss.push(npts_i);
        ps.push(p_i);
        ns.push(n_i);
    }

    let (mut nelm, mut npts, mut p, mut n) =
        (nelms[0], nptss[0], ps[0].clone(), ns[0].clone());
    for i in 1..sys.nbody {
        let (ne, np, pp, nn) =
            combiner(nelm, nelms[i], npts, nptss[i], &p, &ps[i], &n, &ns[i]);
        nelm = ne;
        npts = np;
        p = pp;
        n = nn;
    }

    (nelm, npts, p, n, nelms, nptss, ps)
}

/// Split a combined per-node quantity (npts x 3) into per-body blocks using the
/// per-body node counts.
fn split_per_body(vna: &DMatrix<f64>, nptss: &[usize]) -> Vec<DMatrix<f64>> {
    let mut out = Vec::with_capacity(nptss.len());
    let mut off = 0;
    for &np in nptss {
        let mut v = DMatrix::zeros(np, 3);
        for i in 0..np {
            for j in 0..3 {
                v[(i, j)] = vna[(off + i, j)];
            }
        }
        out.push(v);
        off += np;
    }
    out
}

impl crate::ode::System4<LinearState> for ForceCalculate {
    /// Returns (linear accelerations, torques) stacked over all bodies as 3*N vectors.
    fn system(&self) -> LinearState {
        let mut sys_ref = self.system.lock().unwrap();

        let (nq, mint) = (12_usize, 13_usize);
        let nbody = sys_ref.nbody;
        let rho_f = sys_ref.fluid.density;

        #[cfg(feature = "timing")]
        let t_geom = Instant::now();
        let (nelm, npts, p, n, nelms, nptss, ps) = grid_all_bodies(&sys_ref);

        let (zz, ww) = gauss_leg(nq);
        let (xiq, etq, wq) = gauss_trgl(mint);

        let (alpha, beta, gamma) = abc_vec(nelm, &p, &n);

        let (vna, _vlm, _sa) = elm_geom(npts, nelm, mint, &p, &n, &alpha, &beta, &gamma, &xiq, &etq, &wq);
        #[cfg(feature = "timing")]
        println!("Force geom setup: {:.3}s", t_geom.elapsed().as_secs_f64());

        let vnas = split_per_body(&vna, &nptss);

        // Right-hand side: dphi/dn from each body's rigid-body velocity field.
        #[cfg(feature = "timing")]
        let t_rhs = Instant::now();
        let mut dfdn = DVector::zeros(npts);
        let mut off = 0;
        for (i, b) in sys_ref.bodies.iter().enumerate() {
            let d_i = dfdn_single(
                &b.position,
                &b.linear_velocity(),
                &b.angular_velocity().imag(),
                nptss[i],
                &ps[i],
                &vnas[i],
            );
            for k in 0..nptss[i] {
                dfdn[off + k] = d_i[k];
            }
            off += nptss[i];
        }

        let rhs = lslp_3d(npts, nelm, mint, nq, &dfdn, &p, &n, &vna, &alpha, &beta, &gamma, &xiq, &etq, &wq, &zz, &ww);
        #[cfg(feature = "timing")]
        println!("Force RHS: {:.3}s", t_rhs.elapsed().as_secs_f64());

        // Double-layer influence matrix. For a single rigid body the operator is
        // invariant under rigid-body motion, so we build + factor it once and reuse
        // the cached LU on every subsequent call. Multi-body rebuilds each step.
        let reuse = nbody == 1;
        let mut cache_guard = self.cache.lock().unwrap();

        let f = if reuse && cache_guard.is_some() {
            cache_guard
                .as_ref()
                .unwrap()
                .solve(&rhs)
                .expect("Linear resolution failed")
        } else {
            let mut amat_final = DMatrix::zeros(npts, npts);
            #[cfg(feature = "timing")]
            let t_mat = Instant::now();
            amat_final
                .as_mut_slice()
                .par_chunks_mut(npts)
                .enumerate()
                .for_each(|(j, col)| {
                    let mut q = DVector::zeros(npts);
                    q[j] = 1.0;
                    let dlp = ldlp_3d(npts, nelm, mint, nq, &q, &p, &n, &vna, &alpha, &beta, &gamma, &xiq, &etq, &wq, &zz, &ww);
                    col.copy_from_slice(dlp.as_slice());
                });
            #[cfg(feature = "timing")]
            println!("Force influence matrix: {:.3}s", t_mat.elapsed().as_secs_f64());

            #[cfg(feature = "timing")]
            let t_lu = Instant::now();
            #[cfg(feature = "lapack")]
            let decomp = LU::new(amat_final);
            #[cfg(not(feature = "lapack"))]
            let decomp = amat_final.lu();
            #[cfg(feature = "timing")]
            println!("Force LU: {:.3}s", t_lu.elapsed().as_secs_f64());

            let f = decomp.solve(&rhs).expect("Linear resolution failed");
            if reuse {
                *cache_guard = Some(decomp);
            }
            f
        };
        drop(cache_guard);

        let (_srf_area, ke_integral) = ke_3d(npts, nelm, mint, &f, &dfdn, &p, &n, &vna, &alpha, &beta, &gamma, &xiq, &etq, &wq);
        sys_ref.fluid.kinetic_energy = 0.5 * sys_ref.fluid.density * ke_integral;

        // Pressure (dphi/dt) force and torque per body.
        #[cfg(feature = "timing")]
        let t_press = Instant::now();
        let positions: Vec<Vector3<f64>> = sys_ref.bodies.iter().map(|b| b.position).collect();
        let masses: Vec<f64> = sys_ref.bodies.iter().map(|b| b.mass()).collect();

        // Map each element index to its owning body.
        let mut elm_body = vec![0_usize; nelm];
        let mut eoff = 0;
        for (i, &ne) in nelms.iter().enumerate() {
            for k in 0..ne {
                elm_body[eoff + k] = i;
            }
            eoff += ne;
        }

        let phi_committed = sys_ref.phi_committed.clone();
        let step_dt = sys_ref.step_dt;
        let has_prev = phi_committed.len() == npts;

        let contributions: Vec<(usize, Vector3<f64>, Vector3<f64>)> = (0..nelm)
            .into_par_iter()
            .map(|k| {
                let body = elm_body[k];
                let (force, torque) = if has_prev {
                    dphi_dt_force_element(k, mint, &positions[body], rho_f, &f, &phi_committed, &p, &n, &vna, &alpha, &beta, &gamma, &xiq, &etq, &wq, step_dt)
                } else {
                    (Vector3::zeros(), Vector3::zeros())
                };
                (body, force, torque)
            })
            .collect();

        let mut lin_force = vec![Vector3::zeros(); nbody];
        let mut torque = vec![Vector3::zeros(); nbody];
        for (body, fo, to) in contributions {
            lin_force[body] += fo;
            torque[body] += to;
        }

        sys_ref.phi_committed = sys_ref.phi_prev.clone();
        sys_ref.phi_prev = f.clone();
        #[cfg(feature = "timing")]
        println!("Pressure integration: {:.3}s", t_press.elapsed().as_secs_f64());

        let mut lin_accel = DVector::zeros(3 * nbody);
        let mut ang_accel = DVector::zeros(3 * nbody);
        for i in 0..nbody {
            let a = lin_force[i] / masses[i];
            for c in 0..3 {
                lin_accel[3 * i + c] = a[c];
                ang_accel[3 * i + c] = torque[i][c];
            }
        }

        (lin_accel, ang_accel)
    }
}

impl crate::ode::System2<LinearState> for LinearUpdate {
    fn system(&self, _x: f64, y: &LinearState) -> LinearState {
        let mut sys_ref = self.system.lock().unwrap();

        let (p, v) = y;
        for i in 0..sys_ref.nbody {
            let pos = Vector3::new(p[3 * i], p[3 * i + 1], p[3 * i + 2]);
            let vel = Vector3::new(v[3 * i], v[3 * i + 1], v[3 * i + 2]);
            sys_ref.bodies[i].position = pos;
            let m = sys_ref.bodies[i].mass();
            sys_ref.bodies[i].linear_momentum = vel * m;
        }

        y.clone()
    }
}

impl crate::ode::System2<AngularState> for AngularUpdate {
    fn system(&self, _x: f64, y: &AngularState) -> AngularState {
        let mut sys_ref = self.system.lock().unwrap();

        let (q, omega) = y;
        for i in 0..sys_ref.nbody {
            sys_ref.bodies[i].orientation = q[i];
            let inertia = sys_ref.bodies[i].inertia;
            let o_vec = omega[i].vector();
            let ang_mom_vec = inertia.try_inverse().unwrap() * o_vec;
            sys_ref.bodies[i].angular_momentum = Quaternion::from_imag(ang_mom_vec);
        }

        y.clone()
    }
}
