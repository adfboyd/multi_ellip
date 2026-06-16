use crate::bem::geom::*;
use crate::bem::gmres::gmres;
use crate::bem::integ::*;
use crate::bem::potentials::dfdn_single;
use crate::system::system::Simulation;
use nalgebra::{DMatrix, DVector, Dyn, Quaternion, UnitQuaternion, Vector3};
#[cfg(feature = "lapack")]
use nalgebra_lapack::LU;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
#[cfg(feature = "timing")]
use std::time::Instant;

/// Type of the cached LU factorisation, matching whichever backend is active.
#[cfg(feature = "lapack")]
type CachedLu = LU<f64, Dyn, Dyn>;
#[cfg(not(feature = "lapack"))]
type CachedLu = nalgebra::linalg::LU<f64, Dyn, Dyn>;

/// LU-factor a dense matrix with whichever backend is active.
#[cfg(feature = "lapack")]
fn factor(m: DMatrix<f64>) -> CachedLu {
    LU::new(m)
}
#[cfg(not(feature = "lapack"))]
fn factor(m: DMatrix<f64>) -> CachedLu {
    m.lu()
}

/// Cached factorisation strategy for the influence-matrix solve.
enum SolveCache {
    /// Single body: the whole influence matrix is time-invariant, so its LU is
    /// cached and reused directly.
    Direct(CachedLu),
    /// Multi-body: each per-body self-block is time-invariant. The dense
    /// self-blocks are cached (for the matrix-vector product) along with their
    /// LU factorisations (used as a block-diagonal GMRES preconditioner). Only
    /// the off-diagonal interaction blocks are reassembled each step.
    BlockDiag {
        self_dense: Vec<DMatrix<f64>>,
        self_lu: Vec<CachedLu>,
    },
}

/// Linear state for N bodies: (positions, velocities), each a stacked 3*N vector
/// where body `i` occupies rows `3*i .. 3*i+3`.
pub type LinearState = (DVector<f64>, DVector<f64>);

/// Angular state for N bodies: (orientations, angular velocities), one quaternion
/// per body (index `i` <-> body `i`).
pub type AngularState = (Vec<Quaternion<f64>>, Vec<Quaternion<f64>>);

/// Complete rigid-body configuration + velocities for all bodies: the single
/// state object the integrator owns and hands to the solver.
#[derive(Clone)]
pub struct BodyState {
    pub lin: LinearState,
    pub ang: AngularState,
}

/// Owns the BEM coupling: pushes the rigid-body state into the shared
/// [`Simulation`], solves the boundary-integral problem, and returns the
/// hydrodynamic loads. Replaces the former `LinearUpdate`/`AngularUpdate`/
/// `ForceCalculate` System-trait trio and the flag-toggling that the integrator
/// used to do directly.
pub struct BemSolver {
    pub system: Arc<Mutex<Simulation>>,
    /// Cached factorisation(s) of the time-invariant part of the influence
    /// matrix. The double-layer operator is invariant under rigid-body motion,
    /// so each body's self-block is constant for all time. Single body: cache
    /// the full LU (Direct). Multi-body: cache the self-block LUs as a GMRES
    /// preconditioner (BlockDiag). Built once on the first force evaluation.
    cache: Mutex<Option<SolveCache>>,
    /// Most recent multi-body GMRES solution φ, reused as the warm-start initial
    /// guess for the next solve. Consecutive solves (across fixed-point iterations
    /// and across steps) have nearly identical φ, so this cuts GMRES iterations
    /// sharply. Warm-starting only changes the iteration path, not the converged
    /// solution (still satisfies the same residual tolerance).
    last_phi: Mutex<Option<DVector<f64>>>,
}

impl BemSolver {
    pub fn new(system: Arc<Mutex<Simulation>>) -> Self {
        Self {
            system,
            cache: Mutex::new(None),
            last_phi: Mutex::new(None),
        }
    }

    /// Push the rigid-body linear (positions, velocities) and angular
    /// (orientations, angular velocities) state into the shared simulation so the
    /// next [`solve`](Self::solve) sees the current configuration.
    pub fn set_state(&self, state: &BodyState) {
        let mut sys_ref = self.system.lock().unwrap();
        let (p, v) = &state.lin;
        for i in 0..sys_ref.nbody {
            let pos = Vector3::new(p[3 * i], p[3 * i + 1], p[3 * i + 2]);
            let vel = Vector3::new(v[3 * i], v[3 * i + 1], v[3 * i + 2]);
            sys_ref.bodies[i].position = pos;
            let m = sys_ref.bodies[i].mass();
            sys_ref.bodies[i].linear_momentum = vel * m;
        }
        let (q, omega) = &state.ang;
        for i in 0..sys_ref.nbody {
            sys_ref.bodies[i].orientation = q[i];
            // angular_momentum field holds L = I·ω (angular_velocity() divides it
            // back by I). The integrator state `omega` is the lab angular velocity,
            // so momentum = inertia * omega (matches angular_momentum_from_vel).
            let inertia = sys_ref.bodies[i].inertia;
            let o_vec = omega[i].vector();
            let ang_mom_vec = inertia * o_vec;
            sys_ref.bodies[i].angular_momentum = Quaternion::from_imag(ang_mom_vec);
        }
    }

    /// Solve in impulse mode and return the per-body lab-frame fluid impulse
    /// (L_lin, L_ang). State function: no ∂φ/∂t, no φ-history side effects.
    pub fn impulse(&self) -> (DVector<f64>, DVector<f64>) {
        if let Ok(mut s) = self.system.lock() {
            s.impulse_mode = true;
        }
        let out = self.solve();
        if let Ok(mut s) = self.system.lock() {
            s.impulse_mode = false;
        }
        out
    }

    /// Solve in pressure mode and return per-body (linear acceleration, torque).
    pub fn force(&self) -> LinearState {
        self.solve()
    }

    /// Toggle the shared `freeze_phi_history` flag so strong-coupling trial
    /// evaluations don't push provisional φ into the committed history.
    pub fn set_freeze(&self, freeze: bool) {
        if let Ok(mut s) = self.system.lock() {
            s.freeze_phi_history = freeze;
        }
    }

    /// Fluid kinetic energy recorded by the most recent solve.
    pub fn fluid_kinetic_energy(&self) -> f64 {
        self.system
            .lock()
            .map(|s| s.fluid.kinetic_energy)
            .unwrap_or(0.0)
    }
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
        let (nelm_i, npts_i, p_i, n_i) = ellip_gridder(ndiv, req, &b.shape, &b.position, &orient);
        nelms.push(nelm_i);
        nptss.push(npts_i);
        ps.push(p_i);
        ns.push(n_i);
    }

    let (mut nelm, mut npts, mut p, mut n) = (nelms[0], nptss[0], ps[0].clone(), ns[0].clone());
    for i in 1..sys.nbody {
        let (ne, np, pp, nn) = combiner(nelm, nelms[i], npts, nptss[i], &p, &ps[i], &n, &ns[i]);
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

impl BemSolver {
    /// Solve the boundary-integral problem at the currently-set state. In impulse
    /// mode returns the per-body fluid impulse; otherwise (linear accel, torque)
    /// from the unsteady-pressure model.
    fn solve(&self) -> LinearState {
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

        let (vna, _vlm, _sa) = elm_geom(
            npts, nelm, mint, &p, &n, &alpha, &beta, &gamma, &xiq, &etq, &wq,
        );
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

        let rhs = lslp_3d(
            npts, nelm, mint, nq, &dfdn, &p, &n, &vna, &alpha, &beta, &gamma, &xiq, &etq, &wq, &zz,
            &ww,
        );
        #[cfg(feature = "timing")]
        println!("Force RHS: {:.3}s", t_rhs.elapsed().as_secs_f64());

        // Double-layer influence matrix solve. Each body's self-block is
        // invariant under rigid-body motion. Single body: the whole matrix is
        // constant, so cache its LU and back-substitute each step. Multi-body:
        // cache the self-block LUs once as a block-diagonal preconditioner and
        // solve with GMRES, rebuilding only the (smooth) interaction blocks
        // implicitly via the freshly assembled matrix-vector product.
        let mut cache_guard = self.cache.lock().unwrap();

        let f = if nbody == 1 {
            if cache_guard.is_none() {
                #[cfg(feature = "timing")]
                let t_mat = Instant::now();
                let amat_final = ldlp_3d_assemble(
                    npts, nelm, mint, nq, &p, &n, &vna, &alpha, &beta, &gamma, &xiq, &etq, &wq,
                    &zz, &ww,
                );
                #[cfg(feature = "timing")]
                println!(
                    "Force influence matrix: {:.3}s",
                    t_mat.elapsed().as_secs_f64()
                );
                *cache_guard = Some(SolveCache::Direct(factor(amat_final)));
            }
            match cache_guard.as_ref().unwrap() {
                SolveCache::Direct(lu) => lu.solve(&rhs).expect("Linear resolution failed"),
                _ => unreachable!(),
            }
        } else {
            // Multi-body. Cache the time-invariant self-blocks once; each step
            // reassemble only the (non-singular) interaction blocks.
            #[cfg(feature = "timing")]
            let t_mat = Instant::now();

            let a_int = if cache_guard.is_none() {
                // First evaluation: assemble the full matrix, split out the
                // dense self-blocks (+ LU), then zero their positions so the
                // remainder is exactly the interaction matrix.
                let mut amat_full = ldlp_3d_assemble(
                    npts, nelm, mint, nq, &p, &n, &vna, &alpha, &beta, &gamma, &xiq, &etq, &wq,
                    &zz, &ww,
                );
                let mut self_dense = Vec::with_capacity(nbody);
                let mut self_lu = Vec::with_capacity(nbody);
                let mut off = 0;
                for b in 0..nbody {
                    let np = nptss[b];
                    let sub = amat_full
                        .view_range(off..off + np, off..off + np)
                        .into_owned();
                    self_lu.push(factor(sub.clone()));
                    self_dense.push(sub);
                    amat_full
                        .view_range_mut(off..off + np, off..off + np)
                        .fill(0.0);
                    off += np;
                }
                *cache_guard = Some(SolveCache::BlockDiag {
                    self_dense,
                    self_lu,
                });
                amat_full
            } else {
                ldlp_3d_assemble_interactions(
                    npts, nelm, mint, &nptss, &p, &n, &vna, &alpha, &beta, &gamma, &xiq, &etq, &wq,
                )
            };
            #[cfg(feature = "timing")]
            println!(
                "Force influence matrix: {:.3}s",
                t_mat.elapsed().as_secs_f64()
            );

            let (self_dense, self_lu) = match cache_guard.as_ref().unwrap() {
                SolveCache::BlockDiag {
                    self_dense,
                    self_lu,
                } => (self_dense, self_lu),
                _ => unreachable!(),
            };

            // A v = (interaction blocks) v + sum_b (self-block_b) v_b
            let matvec = |v: &DVector<f64>| -> DVector<f64> {
                let mut out = &a_int * v;
                let mut off = 0;
                for b in 0..nbody {
                    let np = nptss[b];
                    let vb = v.rows(off, np).into_owned();
                    let yb = &self_dense[b] * vb;
                    for k in 0..np {
                        out[off + k] += yb[k];
                    }
                    off += np;
                }
                out
            };

            // Block-diagonal preconditioner: apply each cached self-block LU.
            let precond = |r: &DVector<f64>| -> DVector<f64> {
                let mut out = DVector::zeros(npts);
                let mut off = 0;
                for b in 0..nbody {
                    let np = nptss[b];
                    let rb = r.rows(off, np).into_owned();
                    let xb = self_lu[b].solve(&rb).expect("block solve failed");
                    for k in 0..np {
                        out[off + k] = xb[k];
                    }
                    off += np;
                }
                out
            };

            #[cfg(feature = "timing")]
            let t_solve = Instant::now();
            // Warm start from the previous solve's φ when dimensions match.
            let x0 = {
                let lp = self.last_phi.lock().unwrap();
                match lp.as_ref() {
                    Some(prev) if prev.len() == npts => prev.clone(),
                    _ => DVector::zeros(npts),
                }
            };
            let (f, _iters, _res) = gmres(matvec, precond, &rhs, &x0, 1e-11, 200);
            *self.last_phi.lock().unwrap() = Some(f.clone());
            #[cfg(feature = "timing")]
            println!(
                "Force GMRES: {:.3}s  ({} iters, res {:.1e})",
                t_solve.elapsed().as_secs_f64(),
                _iters,
                _res
            );

            f
        };
        drop(cache_guard);

        let (_srf_area, ke_integral) = ke_3d(
            npts, nelm, mint, &f, &dfdn, &p, &n, &vna, &alpha, &beta, &gamma, &xiq, &etq, &wq,
        );
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

        // Approach A: state-function impulse mode. Return the lab-frame fluid
        // impulse L_lin = ρ∮φn̂dA and L_ang = ρ∮φ(r×n̂)dA per body (packed into
        // the (linear, angular) output vectors), with NO ∂φ/∂t force and NO
        // history push. The integrator differences these between step endpoints
        // to form F = -dL/dt, an energy-consistent (state-function) force.
        if sys_ref.impulse_mode {
            let contributions: Vec<(usize, Vector3<f64>, Vector3<f64>)> = (0..nelm)
                .into_par_iter()
                .map(|k| {
                    let body = elm_body[k];
                    let (l_lin, l_ang) = lamb_impulse_element(
                        k,
                        mint,
                        &positions[body],
                        rho_f,
                        &f,
                        &p,
                        &n,
                        &vna,
                        &alpha,
                        &beta,
                        &gamma,
                        &xiq,
                        &etq,
                        &wq,
                    );
                    (body, l_lin, l_ang)
                })
                .collect();
            let mut l_lin_out = DVector::zeros(3 * nbody);
            let mut l_ang_out = DVector::zeros(3 * nbody);
            for (body, ll, la) in contributions {
                for c in 0..3 {
                    l_lin_out[3 * body + c] += ll[c];
                    l_ang_out[3 * body + c] += la[c];
                }
            }
            return (l_lin_out, l_ang_out);
        }

        // Same-stage BDF2 ∂φ/∂t stencil. The history holds φ from previous force
        // evaluations (most recent last). Same-stage spacing is dt (two calls per
        // step), so φ_{c−2} = phi_history[n−2] is one timestep back and
        // φ_{c−4} = phi_history[n−4] two timesteps back, both at the current stage.
        let step_dt = sys_ref.step_dt;
        let bootstrap = sys_ref.bootstrap_redos > 0;
        let hist = &sys_ref.phi_history;
        let n_hist = hist.len();
        let valid = |idx: usize| -> bool { idx < n_hist && hist[idx].len() == npts };

        let fire_bootstrap = bootstrap && n_hist == 2 && valid(0) && valid(1);
        let phi_dot: DVector<f64> = if fire_bootstrap {
            // First-step bootstrap (stage A at t0 of a repeat pass): the two
            // history entries are the previous provisional pass's φ at t0 and
            // t0 + dt/2; their forward difference seeds φ̇(t0). Each pass
            // contracts the initial added-mass acceleration error geometrically.
            (&hist[1] - &hist[0]) / (0.5 * step_dt)
        } else if n_hist >= 4 && valid(n_hist - 2) && valid(n_hist - 4) {
            // BDF2 (2nd order at the current call's time): h = dt. Optionally
            // blended toward the 1st-order same-stage difference by `phidot_blend`
            // (eps): phi_dot = (1-eps)*BDF2 + eps*BDF1. BDF2's high-frequency
            // (period-2) stencil gain is 4/dt vs 2/dt for BDF1; blending lowers
            // it to (4-2eps)/dt to damp the explicit added-mass instability that
            // appears at fine meshes (ndiv=4). eps=0 is pure BDF2 (default).
            let eps = sys_ref.phidot_blend;
            let bdf2 = (3.0 * &f - 4.0 * &hist[n_hist - 2] + &hist[n_hist - 4]) / (2.0 * step_dt);
            if eps > 0.0 {
                let bdf1 = (&f - &hist[n_hist - 2]) / step_dt;
                (1.0 - eps) * bdf2 + eps * bdf1
            } else {
                bdf2
            }
        } else if n_hist >= 2 && valid(n_hist - 2) {
            // 1st-order backward difference fallback (same-stage spacing dt).
            (&f - &hist[n_hist - 2]) / step_dt
        } else if n_hist == 1 && valid(0) {
            // Stage B of the first step (real or provisional): the single entry
            // is the stage-A φ at t0, half a step earlier.
            (&f - &hist[0]) / (0.5 * step_dt)
        } else {
            // Very first call ever: no temporal information at all.
            DVector::zeros(npts)
        };

        let impulse_transport = sys_ref.impulse_transport;

        let contributions: Vec<(
            usize,
            Vector3<f64>,
            Vector3<f64>,
            Vector3<f64>,
            Vector3<f64>,
        )> = (0..nelm)
            .into_par_iter()
            .map(|k| {
                let body = elm_body[k];
                let (force, torque) = dphi_dt_force_element(
                    k,
                    mint,
                    &positions[body],
                    rho_f,
                    &f,
                    &phi_dot,
                    &p,
                    &n,
                    &vna,
                    &alpha,
                    &beta,
                    &gamma,
                    &xiq,
                    &etq,
                    &wq,
                );
                let (l_lin, l_ang) = if impulse_transport {
                    lamb_impulse_element(
                        k,
                        mint,
                        &positions[body],
                        rho_f,
                        &f,
                        &p,
                        &n,
                        &vna,
                        &alpha,
                        &beta,
                        &gamma,
                        &xiq,
                        &etq,
                        &wq,
                    )
                } else {
                    (Vector3::zeros(), Vector3::zeros())
                };
                (body, force, torque, l_lin, l_ang)
            })
            .collect();

        let mut lin_force = vec![Vector3::zeros(); nbody];
        let mut torque = vec![Vector3::zeros(); nbody];
        let mut l_lin = vec![Vector3::zeros(); nbody];
        let mut l_ang = vec![Vector3::zeros(); nbody];
        for (body, fo, to, ll, la) in contributions {
            lin_force[body] += fo;
            torque[body] += to;
            l_lin[body] += ll;
            l_ang[body] += la;
        }

        if impulse_transport {
            // F = dL_lin/dt = (∂φ/∂t term) + ω × L_lin;  N similarly + ω × L_ang.
            // ω is the body's lab-frame angular velocity (same vector dfdn uses).
            for b in 0..nbody {
                let omega = sys_ref.bodies[b].angular_velocity().imag();
                lin_force[b] += omega.cross(&l_lin[b]);
                torque[b] += omega.cross(&l_ang[b]);
            }
        }

        if fire_bootstrap {
            // Discard the provisional pass's history so the next pass (or the
            // real first step) rebuilds it from this refined starting force.
            sys_ref.phi_history.clear();
            sys_ref.bootstrap_redos -= 1;
        }
        // Strong-coupling trial evaluations must not pollute the committed
        // φ history (they re-solve at provisional velocities); the integrator
        // freezes pushing during iteration and commits the converged φ itself.
        if !sys_ref.freeze_phi_history {
            sys_ref.phi_history.push_back(f.clone());
            while sys_ref.phi_history.len() > 4 {
                sys_ref.phi_history.pop_front();
            }
        }
        #[cfg(feature = "timing")]
        println!(
            "Pressure integration: {:.3}s",
            t_press.elapsed().as_secs_f64()
        );

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

// (LinearUpdate / AngularUpdate / ForceCalculate System-trait impls removed;
// their behaviour now lives in BemSolver::{set_state, impulse, force, solve}.)
