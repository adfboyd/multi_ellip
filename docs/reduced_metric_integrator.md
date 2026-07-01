# Reduced Kirchhoff-metric integrator (proposal + validation)

This note proposes a production integrator that is as physically determined as
the variational reference but much cheaper, and records the numerical evidence
that motivates building it. It should be read alongside
[`development_record.md`](development_record.md), whose "raw multibody impulse
solve is not fully determined" and "full Hamiltonian/global-action branch was
parked" entries this directly addresses.

## The structural fact

In potential flow the fluid carries no independent degrees of freedom. The
velocity potential is a linear response to the instantaneous body velocities
(exactly what the BEM solve `A f = rhs(z)` computes), so the fluid kinetic
energy is a quadratic form in the generalized velocity `z = (v_1, w_1, ..., v_N,
w_N)`:

    T_fluid(q, z) = 0.5 z^T M_a(q) z

where `M_a(q)` is the added-mass matrix and depends only on the configuration
`q` (positions + orientations of all bodies). The whole coupled body-fluid
system is therefore a finite-dimensional Lagrangian system on `Q = SE(3)^N`:

    L(q, z) = 0.5 z^T M(q) z,    M(q) = M_body + M_a(q)

i.e. geodesic motion in the configuration-dependent metric `M(q)` (Kirchhoff /
Lamb reduced dynamics). This system is fully determined, momentum is conserved
by Noether, and energy is an exact invariant of the continuous flow.

Consequences for the current solver routes:

- There is **no per-body impulse-split ambiguity**. The generalized momentum
  `p = M(q) z` is a single global covector and `p_dot = 0.5 d/dq (z^T M z)` is a
  single well-defined force. The "underdetermined multibody impulse exchange"
  in `development_record.md` is an artefact of never forming `M(q)`: the impulse
  method works per body at one velocity and never sees the off-diagonal blocks
  of `M_a(q)` or their configuration gradient. The `impulse_pair_metric_correction`
  hack is an approximation to exactly the term `d M_a / dq` supplies.
- The variational reference *is* correct but recomputes the metric implicitly,
  by finite-differencing the action (= fluid KE) through repeated full BEM
  re-solves inside a Newton/Broyden loop.

## The cheap object nobody assembled: M_a(q) itself

The influence matrix `A` depends only on configuration, not on velocity, and the
solver already caches its LU / block-diagonal preconditioner. The full 6N x 6N
added-mass matrix is obtained by solving `A f_j = rhs_j` for the 6N **unit**
rigid-body velocity fields (unit translation/rotation of each body on each axis)
and contracting — all 6N right-hand sides **share the same factorisation**, so
each extra column is a back-substitution / warm-started GMRES, not a re-assembly.

Even simpler in this codebase: `BemSolver::impulse()` already returns the Lamb
impulse `L = rho * int(phi n dA)`, which is linear in `z` with `L = -M_a(q) z`
(sign: see below). So **column j of `M_a` is one `impulse()` call at a
unit-velocity state**, with the same solver instance reused so the cached
factorisation is shared. No BEM internals need to change to assemble the metric.

Two further economies (not yet exploited, but available):

1. **Self-blocks are analytic.** For an isolated ellipsoid the added-mass tensor
   is closed-form and already in the code (`hamiltonian.rs`: `mf_calc`,
   `if_calc`); in the body frame these blocks are constant. Only the
   off-diagonal interaction blocks need BEM.
2. **Interaction blocks are smooth in relative pose** -> build a surrogate /
   interpolation table (or a reflection/multipole expansion for well-separated
   pairs, full BEM only for near pairs). For the 1440-case paper sweeps this
   could make the fluid cost negligible.

### Sign convention (important)

`impulse()` returns `L = rho * int(phi n dA)`, which is the *negative* of the
positive-definite added mass acting on the velocity — the body conserves
`m_s u - L`, i.e. effective mass `m_s + M_a` (see the note in
`bem_for_ode.rs`). The probe assembles `M_a = -dL/dz` so that `M_a` is
symmetric positive-definite and the reduced fluid energy is `+0.5 z^T M_a z`.
**Any integrator built on this `M_a` must keep this negation consistent with
whatever it consumes from `impulse()`.**

## Numerical validation

`src/bin/added_mass_matrix_probe.rs` assembles `M_a(q)` by the unit-velocity
method above and runs four checks. On `input_ref2body.txt` (N=2, ndiv=2,
close two-body, eps=1e-3):

| Check | Result | Meaning |
|---|---|---|
| Symmetry `max\|M-M^T\|` (rel) | `3.2e-5` | `M_a` is symmetric -> a genuine metric |
| Zero-velocity `\|\|L\|\|` at `z=0` | `0` exactly | `L = M_a z` is linear, no constant term |
| Energy: `0.5 z^T M_a z` vs `fluid_ke` | rel `5.8e-4` | matrix reproduces the solver's fluid KE |
| **Metric gradient**: `0.5 z^T (dM_a/ds) z` vs `d(fluid_ke)/ds` | rel `1.7e-5` | `dM_a/dq` **is** the fluid-KE configuration gradient |
| Cost: full 12-col assembly vs one solve | `~1.9x`, not `12x` | shared factorisation confirmed |

The metric-gradient check is the key result: `0.5 z^T (dM_a/dq) z` matches the
finite-differenced `d(fluid_ke)/dq` to ~4 digits. That gradient is precisely the
body-body exchange force the pair-metric correction approximates -- now available
exactly and to the correct sign/magnitude, straight from the assembled matrix.
And the whole matrix costs ~1.9x a single impulse solve, before the analytic
self-block and interaction-surrogate savings.

To reproduce:

    cargo build --release --bin added_mass_matrix_probe
    ./target/release/added_mass_matrix_probe input_ref2body.txt

## Prototype integrator + results

`src/bin/reduced_metric_integrator.rs` is a standalone implicit-midpoint
integrator on the reduced Hamiltonian (no library changes). Per step, with
`p = M(q) z`:

    z_half fixed point:  M(q_half) z_half = p_n + 0.5 dt F(q_half, z_half)
    q_{n+1} = q_n (+) dt z_half     (positions += dt v_half; orientation via lab exp)
    p_{n+1} = p_n + dt F(q_half, z_half)
    z_{n+1} = M(q_{n+1})^{-1} p_{n+1}

`M(q)` is assembled by the unit-velocity method; the metric force
`F_k = d/dq_k [0.5 z^T M z]` at fixed `z` is central-differenced from the total
kinetic energy (solid analytic + fluid from one solve) along each of the 6N
configuration DOF.

**Correctness scope of the prototype.** The lab-frame metric-force torque omits
the Euler-Poincare gyroscopic term for anisotropic bodies, so the prototype is
*exact only when that term vanishes*: spheres (isotropic inertia,
orientation-independent added mass) or frozen rotation. A proper SE(3)
Lie-group variational angular update is milestone 2.

**Results** (ndiv=2, dt=0.05, t=0..2, 40 steps):

| Case | Energy drift | Linear momentum | Behaviour |
|---|---|---|---|
| Two spheres (`input_two_sphere.txt`) | `6e-7` | `8e-8` | bounded/oscillating |
| Two triaxial ellipsoids + spin (`input_two_ellip_long.txt`) | `1e-4` | `1.7e-9` | secular energy growth |

Reading: the metric machinery is validated -- linear momentum is conserved to
~1e-9 in both cases, and energy is conserved to 6e-7 (bounded, symplectic
signature) exactly where the prototype is theoretically exact. The ellipsoid
energy drifts secularly *only* through the missing gyroscopic term, not through
the reduced-metric formulation. Fixed-point residual ~1e-11..1e-13 in 2-3
iterations. Cost ~1.3 s/step (sphere) to ~2.5 s/step (ellipsoid) at this naive
finite-difference gradient cost, before any of the analytic-self-block /
surrogate economies.

To reproduce:

    cargo build --release --bin reduced_metric_integrator
    ./target/release/reduced_metric_integrator input_two_sphere.txt
    ./target/release/reduced_metric_integrator input_two_ellip_long.txt

## Milestone 2: body-frame Kirchhoff integrator

`src/bin/reduced_metric_kirchhoff.rs` fixes the milestone-1 rotational drift.

Root cause of the prototype drift: differentiating the kinetic energy w.r.t.
orientation *at fixed lab angular velocity* double-counts the R-omega kinematic
link, injecting a spurious `-omega x L` torque for anisotropic bodies. The fix
is to work in the **body frame**, where each body's self-metric block is
constant (a free body then has zero configuration torque), and evolve the
**Kirchhoff equations** with an implicit-midpoint discretization:

    dP_b/dt  = P_b x Omega_b + F_b^cfg
    dPi_b/dt = Pi_b x Omega_b + P_b x V_b + T_b^cfg

with `(P_b, Pi_b)` body-frame linear/angular momenta, `(V_b, Omega_b)`
body-frame velocities, and `F/T^cfg` the body-frame configuration forces from the
*interaction* part of the metric (finite-differenced; identically zero for a
single body). The metric is `M_bf = T(R)^T M_lab T(R)`, `T = blkdiag(R_b, R_b)`.

**Results** (ndiv=2):

| Case | Energy drift | Linear momentum | Notes |
|---|---|---|---|
| Single spinning ellipsoid (`input_one_ellip.txt`, t=5) | `9e-14` | `O(dt^2)` | free Kirchhoff top: energy at machine precision |
| Two ellipsoids + spin (t=1) | `3e-7` | `O(dt^2)` | energy increment *decreasing* (bounded, not secular) |

The single-body energy is conserved to machine precision because implicit
midpoint conserves the quadratic energy `0.5 zeta^T M_bf zeta` and the Casimir
`|Pi|` exactly for the constant-metric (free) top. For the coupled two-body case
the residual `3e-7` is dominated by the finite-difference configuration force and
the varying-metric midpoint error, both reducible; the per-step energy increment
decreases with time rather than accumulating -- contrast the milestone-1 `1e-4`
secular ellipsoid drift.

**Milestone 2b (implemented, `--vi` flag).** The milestone-2 midpoint-on-the-ODE
conserves energy and `|Pi|` exactly but transports the *spatial* momentum map
only to `O(dt^2)` (confirmed: halving dt quarters the drift, `4.7e-3 -> 1.2e-3`
at t=5). Replacing the update with an exact SE(3) coadjoint transport
`mu_{n+1} = Ad*_f(mu_n + dt f_cfg)`, using the true per-leg increments
`f = (R_a^T R_b, R_a^T (x_b - x_a))` so the two half-legs compose to the exact
full increment, restores exact spatial-momentum conservation:

| Case (`--vi`) | Energy | `|dPlin|` | `|dPang|` |
|---|---|---|---|
| Single spinning ellipsoid (t=5) | `2e-5` (bounded) | `2e-14` | `3e-14` |
| Two ellipsoids + spin (t=1) | `1e-4` (bounded) | `2e-5` | `2e-4` |

The single-body case is decisive: both the linear and the angular spatial
momentum maps are conserved to machine precision (was `4.7e-3` / `3.5e-4`), and
energy is now bounded/oscillatory rather than machine-exact -- the correct
symplectic-momentum variational-integrator trade.

**Milestone 2c (implemented).** Making the *interacting* two-body case
machine-exact in momentum needed two fixes, found by a dt-scaling diagnosis
(the drift was `O(dt)` accumulated, i.e. `O(dt^2)` per step -- a symmetry leak,
not pure truncation):

1. *Apply the full configuration impulse once, at the half configuration, then
   transport* (`mu_mid = Ad*_{n->half} mu_n + dt f`, `mu_{n+1} = Ad*_{half->n+1}
   mu_mid`). Splitting it `0.5 dt` before / `0.5 dt` after transport leaves the
   second half in the `n+1` frame, so the per-step spatial-momentum change is
   `0.5 dt sum_b (R_b^half + R_b^{n+1}) F_b` -- the `R^{n+1}` term is a nonzero
   `O(dt^2)` leak. Applying it once makes the change `dt sum_b R_b^half F_b`.
2. *Project the FD config force onto the complement of the 6 global rigid-motion
   directions.* The interaction energy is invariant under a rigid motion of the
   whole configuration (at fixed body-frame `zeta`), so the true force has zero
   net rigid component; central-difference FD respects this only to `~1e-10`.

Both are needed: fix 1 alone reaches `~1e-10`; adding fix 2 reaches machine
precision. Result (VI, `--vi`, default projection on):

| Case | Energy | `|dPlin|` | `|dPang|` |
|---|---|---|---|
| Two ellipsoids, sep=10, t=0.5 | `8e-5` (bounded) | `9e-14` | `1e-13` |
| Two ellipsoids, sep=3.5 (close), t=0.5 | `8e-5` (bounded) | `8e-14` | `9e-14` |

The reduced-metric integrator now conserves both spatial momentum maps to machine
precision and keeps energy bounded, for anisotropic bodies in close contact --
the regime that motivated the whole reduced-metric programme. `--noproject`
toggles fix 2 for the A/B check.

To reproduce:

    cargo build --release --bin reduced_metric_kirchhoff
    ./target/release/reduced_metric_kirchhoff input_one_ellip.txt
    ./target/release/reduced_metric_kirchhoff input_two_ellip_long.txt

## Proposed integrator

Advance the reduced system with a Lie-group variational / symplectic integrator
on `SE(3)^N` (reuse the existing implicit-midpoint quaternion machinery). The
discrete Euler-Lagrange equations are algebraic in `M(q)` evaluated at a few
configurations per step (endpoints/midpoint) plus a finite difference of the
small 6N x 6N matrix for `dM/dq`. Crucially, the action is never
finite-differenced through repeated fluid solves inside Newton -- that is the
expensive step the current variational route pays.

Expected properties:

- as determined as the variational reference (it *is* the discrete variational
  integrator, on the explicit reduced metric);
- exact momentum conservation and bounded energy behaviour by construction;
- no per-body impulse ambiguity, no `pair_metric` calibration knob, no post-hoc
  energy projection;
- metric cost ~2x one impulse solve per configuration (naive), far less with
  analytic self-blocks + interaction surrogate;
- unifies the codebase: one integrator replaces {impulse + pair-metric +
  energy-projection} and the separate variational reference.

### Honest caveats

- Per configuration this is ~6N solves vs the impulse method's 1. The advantage
  rests on shared factorisation (already cached), analytic self-blocks (already
  in the repo), and interaction smoothness (surrogate). Confirm the crossover
  on a real close-contact case before committing.
- `dM_a/dq` by finite-differencing the assembled matrix needs a re-assembly of
  the interaction blocks at neighbour configurations; an analytic shape
  derivative (Hadamard / adjoint surface integral) is the eventual clean version
  but is a real BEM project.
- Orthogonal to the exact-ellipsoid-geometry accuracy work, which remains the
  right discretisation-error route.

### Milestones

1. **Done** -- explicit `M_a(q)` assembly validated (`added_mass_matrix_probe`),
   reduced implicit-midpoint integrator validated on the sphere/frozen-rotation
   regime (`reduced_metric_integrator`): energy `6e-7`, momentum `8e-8`.
2. **Done** -- body-frame Kirchhoff integrator (`reduced_metric_kirchhoff`)
   removes the anisotropic-body energy drift: single ellipsoid `9e-14`, two
   ellipsoids `3e-7` (bounded). Residual: `O(dt^2)` spatial-momentum drift.
2b. **Done** -- discrete Euler-Poincare (`Ad*`) momentum transport
   (`reduced_metric_kirchhoff --vi`): single-body spatial momenta conserved to
   `2e-14`, energy bounded. Two-body momentum limited to `2e-5` by the
   finite-difference config force, not the transport.
2c. **Done** -- full-impulse-at-half + rigid-mode projection make two-body
   spatial momentum machine-exact (`~1e-13`) at close contact, energy bounded.
3. **Next (validation)** -- compare the `--vi` scheme against the existing
   `Variational` scheme's short reference trajectory on a close two-body case
   (trajectory RMS, not just conservation diagnostics).
4. Interaction surrogate / analytic self-blocks to cut the metric cost, then
   promote toward a production `CouplingScheme`.
