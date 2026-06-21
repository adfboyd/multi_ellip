# Multi-Body Energy Balance Diagnostics

This branch starts from `master` and focuses on diagnosing the multi-body
energy drift in the impulse scheme.

## Current Findings

- Reducing `dt` does not remove the multi-body energy drift, so this does not
  look like ordinary time-integration error.
- The global conserved-momentum candidates are usually much flatter than the
  total-energy drift:
  - `P = sum_b pcon_b`
  - `H = sum_b hcon_b`
- The fluid kinetic energy from direct surface integration agrees closely with
  the generalized impulse identity
  `K_f = -0.5 * sum_b(v_b.L_b + omega_b.Lambda_b)`.
- The discrete linear/angular impulse-work residuals are small in the cleaner
  three-body cases, so the most obvious impulse bookkeeping terms are not the
  dominant energy leak.

## Working Interpretation

The remaining issue is likely a multi-body formulation error rather than a
quadrature scale factor or a simple sign error.

The current impulse scheme effectively enforces each body's
`m_b v_b - L_b` as an almost independently conserved quantity. That is valid
for the single-body free problem, but for multiple interacting bodies the
symmetry only guarantees global linear/angular momentum conservation. Individual
bodies should exchange canonical momentum through the configuration-dependent
multi-body added-mass matrix.

That would explain the observed symptoms:

- single-body cases are good;
- translation-only and rotation-only diagnostics are good;
- total multi-body momentum can remain flat while total energy drifts;
- the drift can become more visible as `ndiv` increases, because the
  body-body hydrodynamic interaction is being resolved more accurately.

## Likely Fix Direction

The next physics change should avoid treating per-body impulse differences as
independent conservation laws in the multi-body case. Candidate approaches:

- formulate the impulse update as a global constrained solve that conserves
  only total `P` and total `H`, while allowing per-body exchange terms;
- derive/use the generalized force from the gradient of the total fluid kinetic
  energy with respect to body positions and orientations;
- keep the current per-body impulse update for `nbody == 1`, where it is the
  correct and well-tested special case.

The diagnostic script in this directory is intended to track whether any future
fix improves energy without merely hiding errors in `P`, `H`, or the fluid
energy identity.

## Ruled Out: Bolted-On Fluid-Energy Gradient

A finite-difference prototype was tested locally: central differences of the
fluid kinetic energy with respect to midpoint body positions and orientations
were added directly to the existing per-body impulse residual.

Smoke-test outcome before removing the prototype:

- `ndiv=1`, `t=1`, `dt=0.2`: baseline max drift `0.0074%`; gradient max drift
  about `0.22%`.
- `ndiv=2`, `t=5`, `dt=0.2`: baseline max drift `0.070%`; gradient max drift
  about `1.70%`.

So this naive correction is not the fix, and the code path has been removed.
The result suggests the missing multi-body term cannot simply be bolted onto
the existing per-body impulse conservation law; the multi-body impulse update
likely needs a genuinely coupled discrete variational solve, or the
rotational/configuration gradient needs to be derived in the exact generalized
coordinates used by the PCDM update.
