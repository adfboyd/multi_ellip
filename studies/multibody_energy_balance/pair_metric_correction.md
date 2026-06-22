# Pairwise metric-correction prototype

This branch implements an experimental reduced-action correction for the
multibody impulse scheme. It is disabled by default.

## Input flags

- `impulse_pair_metric_correction=1`: enable the correction.
- `impulse_pair_metric_mode=0`: old local point-gradient mode.
- `impulse_pair_metric_mode=1`: translational discrete-gradient mode
  (current default).
- `impulse_pair_metric_mode=2`: global internal-translation discrete-gradient
  mode. This removes the common translation from the step displacement and
  samples the fluid KE at the whole internal start/end displacement with two
  energy-only BEM solves. It is intended to capture many-body translational
  metric terms that a sum of pairwise closures can miss.
- `impulse_pair_metric_cutoff=<distance>`: legacy hard cutoff. Used when
  `inner/outer` are not supplied. A non-positive value means all pairs.
- `impulse_pair_metric_inner_cutoff=<distance>` and
  `impulse_pair_metric_outer_cutoff=<distance>`: smoothstep pair weighting.
  Weight is one below the inner cutoff, zero above the outer cutoff, and smooth
  in between.
- `impulse_pair_metric_eps=<eps>`: central-difference perturbation size.
- `impulse_pair_metric_linear_scale=<scale>`: scale for the relative-translation
  metric force. Default is `1.0` in discrete-gradient mode and `0.1` in
  point-gradient mode.
- `impulse_pair_metric_angular_scale=<scale>`: scale for the relative-rotation
  metric torque. Default is `0.0`.
- `impulse_internal_load_constraint=1`: enabled by default when the pair metric
  correction is enabled.

## Formulation

At the first impulse fixed-point iterate of each timestep, the code evaluates
the fluid kinetic energy at the step midpoint and finite-differences only
pairwise relative coordinates:

- centre-preserving relative translations;
- centre-preserving relative spatial rotations.

The resulting loads are applied as equal-and-opposite pair contributions and
then passed through the internal-load constraint, so the correction is intended
to approximate a restricted configuration gradient of the eliminated fluid
action rather than a post-step projection.

The gradient is deliberately lagged for the whole timestep. Recomputing it at
every impulse fixed-point iterate would multiply the cost by roughly the fixed
point iteration count and is not a useful reduced model.

In `mode=1`, the translational load uses a pairwise discrete-gradient identity
instead of a local point gradient. For a pair `a,b`,

```text
r_n   = x_b,n   - x_a,n
r_np1 = x_b,n+1 - x_a,n+1
dr    = r_np1 - r_n
Q_ab  = w_ab * (K_f(r_np1) - K_f(r_n)) / |dr|^2 * dr
```

where `K_f` is evaluated at fixed midpoint velocity/orientation with only the
pair relative separation changed. The body loads are then

```text
F_a -= Q_ab
F_b += Q_ab
```

This is cheaper than the point-gradient mode because it needs two extra BEM
evaluations per active pair when the angular scale is zero. The angular pair
finite-difference path is still present for diagnostics, but remains disabled by
default.

In `mode=2`, the translational load is a global internal-coordinate
discrete-gradient. Let `dx_b = x_{b,n+1} - x_{b,n}` and remove the common
translation `mean_b(dx_b)`, giving internal displacements `d_b`. The code
evaluates the fluid KE at midpoint velocity/orientation and positions
`x_mid - d/2` and `x_mid + d/2`, then applies

```text
F_b = w * (K_f(x_mid + d/2) - K_f(x_mid - d/2)) / sum_c |d_c|^2 * d_b
```

where `w` is the mean active pair weight. Thus `sum_b F_b = 0` and
`sum_b F_b.d_b = Delta K_f` by construction. The usual internal-load
constraint is then applied, so the accepted load also has zero net torque about
the midpoint. This mode needs two energy-only BEM solves per step regardless of
the number of bodies, making it a candidate reduced many-body closure.

For two bodies, mode 2 collapses algebraically to the pairwise relative
translation mode. The close two-body calibration gives the same trajectory
scores as mode 1, as expected. Its value is for `nbody >= 3`, where mode 1
requires two energy-only BEM samples per active pair while mode 2 requires only
two samples for the whole internal translational displacement.

Short three-body smoke tests to `t=0.5`, `ndiv=2`, `dt=0.05`, all three pairs
active, scale `1.0`, no projection:

| case | mode | mean step | max active pairs | pair load norm | max global H drift | max per-body H drift |
|---|---|---:|---:|---:|---:|---:|
| `rho=1` | pairwise DG | `0.3512 s` | `3` | `6.53e-4` | `4.42e-6` | `1.73e-5` |
| `rho=1` | global internal DG | `0.2893 s` | `3` | `4.31e-4` | `4.42e-6` | `1.01e-5` |
| `rho=0.1` | pairwise DG | `0.4794 s` | `3` | `9.38e-4` | `5.67e-6` | `9.75e-6` |
| `rho=0.1` | global internal DG | `0.4400 s` | `3` | `6.07e-4` | `5.67e-6` | `1.33e-5` |

These are not accuracy validation cases; they only show that the global
internal mode has the expected lower cost when several pairs are active and
does not immediately damage the conservation diagnostics in short, mild
three-body runs. A real accuracy claim needs a three-body variational reference.

A tiny three-body variational reference was then run at `ndiv=2`, `dt=0.05`,
`t=0.1` with the same `rho=1` three-body setup but with the initial triangle
scaled inward to positions `(2,0,0)`, `(-1,1.732,0)`, `(-1,-1.732,0)`. The
reference took about `30 s/step`, reached residual `4.04e-11`, and conserved
discrete momentum to `2.66e-8`. Against that reference:

| run | mean step | position RMS | velocity RMS | omega RMS | separation RMS | final KE error |
|---|---:|---:|---:|---:|---:|---:|
| uncorrected impulse | `0.301 s` | `2.197e-4` | `5.677e-4` | `1.498e-3` | `1.905e-4` | `-0.0198%` |
| pairwise DG, scale 1 | `0.393 s` | `2.183e-4` | `5.553e-4` | `1.519e-3` | `1.875e-4` | `-0.0252%` |
| global internal DG, scale 1 | `0.333 s` | `2.197e-4` | `5.729e-4` | `1.513e-3` | `1.905e-4` | `-0.0250%` |
| global internal DG, scale 1.6 | `0.439 s` | `2.197e-4` | `5.761e-4` | `1.525e-3` | `1.905e-4` | `-0.0282%` |

This close-reference test is still only two timesteps, but it is the first
direct three-body variational comparison. It suggests:

- pairwise mode 1 is the better accuracy closure in this short close case,
  slightly reducing position, velocity, and separation errors;
- global internal mode 2 is not a better accuracy closure here, even when
  scaled up to match the order of the pairwise load;
- mode 2 should therefore be treated as a cheaper many-body approximation for
  larger active-pair counts, not as an accuracy replacement for pairwise mode 1.

## Initial close-contact tests

Case: spheroids `1:0.7:0.7`, `rho=1`, `E=0.25`, `sep=3`, `ndiv=2`,
`dt=0.025`, no energy projection.

| run | t_end | max KE drift | final KE drift | final sep | max global H drift | mean step |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 1 | 8.27% | 8.27% | 3.20 | 6.35e-3 | 0.100 s |
| linear scale 0.1 | 1 | 7.42% | 7.42% | 3.35 | 6.21e-3 | about 0.19 s |
| common linear/angular scale 1.0 | 1 | 20.17% | 20.17% | 4.40 | 6.40e-1 | about 0.19 s |
| baseline | 5 | 12.21% | 12.18% | 6.03 | 1.93e-2 | 0.092 s |
| linear scale 0.05 | 5 | 11.37% | 11.29% | 6.74 | 1.85e-2 | 0.142 s |
| linear scale 0.1 | 5 | 10.77% | 10.73% | 7.34 | 1.82e-2 | 0.139 s |
| common linear/angular scale -0.1 | 5 | 18.49% | 18.49% | 4.16 | 2.25e-2 | 0.208 s |

After adding the translational discrete-gradient mode:

| run | t_end | max KE drift | final KE drift | final sep | max global H drift | max per-body H drift | mean step |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1 | 8.27% | 8.27% | 3.20 | 6.35e-3 | 2.65e-3 | 0.100 s |
| point gradient, linear 0.1 | 1 | 7.42% | 7.42% | 3.35 | 6.21e-3 | 5.51e-2 | about 0.12 s |
| discrete gradient, linear 0.1 | 1 | 7.81% | 7.81% | 3.27 | 6.26e-3 | 1.80e-2 | about 0.11 s |
| baseline | 5 | 12.21% | 12.18% | 6.03 | 1.93e-2 | 1.09e-2 | 0.0975 s |
| point gradient, linear 0.1 | 5 | 10.77% | 10.73% | 7.34 | 1.82e-2 | 8.59e-2 | 0.1222 s |
| discrete gradient, linear 0.1 | 5 | 11.44% | 11.36% | 6.58 | 1.86e-2 | 2.64e-2 | 0.1117 s |
| discrete gradient, smooth 3.5/4.5 | 5 | 11.45% | 11.37% | 6.58 | 1.86e-2 | 2.77e-2 | 0.1201 s |
| discrete gradient, linear 0.5 | 5 | 9.74% | 9.74% | 8.37 | 1.79e-2 | 7.96e-2 | 0.1042 s |
| discrete gradient, linear 1.0 | 5 | 8.51% | 8.51% | 10.75 | 1.88e-2 | 1.38e-1 | 0.1084 s |

Far-pair cutoff check with `sep=8`, `cutoff=4`: zero active pairs, identical
energy/momentum diagnostics, and negligible overhead.

Short timestep check, same close case to `t=2`, using discrete-gradient scale
`1.0`:

| dt | run | max KE drift | final KE drift | final sep | max global H drift | max per-body H drift |
|---:|---|---:|---:|---:|---:|---:|
| 0.0500 | baseline | 10.57% | 10.36% | 3.69 | 3.90e-2 | 1.70e-2 |
| 0.0500 | pair DG | 8.05% | 8.02% | 5.59 | 4.17e-2 | 1.35e-1 |
| 0.0250 | baseline | 10.41% | 10.21% | 3.70 | 9.78e-3 | 4.29e-3 |
| 0.0250 | pair DG | 7.94% | 7.94% | 5.55 | 1.04e-2 | 1.32e-1 |
| 0.0125 | baseline | 10.37% | 10.18% | 3.71 | 2.45e-3 | 1.08e-3 |
| 0.0125 | pair DG | 7.95% | 7.95% | 5.53 | 2.59e-3 | 1.30e-1 |

The correction reduces the dt-invariant energy drift by roughly one quarter in
this case. The remaining drift is still essentially timestep-independent, so
the reduced pair term does not fully replace the global action formulation. The
larger final separation is also a real dynamical change, not a harmless output
diagnostic change.

## Comparison against a variational reference

To check whether the larger separation is plausible or just an over-correction,
the close spheroid case was compared against the fully determined variational
scheme at `ndiv=2`, `dt=0.025`. These short references are expensive but
well-conditioned:

- `t=0.25`: variational residual max `3.32e-9`, discrete momentum drift max
  `4.52e-6`, mean step about `6.21 s`;
- `t=0.50`: variational residual max `3.32e-9`, discrete momentum drift max
  `5.72e-6`, mean step about `6.28 s`.

Endpoint KE drift is shown for orientation only; at finite timestep, the
variational method's correct invariant is the discrete Noether momentum, not
the old endpoint continuous impulse/energy diagnostic.

| t_end | run | max KE drift | final sep | position RMS error vs variational | final sep error vs variational | velocity RMS error vs variational |
|---:|---|---:|---:|---:|---:|---:|
| 0.25 | impulse | 1.71% | 3.039 | 3.73e-2 | -1.06e-1 | 3.44e-1 |
| 0.25 | pair DG | 1.61% | 3.125 | 2.51e-2 | -2.01e-2 | 2.38e-1 |
| 0.25 | variational | 0.76% | 3.145 | - | - | - |
| 0.50 | impulse | 5.27% | 3.053 | 1.05e-1 | -3.64e-1 | 5.21e-1 |
| 0.50 | pair DG | 4.59% | 3.287 | 6.32e-2 | -1.30e-1 | 3.43e-1 |
| 0.50 | variational | 0.85% | 3.417 | - | - | - |

This supports the sign and qualitative direction of the pair discrete-gradient
correction: it moves the close-contact impulse trajectory toward the global
action reference, rather than merely improving an energy number in isolation.
It is still not a complete replacement for the variational method, because the
separation error remains significant by `t=0.5`.

The variational-defect probe also reports how the reduced pair load aligns with
the expensive forced-discrete-Euler-Lagrange residual. This diagnostic uses
`R - dt Q = 0`, so a positive pair scale should give positive alignment when
the pair load is correcting the missing variational force. In a two-step
`t=0.05` close-pair smoke run:

| run | defect vs pair load cos | fitted pair scale |
|---|---:|---:|
| uncorrected impulse endpoint | `7.20e-1` | `9.84e-1` |
| scale-1 pair-corrected endpoint | `1.39e-1` | `1.45e-1` |

This is an important consistency check: before correction, the pair load
explains a large part of the missing variational generalized force with a scale
close to the derived value `1.0`; after applying it, the remaining variational
defect is much less pair-like. That supports the pair correction as the right
low-cost first closure term and points the remaining error toward missing
many-body and/or rotational terms rather than a sign error.

## Interpretation

The translational pair metric force has the right sign: positive scale reduces
the energy drift, while negative scale is destabilising. The discrete-gradient
work identity gives scale `1.0`; that is now the default for
`impulse_pair_metric_mode=1`. This improves energy most, but also changes the
trajectory/separation most strongly. Smaller scales are useful diagnostics for
continuity with the uncorrected impulse solution, but they are no longer the
default physical choice.

The relative-rotation finite-difference component is not reliable yet; it can
damage angular-momentum diagnostics and worsen energy drift. For that reason
the current default leaves `impulse_pair_metric_angular_scale=0`.

The pair finite-difference samples, including the final end-state KE restore,
use the BEM `EnergyOnly` path, not a full impulse quadrature, because the
reduced metric force needs only the fluid kinetic energy at the sampled
configurations. In the close `t=5`, `dt=0.025` scale-1 probe this preserved the
output file bit-for-bit while changing the mean runtime from `0.1084 s/step` to
`0.1056 s/step`. The small speed gain confirms that the BEM solve itself
dominates this path; avoiding the extra surface quadrature is still the cleaner
implementation.

The impulse solver also reuses the previous accepted endpoint impulse as the
next step's start impulse. This is a state-function cache, not an approximation:
the next step starts from exactly the endpoint state whose impulse was already
solved in the previous fixed-point iteration. Projection clears the cache. In
the same close scale-1 probe the cache gave `199 / 1` start cache hits/direct
solves and preserved the output file bit-for-bit while reducing the clean mean
runtime further to `0.0980 s/step`.

If no pair actually samples an offset state (for example after all pairs have
moved outside the cutoff), the pair helper now skips the final end-state KE
restore solve. The solver is already at the accepted end-state impulse solve in
that case. This also preserved the output file bit-for-bit and reduced the same
clean probe to `0.0926 s/step`.

The focused regression runner
`studies/multibody_energy_balance/benchmark_pair_metric_regression.py` recreates
this close scale-1 case, reports the KE drift, separation, cache counters, and
output hash, and can compare the generated `.dat` file against a supplied
baseline. It is intended as a quick guard that later speed changes preserve the
validated pair-DG trajectory.

This is not yet a production fix. It is a physically motivated reduced-action
prototype that gives a controlled way to test whether close-contact metric
forces can reduce the impulse energy drift without using projection.

## Calibration against variational references

The scale `1.0` follows from the reduced translational discrete-gradient work
identity and remains the default. It is not necessarily the best reduced model
for the full multibody variational trajectory, because the pair approximation
omits higher-order many-body and rotational metric terms. A short calibration
against the expensive variational reference therefore tested whether a single
stronger translational pair scale can move the impulse solution closer to the
fully determined solution without projection.

Close spheroids `1:0.7:0.7`, `rho=1`, `E=0.25`, `sep=3`, `ndiv=2`,
`dt=0.025`, no projection:

| t_end | linear scale | angular scale | position RMS vs variational | velocity RMS vs variational | final sep error | max KE drift | max per-body H drift |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 1.0 | 0.0 | `2.66e-2` | `1.82e-1` | `-2.01e-2` | `1.607%` | `2.17e-2` |
| 0.25 | 1.3 | 0.0 | `2.60e-2` | `1.90e-1` | `+9.02e-3` | `1.779%` | `2.97e-2` |
| 0.25 | 1.4 | 0.0 | `2.64e-2` | `1.97e-1` | `+1.87e-2` | `1.842%` | `3.25e-2` |
| 0.50 | 1.0 | 0.0 | `7.56e-2` | `4.25e-1` | `-1.30e-1` | `4.594%` | `4.91e-2` |
| 0.50 | 1.4 | 0.0 | `6.60e-2` | `3.81e-1` | `-5.72e-3` | `4.816%` | `8.05e-2` |
| 0.50 | 1.5 | 0.0 | `6.65e-2` | `3.76e-1` | `+2.65e-2` | `4.892%` | `8.80e-2` |

Small angular scales (`0.05`, `0.1`) worsened the short-reference errors and
energy drift, so `impulse_pair_metric_angular_scale=0` remains the recommended
setting.

The same close case run to `t=5` showed:

| linear scale | max KE drift | final KE drift | final separation | max per-body H drift |
|---:|---:|---:|---:|---:|
| 1.0 | `8.512%` | `8.512%` | `10.745` | `0.1377` |
| 1.4 | `7.959%` | `7.822%` | `12.585` | `0.1602` |

For studies where the goal is closer agreement with the fully determined
variational trajectory at much lower cost, `impulse_pair_metric_linear_scale`
around `1.3--1.4` is the best current reduced approximation in this close-pair
test. For strict conservation diagnostics, the derived scale `1.0` is still the
more defensible default because it has lower per-body angular drift and follows
directly from the pair discrete-gradient work identity.

The calibration can now be reproduced with:

```text
python studies/multibody_energy_balance/calibrate_pair_metric_scale.py --skip-build --scales 0.8 1.0 1.2 1.3 1.35 1.4 1.45 1.5 1.6
```

The script writes a local ignored CSV and reports a normalized full-trajectory
score combining time-series position error, velocity error, separation error,
KE drift, and per-body angular impulse drift. With the default weights, the
finer sweep gives:

| reference horizon | best linear scale | score | separation RMS error | final sep error | max KE drift |
|---:|---:|---:|---:|---:|---:|
| `t=0.25` | `1.2` | `1.84e-2` | `1.56e-3` | `-6.78e-4` | `1.718%` |
| `t=0.50` | `1.3` | `2.67e-2` | `1.23e-2` | `-3.76e-2` | `4.747%` |

This is useful but not a proof of a universal closure. The optimum moving from
about `1.2` at `t=0.25` to about `1.3` at `t=0.50`, while the best endpoint
separation match at `t=0.50` is closer to `1.4`, says the pairwise
translation-only closure is still missing part of the full variational action.
The likely missing pieces are higher-order many-body and rotational metric
terms, not an energy projection. The practical reduced-model guidance is
therefore:

- use scale `1.0` for the physically derived pair work identity and cleaner
  conservation diagnostics;
- use scale `1.2--1.4` when the priority is closer short-time agreement with
  the expensive variational reference in close-contact two-body dynamics;
- do not treat any single calibrated scale as validated outside this regime
  until a matching variational reference has been run for that geometry,
  density, separation, and mesh.

## Variational reference cost reduction

The fully variational midpoint solve remains the physically determined
multibody reference, but finite-difference Newton Jacobians are expensive. The
close three-body reference case, `1:0.7:0.7` spheroids at `rho=1`, `ndiv=2`,
`dt=0.05`, `t_end=0.1`, was therefore rerun with the same residual equations
but with Broyden Jacobian updates and cross-step Jacobian reuse.

| solve strategy | mean step | wall time | Jacobian builds | final residual | discrete momentum drift |
|---|---:|---:|---:|---:|---:|
| full finite-difference Newton | `30.946 s` | `61 s` | `4` | `4.04e-11` | `2.66e-8` |
| Broyden + step-Jacobian reuse | `5.954 s` | `28 s` | `1` | `4.10e-9` | `2.61e-8` |

The numerical trajectory difference against the full Newton output was
negligible for this reference: position RMS `1.03e-11`, velocity RMS
`2.06e-10`, angular-velocity RMS `2.12e-10`, and max output-column difference
`4.42e-9`. This makes Broyden/reuse the default variational solve strategy.
The equations solved are unchanged; only the nonlinear linearisation strategy
changes. Full finite-difference Newton can still be forced with
`variational_reuse_step_jacobian=0`, `variational_reuse_jacobian=0`, and
`variational_broyden_update=0`.

The discrete Noether momentum diagnostic is also now opt-in through
`variational_momentum_diagnostic=1`. It is physically useful for validation, but
it performs additional finite-difference action probes and does not feed back
into the solve. On the same close three-body case, disabling only this
diagnostic changed physical output columns by at most `3.3e-11` while reducing
wall time from `28 s` to `19 s` and mean post-first-step cost from `6.79 s` to
`3.94 s`.

The variational probe restore path now also restores only the rigid state,
without immediately re-solving the BEM at that restored state. Each finite
difference action probe sets and solves its own midpoint state, and the accepted
step still performs the committed endpoint solve. On the same diagnostic-off
case this reduced wall time from `20.3 s` to `19.4 s`; physical output columns
changed by at most `1.1e-9`, with position RMS `2.4e-12`, velocity RMS
`3.7e-11`, and angular-velocity RMS `1.4e-11`.

If the reused Broyden Jacobian fails and the full finite-difference fallback
converges, that fresh fallback Jacobian is now retained for the next step
instead of being discarded. On the two-body `t_end=0.5` variational reference,
this reduced Jacobian builds from `4` to `2`, wall time from `48.8 s` to
`45.0 s`, and mean step time from `2.38 s` to `2.15 s`. The physical difference
from the previous fast variational output remained small: position RMS
`2.5e-11`, velocity RMS `2.7e-10`, and angular-velocity RMS `4.1e-10`. The
close three-body `t_end=0.1` smoke still used one Jacobian build and matched the
previous fast output exactly.

The code now reports variational discrete-action evaluation counts. These are a
better cost diagnostic than Jacobian builds alone, because after Broyden reuse
most cost is residual probing. In the close three-body `t_end=0.1` smoke the
two steps used `792` and `216` action evaluations (`1008` total). In the
two-body `t_end=0.5` reference the 20 steps used `5160` total action
evaluations, with typical later steps around `216--240`.

Relaxing the variational residual tolerance is therefore an effective optional
speed/accuracy tradeoff. For the two-body `t_end=0.5` reference:

| residual tolerance | action evals | Jacobian builds | wall time | position RMS vs `1e-8` | velocity RMS vs `1e-8` | omega RMS vs `1e-8` |
|---:|---:|---:|---:|---:|---:|---:|
| `1e-8` | `5160` | `2` | `45.2 s` | - | - | - |
| `1e-7` | `4632` | `1` | `41.5 s` | `6.3e-10` | `4.6e-9` | `2.9e-8` |
| `1e-6` | `4296` | `1` | `38.6 s` | `1.3e-8` | `6.1e-8` | `2.3e-7` |

The conservative recommendation is to keep `1e-8` for reference generation and
use `variational_tol=1e-7` / `hamiltonian_floor_tol=1e-7` when a cheaper
near-reference trajectory is acceptable.
