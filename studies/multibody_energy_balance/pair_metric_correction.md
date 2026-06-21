# Pairwise metric-correction prototype

This branch implements an experimental reduced-action correction for the
multibody impulse scheme. It is disabled by default.

## Input flags

- `impulse_pair_metric_correction=1`: enable the correction.
- `impulse_pair_metric_mode=0`: old local point-gradient mode.
- `impulse_pair_metric_mode=1`: translational discrete-gradient mode
  (current default).
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

This is not yet a production fix. It is a physically motivated reduced-action
prototype that gives a controlled way to test whether close-contact metric
forces can reduce the impulse energy drift without using projection.
