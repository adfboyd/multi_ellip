# Pairwise metric-correction prototype

This branch implements an experimental reduced-action correction for the
multibody impulse scheme. It is disabled by default.

## Input flags

- `impulse_pair_metric_correction=1`: enable the correction.
- `impulse_pair_metric_cutoff=<distance>`: only correct body pairs whose
  midpoint centre distance is below this value. A non-positive value means all
  pairs.
- `impulse_pair_metric_eps=<eps>`: central-difference perturbation size.
- `impulse_pair_metric_linear_scale=<scale>`: scale for the relative-translation
  metric force. Default when enabled through the common scale is `0.1`.
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

Far-pair cutoff check with `sep=8`, `cutoff=4`: zero active pairs, identical
energy/momentum diagnostics, and negligible overhead.

## Interpretation

The translational pair metric force has the right broad energy direction in the
close case, but the effect is modest and visibly changes the trajectory. The
relative-rotation finite-difference component is not reliable yet; it can
damage angular-momentum diagnostics and worsen energy drift. For that reason
the current default leaves `impulse_pair_metric_angular_scale=0`.

This is not yet a production fix. It is a physically motivated reduced-action
prototype that gives a controlled way to test whether close-contact metric
forces can reduce the impulse energy drift without using projection.
