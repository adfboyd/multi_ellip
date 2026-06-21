# Ruled-out multibody energy fixes

These are diagnostic paths that have been tested and should not be treated as
promising fixes unless there is new evidence.

## Off-body near-singular quadrature

Close initial body separation strongly increases the no-projection impulse
energy drift, so one candidate was under-resolved off-body BEM interaction
quadrature. A temporary implementation replaced cross-body single- and
double-layer element interactions with uniformly subdivided parametric
triangles when the target point was close to the source element.

Short spheroid `1:0.7:0.7`, `rho=1`, `E=0.25`, `sep=3`, `ndiv=2`, `dt=0.025`
tests showed no useful change:

- standard run to `t=2.5`: total KE `499.0186`;
- forced near rule on all cross elements, one subdivision: total KE
  `499.0185707875`, mean step about `1.22 s`;
- forced near rule on all cross elements, two subdivisions: total KE
  `499.0185707886`, mean step about `4.19 s`.

The energy result was unchanged to roundoff while the run became roughly
10--30 times slower. The close-contact error is therefore not explained by
ordinary off-body cross-quadrature resolution. The more likely issue remains
the impulse scheme's multibody exchange model: the current update effectively
locks per-body continuous impulse candidates even though only the global
linear and angular impulse are symmetry invariants.

## Enabling the existing metric/pressure correction candidates directly

A short close-contact defect probe was run for the same spheroid case with
`impulse_variational_defect_probe=1`, `dt=0.025`, and `t=0.25`. This compares
the impulse residual defect against the existing finite-difference fluid-energy
metric gradient and quadratic-pressure load candidates.

Outcome:

- runtime increased from about `0.13 s/step` to about `1.02 s/step`;
- the defect norm remained O(10);
- the metric-gradient alignment was mostly negative;
- the pressure-load alignment changed sign over the short run.

Representative per-step cosine/scale values:

| time | defect norm | metric cos | metric scale | pressure cos | pressure scale |
|---:|---:|---:|---:|---:|---:|
| 0.000 | 41.66 | -0.530 | -0.587 | -0.608 | -0.584 |
| 0.100 | 19.39 | -0.324 | -0.158 | 0.471 | 0.188 |
| 0.175 | 16.50 | -0.908 | -0.836 | -0.083 | -0.026 |
| 0.225 | 15.74 | -0.536 | -0.297 | -0.391 | -0.120 |

So the existing correction vectors do not provide a stable physical
sign/scale. Turning them on directly would be another projection-like patch, not
a determined multibody impulse model.

## Refreshing the pair metric force every fixed-point iteration

The pair discrete-gradient correction is currently evaluated once per step and
then held fixed through the endpoint fixed-point solve. A more literal reduced
action residual would refresh that pair force at the current endpoint on every
fixed-point iteration, but this is much more expensive because each active pair
adds two BEM energy solves per refresh.

For the close spheroid `1:0.7:0.7`, `rho=1`, `E=0.25`, `sep=3`, `ndiv=2`,
`dt=0.025`, `t=0.5` probe:

| pair metric evaluation | mean step | final KE drift | final separation |
|---|---:|---:|---:|
| frozen once per step | `0.1237 s` | `4.594%` | `3.2866` |
| refreshed every FP iteration | `0.3937 s` | `4.648%` | `3.2798` |

The variational reference final separation for the same short case is about
`3.417`. Refreshing the pair force therefore costs roughly three times more
without improving either energy drift or trajectory agreement. The frozen
single evaluation remains the better reduced approximation unless a different
pair force derivation is introduced.

## Warm-starting the implicit torque iterate

The impulse fixed-point solve normally starts each step with zero torque. A
trial used the previous accepted step's torque as the next initial guess. This
does not change the residual being solved, but with the current finite
fixed-point tolerance it does change the nonlinear iteration path slightly.

For the close spheroid `1:0.7:0.7`, `rho=1`, `E=0.25`, `sep=3`, `ndiv=2`,
`dt=0.025`, `t=5` pair-DG case:

| initial torque guess | mean step | mean FP iters | final KE drift | final separation |
|---|---:|---:|---:|---:|
| zero | `0.0980 s` | `10.060` | `8.512%` | `10.7454` |
| previous step torque | `0.0963 s` | `9.965` | `8.506%` | `10.7686` |

The speed gain is only about two percent, while positions differ by up to
O(`1e-2`) over this run. That is too much trajectory dependence for such a
small gain, so the zero-torque initial guess is kept.

## Surface-clearance weighting for pair activation

The pair metric correction currently uses centre distance in the smooth
inner/outer cutoff. A trial replaced that distance with an ellipsoid
line-of-centres surface clearance,

`gap = |x_b - x_a| - h_a(e) - h_b(-e)`,

where `e` is the centre-line direction and `h_i` is the oriented ellipsoid
support radius. This is geometrically better motivated than centre distance,
but it did not improve the validated close-contact case.

For the close spheroid pair-DG case to `t=5`, `ndiv=2`, `dt=0.025`:

| weighting | cutoff band | max KE drift | final separation | active output rows | max per-body H drift |
|---|---:|---:|---:|---:|---:|
| centre distance | `4 / 4` | `8.512%` | `10.7454` | `12 / 51` | `0.1377` |
| surface clearance | `4 / 4` | `9.200%` | `12.3481` | `22 / 51` | `0.1950` |
| surface clearance | `1 / 2` | `8.626%` | `9.3443` | `12 / 51` | `0.0841` |

Against the short `t=0.5` variational reference, the tuned `1 / 2` clearance
band was also slightly worse than centre distance:

| run | max KE drift | final separation error | position RMS error | velocity RMS error |
|---|---:|---:|---:|---:|
| impulse | `5.270%` | `-0.3638` | `0.1262` | `0.5767` |
| pair, centre cutoff | `4.594%` | `-0.1301` | `0.0756` | `0.4251` |
| pair, clearance `1 / 2` | `4.569%` | `-0.1384` | `0.0768` | `0.4333` |

Although clearance weighting is physically plausible and can reduce the
per-body angular drift in one long diagnostic, it does not improve the
trajectory against the variational reference. It should not replace the current
centre-distance cutoff without a better reduced-force derivation.

## Energy-only samples for the full global discrete-gradient correction

The reduced pair metric correction benefits from using BEM `EnergyOnly` solves
for its finite-difference samples, but the older full global
`impulse_metric_correction` path did not show the same benefit. A trial changed
`fluid_energy_configuration_discrete_gradient` to use `kinetic_energy_only()`
for its KE samples and final restore.

For the close spheroid global-discrete-gradient diagnostic, `ndiv=2`,
`dt=0.025`, `t=0.1`:

| sample solve path | mean step | total wall time | max global H drift | max per-body H drift |
|---|---:|---:|---:|---:|
| full impulse samples | `2.6730 s` | `10.84 s` | `1.510e-3` | `1.190e-1` |
| energy-only samples | `3.0053 s` | `11.87 s` | `1.510e-3` | `1.190e-1` |

The output diagnostics were unchanged, but the energy-only version was slower
for this path, likely because the full impulse path keeps a more favourable
warm-start sequence for the subsequent solves. The global correction remains a
diagnostic and should not be switched to `EnergyOnly` unless a broader benchmark
shows a real benefit.

## Direct combined-array RHS boundary condition loop

`BemSolver::solve` used to split the combined node normal matrix into per-body
blocks before evaluating `dfdn = n . (u + omega x (x - c))`. A trial removed
that allocation and computed `dfdn` directly from the combined point/normal
arrays with per-body offsets. This is algebraically identical, but it was not
faster in the measured pair-DG path.

For the close pair-DG benchmark, `ndiv=2`, `dt=0.025`, `t=5`, the output hash
matched the baseline exactly, but the measured mean step was noisy and not
better (`0.0962 s` in that run). The clearer `ndiv=3`, `t=0.075` comparison was
also slower:

| RHS path | first-step setup | mean step | total wall time |
|---|---:|---:|---:|
| existing split/block RHS | `4.405 s` | `1.7689 s` | `7.94 s` |
| direct combined-array RHS | `4.774 s` | `1.8431 s` | `8.46 s` |

The likely reason is that the per-body block loop has better locality in the
small dense matrices used by `dfdn_single`, while the copy cost is not dominant.
The existing RHS path is kept.
