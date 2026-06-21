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
