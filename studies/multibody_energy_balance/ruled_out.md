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
