# Development record

This is the shared memory file for solver development. It records the main
methods attempted, what they were meant to fix, what succeeded, and what should
not be retried without new evidence.

## Current working picture

The code now has two conceptually different solver routes:

- **Impulse method**: the practical production method. It is comparatively
  fast, conserves total linear momentum very well, and has good single-body
  behaviour after the angular-transport fix. In multibody close-contact cases
  it can show a timestep-insensitive energy drift, especially at low density.
- **Variational midpoint method**: the physically determined reference method.
  It solves a discrete action problem and conserves the associated discrete
  Noether momentum. It is much more expensive, but it gives the cleanest target
  for understanding what the impulse method is missing.

The current strategy is **not** to use energy or momentum projection as a
production fix. Projection can improve a scalar diagnostic, but it does not by
itself make an under-modelled multibody impulse exchange physically determined.
The productive direction has been to compare against the variational reference
and add physically motivated reduced-action terms where they can be justified.

## Important conceptual lessons

### Continuous endpoint diagnostics are not always the right invariant

For the variational scheme, the old endpoint `pcon_*` and `hcon_*` columns are
not the invariant to judge at finite timestep. The correct invariant is the
discrete Noether momentum of the discrete action. This matters because the
variational method can look imperfect in endpoint continuous diagnostics while
still preserving the momentum associated with the actual discrete equations.

See: `studies/impulse_variational_comparison/NOTES.md`.

### The raw multibody impulse solve is not fully determined

The single-body impulse problem is effectively determined by the body's
velocity, angular velocity, BEM solve, and the body inertia/added-fluid
interaction. For multiple bodies, global linear and angular impulse are the
symmetry invariants, but the current impulse update effectively assigns
per-body continuous impulse candidates. Those per-body exchanges are not
uniquely fixed by the same symmetries. This is why a correction that merely
forces a global scalar/momentum condition is suspicious unless it follows from
a discrete action or another physical closure.

This is also why close-contact cases are the stress test: multibody added-fluid
coupling becomes strong, and the missing exchange model becomes visible as
large, timestep-insensitive energy drift.

## Successful fixes and retained methods

### Angular transport term in the impulse balance

Problem: single-body impulse runs had large energy oscillations.

Fix: include the missing angular transport term for origin angular momentum,
the rotating-frame `omega x L` contribution.

Outcome:

- single-body impulse energy oscillation was fixed;
- single-body drift at `ndiv=2`, `dt=0.05`, `t=50` stayed small across
  densities: about `0.011%` at `rho=4` to `0.147%` at `rho=0.1`;
- the old runtime toggle that could remove this term was deleted, because it
  represented an incomplete physical model.

Status: **production fix**.

### Implicit-midpoint rotation update

Problem: the orientation update contributed to timestep/grid-sensitive energy
behaviour and was not aligned with the midpoint structure used elsewhere.

Fix: make the rotation update implicit-midpoint rather than the earlier
explicit-style update.

Outcome:

- validation cases with the implicit-midpoint binary completed;
- the change was committed and merged into the main development line;
- subsequent studies use this as the baseline rotation update.

Status: **production fix**.

### BEM/refactor work

Problem: the solver code had grown around shared mutable state and mode toggles,
making it harder to reason about impulse/force solves and later variational
work.

Fixes retained:

- `BemSolver` owns `Simulation` directly rather than through
  `Arc<Mutex<Simulation>>`;
- solve mode is explicit internally rather than toggling
  `Simulation.impulse_mode`;
- removed stale `strong_couple` and old prototype blocks;
- split/refactored ODE/BEM responsibilities so the code is easier to navigate.

Outcome: no intended equation change; this was structural cleanup to make later
solver development safer.

Status: **production cleanup**.

### NaN kill/abort behaviour

Problem: very low-density runs could continue after producing NaNs, wasting
time and producing unusable outputs.

Fix: add checks so runs abort when non-finite state/output appears.

Outcome: failed very-low-density cases terminate more quickly and visibly.

Status: **production safety feature**.

### Exact singular treatment / BEM integration fixes

Problem: convergence and added-mass validation exposed BEM integration
inconsistencies, including singular-treatment issues.

Fix: repair the singular handling and add exact/validation studies around
surface discretisation, sphere/ellipsoid checks, and energy tests.

Outcome:

- the BEM validation state improved;
- without exact geometry, surface/grid convergence still plateaued in some
  metrics, so geometry error remains a real limit;
- exact ellipsoid geometry and higher-order integration remain plausible
  accuracy directions, but they are larger projects.

Status: **partial production fix, further accuracy work possible**.

### Variational midpoint reference solver

Problem: the impulse method is fast but not fully determined for multibody
energy exchange.

Fix: implement a discrete-action/variational midpoint route. It solves the
coupled body-fluid discrete residual and tracks the discrete Noether momentum.

Outcome:

- gives the physically clean reference for short multibody comparisons;
- discrete momentum diagnostics are conserved to small tolerances in reference
  runs;
- endpoint continuous KE is not the fundamental invariant, but it remains a
  useful diagnostic for comparisons.

Cost:

- initially extremely expensive because residual/Jacobian evaluations require
  many BEM solves;
- after Broyden/Jacobian reuse optimisations, a close three-body reference
  dropped from about `30.9 s/step` to about `6.0 s/step` while matching the
  full Newton output to tiny tolerances over the smoke test.

Status: **reference method, not production sweep method**.

See: `studies/impulse_variational_comparison/NOTES.md`.

### Variational speedups retained

Retained changes:

- Broyden updates and cross-step Jacobian reuse;
- fallback finite-difference Jacobian reuse;
- opt-in discrete momentum diagnostic via `variational_momentum_diagnostic=1`;
- lighter restore path for finite-difference probes;
- action evaluation counters in logs/diagnostics;
- optional relaxed tolerance for cheaper near-reference trajectories.

Outcome:

- close three-body reference with Broyden/reuse matched full Newton output with
  position RMS around `1e-11` in the smoke comparison;
- disabling the expensive momentum diagnostic reduced runtime while preserving
  physical output to tiny tolerances;
- `variational_tol=1e-7` is a reasonable cheaper near-reference option when
  exact reference strictness is not needed.

Status: **retained reference-solver optimisation**.

Detailed notes: `studies/multibody_energy_balance/pair_metric_correction.md`.

### Pairwise metric-correction prototype

Problem: raw impulse has close-contact, timestep-insensitive energy drift.

Idea: approximate the missing configuration work from the eliminated fluid
action with a reduced pairwise metric term, rather than projecting energy after
the fact.

Retained implementation:

- disabled by default;
- `impulse_pair_metric_correction=1`;
- `impulse_pair_metric_mode=1` is the current pairwise translational
  discrete-gradient mode;
- `impulse_pair_metric_linear_scale=1.0` follows from the reduced work identity;
- angular finite-difference component exists diagnostically but defaults to
  zero because it was not robust.

Main findings:

- positive translational scale has the right sign;
- negative scale is destabilising;
- scale `1.0` reduces close-contact energy drift and moves short two-body
  trajectories toward the variational reference;
- calibrated scales around `1.2--1.4` can improve short-time trajectory match in
  the close spheroid two-body reference, but this is not a universal closure;
- the correction still leaves timestep-insensitive drift, so it is not a full
  replacement for the variational action.

Status: **physically motivated experimental reduced model, disabled by default**.

Detailed notes: `studies/multibody_energy_balance/pair_metric_correction.md`.

### Global internal discrete-gradient prototype

Problem: pairwise correction scales with active pairs; many-body cases might
need a cheaper reduced approximation.

Idea: sample two global internal translations per step, remove common
translation, and apply an internal discrete-gradient load with zero net force
and constrained net torque.

Outcome:

- for two bodies it collapses algebraically toward the pairwise relative
  translation construction;
- for three bodies it is cheaper than pairwise mode when all pairs are active;
- short three-body variational comparison did not show better accuracy than
  pairwise mode.

Status: **experimental cheaper many-body approximation, not an accuracy
replacement for pairwise mode**.

### Recurrence classification and paper sweep tooling

Problem: the qualitative behaviour needs objective classification across large
sweeps.

Implemented:

- recurrence/spectral classification for two-body sweeps;
- use both bodies as repeat observations where appropriate;
- use full quaternion orientation metric for triaxial ellipsoids, with `q` and
  `-q` treated as the same orientation;
- use the axisymmetric-axis metric for spheroids;
- generate behaviour maps, broadband recurrence score maps, energy drift maps,
  minimum separation maps, slice plots, scatter plots, dashboards, and docs
  mirrors.

Outcome:

- full two-body paper sweep: 1440 output files present;
- spheroid `1:0.7:0.7`: 137/144 groups have all five repeats complete;
- ellipsoid `1:0.8:0.6`: 130/144 groups have all five repeats complete;
- missing/short repeats concentrate in low-density, close-separation cases;
- five repeats expose more mixed behaviour than the earlier partial data.

Status: **retained analysis tooling**.

See:

- `studies/two_body_paper_sweeps/analysis_overview/NOTES.md`;
- `docs/paper_figures/section4_two_body/`;
- `studies/departing_sphericity_sweep/analyze_departing_sphericity_outputs.py`.

## Methods tried and not retained as production fixes

### Energy/momentum projection

Idea: correct the post-step state to enforce energy or momentum diagnostics.

Outcome:

- can improve selected diagnostics;
- not accepted as a production multibody fix because it is not a determined
  physical closure for the missing body-body fluid exchange;
- particularly unwelcome where it masks an underdetermined impulse solve.

Status: **avoid unless explicitly running a diagnostic**.

### Full Hamiltonian/global-action development branch

Idea: move fully to a Hamiltonian/global-action formulation for multibody
physics.

Outcome:

- conceptually the clean direction;
- development showed that the fully determined variational route is the more
  concrete reference implementation already available;
- the Hamiltonian branch was parked in favour of improving the impulse regime
  and using the variational solver for reference comparisons.

Status: **parked**.

### Existing metric-gradient and pressure-load candidates

Idea: use existing finite-difference fluid-energy gradient or quadratic
pressure-load candidates to cancel the impulse variational defect.

Outcome:

- defect alignment was inconsistent;
- metric-gradient alignment was often negative;
- pressure-load alignment changed sign over a short run;
- turning them on directly would be projection-like rather than a stable
  physical correction.

Status: **ruled out**.

See: `studies/multibody_energy_balance/ruled_out.md`.

### Off-body near-singular cross-quadrature refinement

Idea: close-contact energy drift might come from under-resolved off-body
near-singular BEM interactions.

Outcome:

- forced near-rule subdivision on cross interactions changed energy only at
  roundoff level;
- runtime became about `10--30x` slower in the test;
- close-contact drift is not explained by ordinary off-body cross-quadrature
  resolution.

Status: **ruled out**.

See: `studies/multibody_energy_balance/ruled_out.md`.

### Refreshing pair metric force every fixed-point iteration

Idea: recompute the pair metric force inside each impulse fixed-point iterate.

Outcome:

- about `3x` slower in the close-pair test;
- did not improve energy drift or trajectory agreement;
- frozen once-per-step pair force remains the better reduced approximation.

Status: **ruled out**.

### Warm-starting the implicit torque iterate

Idea: start the next fixed-point solve with the previous step's torque.

Outcome:

- about `2%` speed gain;
- changed trajectories by more than justified by the tiny speedup;
- zero initial torque kept.

Status: **ruled out**.

### Surface-clearance weighting for pair activation

Idea: use ellipsoid surface gap along the line of centres instead of centre
distance for pair-metric cutoff weights.

Outcome:

- geometrically plausible;
- did not improve trajectory against the variational reference;
- sometimes increased long-run separation and per-body angular drift.

Status: **not retained**.

### Energy-only samples for full global metric correction

Idea: use `kinetic_energy_only()` for the older full global
`impulse_metric_correction` finite-difference samples.

Outcome:

- diagnostics unchanged;
- runtime was slower in the tested path, likely due to worse warm-start
  sequence;
- the global correction remains diagnostic.

Status: **not retained**.

### Direct combined-array RHS boundary condition loop

Idea: avoid splitting combined BEM arrays into per-body blocks for RHS
evaluation.

Outcome:

- output matched baseline;
- timing was not faster and was slower in the clearer `ndiv=3` test;
- existing block RHS likely has better locality.

Status: **not retained**.

### Variational forward differences

Idea: replace central finite differences in variational gradients with forward
differences.

Outcome:

- fewer BEM evaluations in a timing build;
- release runtime improvement small/noisy;
- trajectory changed measurably unless epsilon was tightened enough to remove
  the speed advantage.

Status: **removed; keep central differences**.

See: `docs/rejected_solver_optimizations.md`.

### Endpoint BEM result cache in variational route

Idea: reuse accepted endpoint BEM result as next step's start BEM result.

Outcome:

- slower in benchmark;
- changed output, likely by disturbing BEM warm-start convergence;
- removed.

Status: **removed**.

### Disabling variational impulse diagnostics in output

Idea: skip impulse solve used to populate variational diagnostic output columns.

Outcome:

- made output less useful;
- no reliable speedup;
- removed.

Status: **removed**.

### KE-only restore inside variational probes

Idea: restore BEM warm-start state after variational probes with
`kinetic_energy_only()`.

Outcome:

- byte-identical output in the short benchmark;
- runtime worse;
- removed.

Status: **removed**.

## Accuracy and convergence studies

### Single-body dynamics

The high-density/low-energy periodic single-body cases are the preferred
convergence tests. Self-convergence of centroid and orientation at shorter
horizons (`t=2`, `t=5`, and `t=10` depending on cost) is more useful than
comparing only to an analytic reference when discretisation and exact geometry
are both in play.

Key lesson: decreasing `dt` gives clearer convergence than simply increasing
`ndiv` once geometry/discretisation error plateaus.

### Grid visualisation and surface norm error

A script was added to visualise the triangulated grid against the analytic
ellipsoid and plot surface-normal error. This helped separate solver error from
surface geometry/discretisation error.

Key lesson: higher `ndiv` is not a magic fix if the geometry representation and
surface integration order dominate.

### Exact ellipsoid geometry / higher-order integration

Exact ellipsoid geometry would mean keeping the computational mesh as a
quadrature/discretisation device while evaluating positions, normals, and
surface measures on the true ellipsoid rather than on a flat triangle
approximation. This is a promising accuracy route, but it is a real BEM
geometry/integration project, not a small toggle.

Status: **future accuracy direction**.

### Sphere checks

Sphere cases are useful because the added-mass behaviour should converge
quickly and symmetries remove many orientation complications. They are good
regression tests for BEM consistency and singular integration, but they do not
stress the triaxial nonintegrable rotational dynamics.

## Large study infrastructure

### Two-body paper sweeps

Implemented Archer2 packed-job scripts so many cases can run sequentially or in
controlled parallel groups without exceeding queue limits. The final paper
sweep varies:

- shape: spheroid `1:0.7:0.7`, triaxial ellipsoid `1:0.8:0.6`;
- density including `rho=10`;
- energy ratio from roughly `0.2` to `30`;
- separation including `11`;
- five repeats with randomised initial orientations and velocity/angular
  directions.

Status: **retained study infrastructure**.

### Departing-sphericity sweeps

Implemented sweeps at `rho=1`, `E=0.5`, separation `8`, with 10 repeats across:

- oblate spheroids;
- prolate spheroids;
- triaxial ellipsoids;
- homogeneous, body-plus-sphere, and body-plus-`eps=1` pair suites.

Analysis uses axisymmetric metrics for spheroids and quaternion metrics for
triaxial cases.

Status: **retained study infrastructure**.

## Current recommendations

For production-style impulse studies:

- use the raw impulse method with implicit-midpoint rotation;
- do not enable projection unless deliberately running a diagnostic;
- use the pair metric correction only as an explicitly labelled experimental
  reduced-action model;
- treat low-density close-contact results carefully, especially where runs are
  short/missing or energy drift is huge.

For validation:

- use high-density/low-energy single-body periodic cases for clean
  convergence;
- use short variational references for multibody close-contact comparisons;
- judge variational runs by discrete Noether momentum, not only endpoint
  continuous momentum/energy columns;
- keep notes and plots committed so cross-model work does not depend on chat
  history.

For future accuracy work:

- exact ellipsoid geometry and higher-order surface integration are the most
  credible route to improving grid-convergence limits;
- the reduced pair metric model is the most credible route to improving impulse
  multibody close-contact drift without paying full variational cost;
- a complete global-action/variational formulation remains the reference for
  physical correctness, but it is too expensive for full sweeps without further
  algorithmic advances.

