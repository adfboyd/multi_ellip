# Rejected solver optimization attempts

These are implementation ideas that were tested and then removed or left disabled because they did not improve the solver in a useful way.

## Variational forward differences

Idea: replace central finite differences in the variational discrete-action gradients with forward differences.

Result: the timing-feature build confirmed fewer BEM evaluations on the short two-body variational benchmark: 1161 solves dropped to 655 solves over five timesteps. However, the normal release runtime improvement was small and noisy, while the trajectory changed measurably. With `hamiltonian_coupled_eps=1e-5`, the max total-KE difference versus central differences was about `1.5e-2` over the short benchmark. Tightening to `1e-6` reduced the difference to about `1.5e-3`, but removed the useful speed advantage in the tested run.

Decision: removed. Keep central differences for the variational residual so the method remains the more accurate, symmetric finite-difference implementation.

## Endpoint BEM result cache

Idea: reuse the BEM solve already performed at the accepted variational endpoint as the next step's start-of-step BEM result.

Result: this was slower on the short nd2 benchmark and changed the numerical output. The likely cause is that it altered the BEM warm-start sequence enough to hurt GMRES convergence or finite-difference consistency.

Decision: removed. Preserve the established start/end BEM solve sequence.

## Disabling variational impulse diagnostics in output

Idea: add a switch to skip the impulse solve used to populate `lfluid`, `lambdafluid`, `pcon`, and `hcon` output columns for variational runs.

Result: it made output less useful and did not produce a reliable steady-state speedup. Some apparent total-time wins were just first-step timing noise.

Decision: removed. Keep the diagnostic columns populated.

## KE-only restore solve inside variational probes

Idea: when `variational_energy_only_lagrangian=1`, restore the BEM warm-start state after finite-difference probes using `kinetic_energy_only()` instead of `impulse()`.

Result: output remained byte-identical, but runtime was worse on the short benchmark. The cheaper-looking solve was not a cheaper warm-start strategy in practice.

Decision: removed. Restore probe state with the normal impulse solve.

## Kept optimization: KE-only action evaluation

Idea: when evaluating the variational discrete action, solve only for fluid kinetic energy and skip force/impulse surface quadratures.

Result: byte-identical output on the short benchmark, with a useful measured speedup in earlier tests.

Decision: keep as `variational_energy_only_lagrangian=1`.

## Removed option: impulse transport toggle

Idea: keep `impulse_transport=0` as a debugging switch that drops the rotating-frame
`omega x L` transport term from the force-mode Lamb impulse balance.

Result: the missing transport term was the cause of the earlier single-body impulse
energy oscillation. It is part of the physical force/torque balance, so keeping a
runtime switch made it too easy to run an incomplete model accidentally.

Decision: removed. The transport term is always included.

## Removed alias: impulse metric project

Idea: keep `impulse_metric_project` as an alias for the impulse internal-load
constraint used by old metric-correction experiments.

Result: the name hid what the option actually did and overlapped with the
unwelcome projection experiments. The remaining implementation is diagnostic /
experimental impulse correction code, not part of the production impulse scheme.

Decision: removed the alias. Use `impulse_internal_load_constraint` only when
deliberately running that diagnostic path.
