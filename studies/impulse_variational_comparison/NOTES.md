# Impulse vs variational comparison

This study compares the raw impulse step with the discrete-Lagrangian variational step on the same two-body ellipsoid case.

The important distinction is that `pcon_*` and `hcon_*` are continuous endpoint diagnostics, while `jdisc_*` is the discrete Noether momentum associated with the variational update.

## Interpretation

- Raw impulse keeps total continuous linear momentum near roundoff, but its maximum KE drift stays in a narrow band of 0.435--0.467% as `dt` is reduced.
- The variational method shows approximately first-order convergence in the endpoint KE diagnostic over this short test.
- The variational discrete Noether momentum is conserved to 2.94e-08 or better in all completed cases.
- The old continuous endpoint `pcon/hcon` columns are therefore not the correct invariant for judging the variational scheme at finite timestep.
- The practical next target is a cheaper impulse correction that reproduces the missing variational metric/configuration work, not full finite-difference variational Newton for production sweeps.

## Files

- `manifest.csv`: generated cases.
- `summary.csv`: postprocessed conservation and trajectory metrics.
- `conservation_vs_dt.png`: conservation diagnostics versus timestep.
- `impulse_vs_variational_state_error.png`: impulse trajectory difference relative to variational at matched `ndiv,dt`.

## Summary

| scheme | ndiv | dt | max KE drift (%) | max endpoint dP | max endpoint dH | max discrete dJ |
|---|---:|---:|---:|---:|---:|---:|
| impulse | 1 | 0.025 | 0.438237 | 8.12517e-09 | 1.60945e-05 | nan |
| variational | 1 | 0.025 | 0.054353 | 0.0115384 | 0.0109145 | 7.32651e-09 |
| impulse | 1 | 0.05 | 0.437533 | 2.76523e-09 | 6.43777e-05 | nan |
| variational | 1 | 0.05 | 0.108406 | 0.0216325 | 0.018233 | 1.6932e-08 |
| impulse | 1 | 0.1 | 0.434722 | 1.1439e-09 | 0.000257352 | nan |
| variational | 1 | 0.1 | 0.215717 | 0.0417562 | 0.0343567 | 2.94084e-08 |
| impulse | 2 | 0.025 | 0.467041 | 8.05586e-09 | 1.61538e-05 | nan |
| variational | 2 | 0.025 | 0.0545987 | 0.0104124 | 0.00724252 | 7.74067e-09 |
| impulse | 2 | 0.05 | 0.466331 | 2.74237e-09 | 6.46148e-05 | nan |
| variational | 2 | 0.05 | 0.108896 | 0.0207182 | 0.0148058 | 1.59084e-08 |
| impulse | 2 | 0.1 | 0.463494 | 1.14724e-09 | 0.0002583 | nan |
| variational | 2 | 0.1 | 0.216692 | 0.0410902 | 0.0310428 | 2.67554e-08 |
