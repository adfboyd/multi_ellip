# Multibody Convergence Sweep

Focused two-body impulse/no-projection study for the paper convergence and
energy-drift story. The run matrix is short-time (`t=10`) and uses
self-convergence, not projection or exact-added-mass references.

## What This Sweep Tests

Primary periodic convergence block:

- shapes: `1:0.7:0.7` spheroid and `1:0.8:0.6` triaxial ellipsoid;
- density and energy ratio: `rho=4`, `E=1`;
- separations: `3`, `5`, and `8`;
- temporal convergence: `ndiv=3`, `dt = 0.2, 0.1, 0.05, 0.025`;
- grid convergence: `dt=0.05`, `ndiv = 1, 2, 3, 4`.

This is the clean paper dataset: high-density, regular dynamics, short enough
that endpoint self-convergence should be interpretable.

Stress/energy-drift block:

- shape: `1:0.7:0.7` spheroid;
- densities: `rho=1`, `0.1`;
- energy ratio: `E=0.25`;
- separations: `3` and `8`;
- temporal convergence: `ndiv=2`, `dt = 0.1, 0.05, 0.025, 0.0125`;
- grid convergence: `dt=0.025`, `ndiv = 1, 2, 3`.

This is not the primary convergence proof. It checks the known close-contact
energy-drift regime against a far-separation control.

The generator de-duplicates overlapping temporal/grid reference cases. The
current matrix is 66 cases.

## ARCHER2 Launch

From a checked-out repository on ARCHER2:

```bash
cd /work/e643/e643/adfboyd/multi_ellip
git pull
bash studies/multibody_convergence_sweep/submit_archer2_multibody_convergence_packed.sh
```

The default packed job uses one standard node for 24 hours and runs 32 cases at
once with 8 Rayon threads per case:

```text
CASES_CONCURRENT=32
THREADS_PER_CASE=8
```

Useful variants:

```bash
# Smoke test only a few cases.
MAX_CASES=4 WALLTIME=00:30:00 bash studies/multibody_convergence_sweep/submit_archer2_multibody_convergence_packed.sh

# More conservative packing.
CASES_CONCURRENT=16 THREADS_PER_CASE=16 bash studies/multibody_convergence_sweep/submit_archer2_multibody_convergence_packed.sh

# Resume after walltime or cancellation. Completed outputs are skipped.
bash studies/multibody_convergence_sweep/submit_archer2_multibody_convergence_packed.sh
```

Progress:

```bash
tail -f studies/multibody_convergence_sweep/archer2_packed_progress.log
```

Outputs:

```text
studies/multibody_convergence_sweep/runs/<case>/
```

Each run directory contains `input.txt`, `run.log`, and
`multiple_body_complete.dat`.

## Postprocessing

After the job finishes:

```bash
python3 studies/multibody_convergence_sweep/analyze_multibody_convergence_sweep.py
```

This writes summary CSVs and plots under
`studies/multibody_convergence_sweep/figures/`.
