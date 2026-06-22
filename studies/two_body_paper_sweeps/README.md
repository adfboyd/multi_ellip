# Two-Body Paper Sweeps

Production two-body impulse sweeps for Section 4-style recurrence/behaviour
maps. These are separate from the older prototype sweep directories and exclude
spheres, since those have already been run.

## Defaults

- shapes: `1:0.7:0.7` spheroid and `1:0.8:0.6` triaxial ellipsoid;
- densities: `rho = 10, 4, 1, 0.3, 0.1, 0.03`;
- energy ratios: `E = 0.2, 0.55, 1.5, 4, 11, 30`;
- initial separations: `3, 5, 8, 11`;
- repeats: `2`;
- method: `impulse_scheme=1`, `energy_projection=0`;
- mesh/time: `ndiv=2`, `dt=0.05`, `t=100`;
- angular-speed normalisation: 100 rotations over the run;
- initial translational velocities are parallel; velocity and angular-velocity
  directions are deterministic-random per case.

This gives 288 cases per shape, or 576 cases for `SHAPE=all`.

## ARCHER2 Launch

From a checked-out repository on ARCHER2:

```bash
cd /work/e643/e643/adfboyd/multi_ellip
git pull

# Run spheroids first.
bash studies/two_body_paper_sweeps/submit_archer2_two_body_paper_packed.sh

# Run triaxial ellipsoids separately.
SHAPE=ellipsoid_1_0p8_0p6 bash studies/two_body_paper_sweeps/submit_archer2_two_body_paper_packed.sh
```

The launcher submits one resumable packed job. By default it runs 32 cases at
once with 8 Rayon threads each.

Useful variants:

```bash
# Smoke test.
MAX_CASES=8 WALLTIME=01:00:00 bash studies/two_body_paper_sweeps/submit_archer2_two_body_paper_packed.sh

# More conservative packing.
CASES_CONCURRENT=16 THREADS_PER_CASE=16 bash studies/two_body_paper_sweeps/submit_archer2_two_body_paper_packed.sh

# Submit both shapes in one manifest if queue time is available.
SHAPE=all bash studies/two_body_paper_sweeps/submit_archer2_two_body_paper_packed.sh
```

Progress logs:

```bash
tail -f studies/two_body_paper_sweeps/archer2_spheroid_1_0p7_0p7_progress.log
tail -f studies/two_body_paper_sweeps/archer2_ellipsoid_1_0p8_0p6_progress.log
```

Outputs:

```text
studies/two_body_paper_sweeps/runs/<shape>/<case>/
```

Each run directory contains `input.txt`, `run.log`, and
`multiple_body_complete.dat`. Re-submitting skips completed outputs unless
`RERUN=1` is set.

## Postprocessing

Use the existing two-body postprocessing scripts by pointing them at one of the
new manifests/run roots, for example:

```bash
python3 studies/two_body_parameter_sweep/classify_two_body_dynamics.py \
  --manifest studies/two_body_paper_sweeps/manifest_spheroid_1_0p7_0p7.csv \
  --out-dir studies/two_body_paper_sweeps/classification_spheroid_1_0p7_0p7

python3 studies/two_body_parameter_sweep/analyze_two_body_sweep_results.py \
  --run-root studies/two_body_paper_sweeps/runs/spheroid_1_0p7_0p7 \
  --manifest studies/two_body_paper_sweeps/manifest_spheroid_1_0p7_0p7.csv \
  --out-dir studies/two_body_paper_sweeps/analysis_spheroid_1_0p7_0p7
```

If the current postprocessors need a small adapter for these nested run roots,
add it after the ARCHER2 outputs are back rather than before spending compute.
