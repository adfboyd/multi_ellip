# Departing-Sphericity Two-Body Sweep

This study mirrors the departing-sphericity cases described in the paper draft, but uses the current impulse solver without energy or momentum projection.

Fixed parameters:

- `rho = 1`
- `energy_ratio = 0.5`
- initial centre separation `= 8`
- `ndiv = 2`
- `dt = 0.05`
- `tend = 100`
- 100 nominal rotations over the run
- 10 deterministic-random repeats per parameter point

Shape families:

- `spheroid_prolate`: axes proportional to `(1 + eps)^2 : (1 + eps)^-1 : (1 + eps)^-1`
- `spheroid_oblate`: axes proportional to `(1 + eps) : (1 + eps) : (1 + eps)^-2`
- `triaxial`: axes proportional to `(1 + eps) : 1 : (1 + eps)^-1`

All shapes are already volume-preserving in these definitions. The solver still receives `req = 1`.

Pair suites:

- `homogeneous`: both bodies use the same `eps`
- `heterogeneous_sphere`: body 1 uses `eps`, body 2 is spherical
- `heterogeneous_aspherical`: body 1 uses `eps`, body 2 uses `eps = 1`

The epsilon grid is:

```text
0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1
```

The full matrix is `3 families * 3 suites * 10 eps values * 10 repeats = 900 cases`.

## Local Dry Run

```bash
python studies/departing_sphericity_sweep/setup_departing_sphericity_sweep.py
python studies/departing_sphericity_sweep/setup_departing_sphericity_sweep.py --family triaxial --suite homogeneous
```

## Archer2

Run the wrapper with `bash`, not `sbatch`. It generates inputs/manifests and submits the packed Slurm job with default batch parameters.

Full study:

```bash
FAMILY=all SUITE=all bash studies/departing_sphericity_sweep/submit_archer2_departing_sphericity_packed.sh
```

Small test:

```bash
MAX_CASES=8 WALLTIME=01:00:00 FAMILY=triaxial SUITE=homogeneous bash studies/departing_sphericity_sweep/submit_archer2_departing_sphericity_packed.sh
```

Useful selectors:

```bash
FAMILY=spheroid_prolate bash studies/departing_sphericity_sweep/submit_archer2_departing_sphericity_packed.sh
FAMILY=spheroid_oblate bash studies/departing_sphericity_sweep/submit_archer2_departing_sphericity_packed.sh
FAMILY=triaxial bash studies/departing_sphericity_sweep/submit_archer2_departing_sphericity_packed.sh
SUITE=homogeneous bash studies/departing_sphericity_sweep/submit_archer2_departing_sphericity_packed.sh
SUITE=heterogeneous_sphere bash studies/departing_sphericity_sweep/submit_archer2_departing_sphericity_packed.sh
SUITE=heterogeneous_aspherical bash studies/departing_sphericity_sweep/submit_archer2_departing_sphericity_packed.sh
```

Progress log:

```bash
tail -f studies/departing_sphericity_sweep/archer2_all_all_progress.log
```

Resume by running the same wrapper command again. Completed cases are skipped unless `RERUN=1` is set.
