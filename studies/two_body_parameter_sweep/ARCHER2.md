# ARCHER2 two-body parameter sweep

This study can be launched from a checked-out copy of `multi_ellip` on ARCHER2.
The helper script generates the two-body manifest with repo-relative paths, submits
a one-node LAPACK build job, then submits one array task per case after the build
has completed.

```bash
cd /work/e643/e643/adfboyd/multi_ellip
git pull
bash studies/two_body_parameter_sweep/submit_archer2_two_body_study.sh
```

Useful options:

```bash
# Keep at most eight cases running at once.
MAX_CONCURRENT=8 bash studies/two_body_parameter_sweep/submit_archer2_two_body_study.sh

# Ignore existing complete outputs and run everything again.
RERUN=1 bash studies/two_body_parameter_sweep/submit_archer2_two_body_study.sh
```

Outputs are written under:

```text
studies/two_body_parameter_sweep/runs/<case>/
```

Each run directory contains the solver stdout in `run.log` and the trajectory data
in `multiple_body_complete.dat`. The array script skips a case when that data file
already exists with at least the expected number of rows, unless `RERUN=1` is set.

The array job intentionally runs the release binary directly rather than via
`srun`, matching `submit_w_lapack.slurm`. With `--tasks-per-node=128`, a plain
`srun` command can start many copies of the same serial/Rayon executable inside a
single array element.
