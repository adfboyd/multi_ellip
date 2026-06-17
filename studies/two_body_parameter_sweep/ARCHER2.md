# ARCHER2 two-body parameter sweep

This study can be launched from a checked-out copy of `multi_ellip` on ARCHER2.
The default helper generates the two-body manifest with repo-relative paths, then
submits one resumable packed job. The packed job builds the code once and runs
multiple cases concurrently inside a single node allocation.

```bash
cd /work/e643/e643/adfboyd/multi_ellip
git pull
bash studies/two_body_parameter_sweep/submit_archer2_two_body_packed.sh
```

By default the packed launcher runs 32 cases at once with 8 Rayon threads per
case. That is usually a better use of a full node for `ndiv=2` than assigning
hundreds of threads to one small BEM solve.

For a short-queue smoke test, submit one case with a 20 minute walltime:

```bash
bash studies/two_body_parameter_sweep/submit_archer2_two_body_short_test.sh
```

This defaults to `--partition=standard`, `--qos=short`, `--time=00:20:00`, and
`MAX_CASES=1`. If the local queue naming differs, override it at submit time:

```bash
SHORT_PARTITION=standard SHORT_QOS=standard bash studies/two_body_parameter_sweep/submit_archer2_two_body_short_test.sh
```

Useful options:

```bash
# Request a shorter walltime. Re-submit the same command to resume.
WALLTIME=12:00:00 bash studies/two_body_parameter_sweep/submit_archer2_two_body_packed.sh

# Change the node packing.
CASES_CONCURRENT=16 THREADS_PER_CASE=16 bash studies/two_body_parameter_sweep/submit_archer2_two_body_packed.sh

# Smoke-test the packed path on the short QoS.
QOS=short WALLTIME=00:20:00 MAX_CASES=32 bash studies/two_body_parameter_sweep/submit_archer2_two_body_packed.sh

# Run at most ten new cases in this job.
MAX_CASES=10 bash studies/two_body_parameter_sweep/submit_archer2_two_body_packed.sh

# Start from a later manifest row.
START_INDEX=25 bash studies/two_body_parameter_sweep/submit_archer2_two_body_packed.sh

# Ignore existing complete outputs and run everything again.
RERUN=1 bash studies/two_body_parameter_sweep/submit_archer2_two_body_packed.sh
```

Outputs are written under:

```text
studies/two_body_parameter_sweep/runs/<case>/
```

Each run directory contains the solver stdout in `run.log` and the trajectory data
in `multiple_body_complete.dat`. The packed script skips a case when that data file
already exists with at least the expected number of rows, unless `RERUN=1` is set.
This makes the job resumable: if it runs out of walltime or is cancelled, submit it
again and completed cases will be skipped.

Progress is appended to:

```text
studies/two_body_parameter_sweep/archer2_packed_progress.log
```

The purely consecutive serial launcher is still available:

```bash
bash studies/two_body_parameter_sweep/submit_archer2_two_body_serial.sh
```

The older array launcher is still available:

```bash
bash studies/two_body_parameter_sweep/submit_archer2_two_body_study.sh
```

It submits a one-node LAPACK build job, then one array task per manifest row. Use
this only when the queue policy allows that many submitted jobs.

The array job intentionally runs the release binary directly rather than via
`srun`, matching `submit_w_lapack.slurm`. With `--tasks-per-node=128`, a plain
`srun` command can start many copies of the same serial/Rayon executable inside a
single array element.
