# Two-body paper sweep analysis notes

This analysis uses the preferred orientation recurrence metrics:

- `spheroid_1_0p7_0p7`: axisymmetric-axis metric, from `classification_spheroid_1_0p7_0p7_axis`.
- `ellipsoid_1_0p8_0p6`: full quaternion metric, from `classification_ellipsoid_1_0p8_0p6_quaternion`.

The manifests currently request five repeats per parameter group. The raw outputs present in this checkout contain 576 completed runs across the two shapes, so most groups have two complete repeats rather than all five. The `incomplete_map.png` and `group_metrics.csv` outputs record the current repeat coverage.

The departing-sphericity sweep progress logs report completed Archer2 jobs, but the run directories and `multiple_body_complete.dat` files are not present in this checkout. Those results still need to be added before they can be post-processed.
