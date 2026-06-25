# Two-body paper sweep analysis notes

This analysis uses the preferred orientation recurrence metrics:

- `spheroid_1_0p7_0p7`: axisymmetric-axis metric, from `classification_spheroid_1_0p7_0p7_axis`.
- `ellipsoid_1_0p8_0p6`: full quaternion metric, from `classification_ellipsoid_1_0p8_0p6_quaternion`.

The manifests request five repeats per parameter group. The raw outputs present in this checkout contain 1440 output files across the two shapes. Most parameter groups have all five repeats complete, but a small number are short or missing according to the expected `tend / dt + 1` row count:

- `spheroid_1_0p7_0p7`: 137/144 groups have all five complete repeats.
- `ellipsoid_1_0p8_0p6`: 130/144 groups have all five complete repeats.

The `incomplete_map.png` and `group_metrics.csv` outputs record the current repeat coverage.
