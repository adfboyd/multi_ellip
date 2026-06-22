# Paper Figure Exports

Generated/exported figure assets for the paper draft.  These are copied here so
they are available from the remote repository without relying on ignored study
run directories.

## Section 3 Solver and Single-Body Figures

- `section3_clean/`: curated paper-ready Section 3 exports assembled from the
  study outputs:
  - `section3_grid_discretisation.png`
  - `section3_bem_convergence.png`
  - `section3_geometric_error_convergence.png`
  - `section3_recurrence_examples.png`
  - `section3_marker_orbit_examples.png`
- `default_grid_overlay_panel.png`: ellipsoid surface discretisation for
  increasing `ndiv`.
- `default_grid_error_panel.png`: local geometric surface error.
- `default_grid_error_summary.png` and `default_grid_error_summary.csv`:
  geometric surface error convergence.
- `default_grid_surface_norm_summary.png`: surface normalisation / surface-norm
  convergence metric.
- `default_grid_normal_error_panel.png` and
  `default_grid_normal_error_summary.png`: normal-vector error diagnostics.
- `exact_singular_convergence.png`: BEM convergence against the analytic
  ellipsoid solution.
- `exact_singular_energy_drift.png`: short energy-drift comparison for exact
  singular geometry cases.
- `exact_singular_convergence_orders.csv` and
  `exact_singular_energy_summary.csv`: source metrics for the exact-singular
  plots.
- `single_body_triaxial_impulse_nd2_dashboard.png`: triaxial single-body
  impulse run summary at `ndiv=2`.
- `single_body_triaxial_impulse_nd3_dashboard.png`: triaxial single-body
  impulse run summary at `ndiv=3`.
- `single_body_spheroid_impulse_nd2_dashboard.png`: spheroid single-body
  impulse run summary at `ndiv=2`.
- `single_body_triaxial_exact_dashboard.png`: exact-added-mass triaxial
  reference run summary.
- `single_body_spheroid_exact_dashboard.png`: exact-added-mass spheroid
  reference run summary.
- `single_body_triaxial_orientation_nd2.png`,
  `single_body_triaxial_orientation_nd3.png`, and
  `single_body_spheroid_orientation_nd2.png`: orientation-marker panels from
  the impulse solver.
- `single_body_triaxial_exact_orientation.png` and
  `single_body_spheroid_exact_orientation.png`: exact-added-mass orientation
  marker panels.
- `representative_single_body_orbits/`: compact 3D marker-orbit figures using
  one representative regular/quasiperiodic trajectory and, where present in the
  current data, one representative chaotic-like trajectory.  The spheroid
  single-body data currently contains regular cases only.
- `ke_ratio_recurrence_current_comparison/`: recurrence plots and metrics for
  current triaxial/spheroid single-body runs and exact references.

## Section 4 Setup Figure

- `section4_setup/section4_two_body_setup_schematic.png`: schematic of the
  two-body initial configuration with different orientations, initial
  separation, parallel translational velocities, and independent angular
  velocities.

Two-body sweep figures are intentionally not included in this export.
