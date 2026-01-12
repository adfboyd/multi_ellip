# multi_ellip

## Features

- `lapack`: enable LAPACK-backed LU/solve via `nalgebra-lapack`.
- `lapack-netlib`: build LAPACK/BLAS from Netlib sources (adds `lapack`).
- `lapack-system`: link against system-provided LAPACK/BLAS (adds `lapack`).
- `timing`: enable per-stage timing prints in the BEM solve.

## Build examples

- Default build (no optional features):
  `cargo build --release`
- ARCHER2 (LibSci LAPACK):
  `cargo build --release --features lapack,lapack-system`
- Enable timing prints:
  `cargo build --release --features timing`
