//! Boundary-element integration. Split into cohesive submodules; this module is
//! a facade that re-exports their public API so existing callers
//! (`use crate::bem::integ::*`) are unaffected.
//!
//! - [`assembly`]: Green's-function kernels, element interpolation/quadrature,
//!   and influence-matrix / right-hand-side assembly (the BEM solve core).
//! - [`surface`]: post-solve surface integrals (gradient interpolation,
//!   hydrodynamic forces, Lamb impulse, fluid kinetic energy).
//! - [`gradient`]: surface-gradient reconstruction (`grad_3d_*` family).

mod assembly;
mod gradient;
mod surface;

pub use assembly::*;
pub use gradient::*;
pub use surface::*;
