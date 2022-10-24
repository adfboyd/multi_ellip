use serde::Serialize;

#[derive(Debug, Copy, Clone, Serialize)]
#[serde(rename_all = "PascalCase")]
pub struct Fluid {
    pub density: f64,
    pub kinetic_energy: f64,
}
