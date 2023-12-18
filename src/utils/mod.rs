use std::path::PathBuf;

#[derive(Debug)]
pub struct SimName {
    rk4_path: PathBuf,
    pcdm_path: PathBuf,
    lab_path: PathBuf,
    olab_path: PathBuf,
    complete_path: PathBuf,
}

impl SimName {
    pub fn new<P: Into<PathBuf>>(path: P) -> Self {
        let path_base = path.into();
        Self {
            rk4_path: path_base.join("single_body_rk4.dat"),
            pcdm_path: path_base.join("single_body_pcdm.dat"),
            lab_path: path_base.join("single_body_lab.dat"),
            olab_path: path_base.join("single_body_olab.dat"),
            complete_path: path_base.join("multiple_body_complete.dat"),
        }
    }
    pub fn rk4_path(&self) -> &PathBuf {
        &self.rk4_path
    }
    pub fn pcdm_path(&self) -> &PathBuf {
        &self.pcdm_path
    }
    pub fn lab_path(&self) -> &PathBuf {
        &self.lab_path
    }
    pub fn olab_path(&self) -> &PathBuf {
        &self.olab_path
    }
    pub fn complete_path(&self) -> &PathBuf {
        &self.complete_path
    }
}
