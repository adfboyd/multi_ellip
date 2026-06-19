//! Exact ellipsoid surface geometry for curved BEM elements.
//!
//! The existing mesh stores six nodes per element on a subdivided sphere, then
//! scales those nodes to the ellipsoid and interpolates quadratically between
//! the scaled positions. This module keeps the same element connectivity, but
//! evaluates quadrature points by interpolating on the reference sphere,
//! re-projecting to the sphere, and then applying the exact ellipsoid map.

use nalgebra::{Matrix3, UnitQuaternion, Vector3};

#[derive(Debug, Clone, Copy)]
pub struct SurfacePoint {
    pub position: Vector3<f64>,
    pub normal: Vector3<f64>,
    pub jacobian: f64,
}

#[derive(Debug, Clone)]
pub struct ExactEllipsoidPatch {
    unit_nodes: [Vector3<f64>; 6],
    semi_axes: Vector3<f64>,
    centre: Vector3<f64>,
    orient: UnitQuaternion<f64>,
    alpha: f64,
    beta: f64,
    gamma: f64,
}

impl ExactEllipsoidPatch {
    pub fn new(
        unit_nodes: [Vector3<f64>; 6],
        semi_axes: Vector3<f64>,
        centre: Vector3<f64>,
        orient: UnitQuaternion<f64>,
        alpha: f64,
        beta: f64,
        gamma: f64,
    ) -> Self {
        let unit_nodes = unit_nodes.map(|u| u.normalize());
        Self {
            unit_nodes,
            semi_axes,
            centre,
            orient,
            alpha,
            beta,
            gamma,
        }
    }

    pub fn evaluate(&self, xi: f64, eta: f64) -> SurfacePoint {
        let (phi, dxi, deta) = quadratic_shape(self.alpha, self.beta, self.gamma, xi, eta);

        let mut raw = Vector3::zeros();
        let mut raw_xi = Vector3::zeros();
        let mut raw_eta = Vector3::zeros();
        for i in 0..6 {
            raw += self.unit_nodes[i] * phi[i];
            raw_xi += self.unit_nodes[i] * dxi[i];
            raw_eta += self.unit_nodes[i] * deta[i];
        }

        let raw_norm = raw.norm();
        assert!(
            raw_norm > 0.0,
            "degenerate exact-ellipsoid patch evaluation"
        );
        let sphere_point = raw / raw_norm;
        let projector = Matrix3::identity() - sphere_point * sphere_point.transpose();
        let sphere_xi = projector * raw_xi / raw_norm;
        let sphere_eta = projector * raw_eta / raw_norm;

        let axes = self.semi_axes;
        let scale = Matrix3::from_diagonal(&axes);
        let body_point = scale * sphere_point;
        let body_xi = scale * sphere_xi;
        let body_eta = scale * sphere_eta;

        let position = self.centre + self.orient.transform_vector(&body_point);
        let tangent_xi = self.orient.transform_vector(&body_xi);
        let tangent_eta = self.orient.transform_vector(&body_eta);

        let mut area_vec = tangent_xi.cross(&tangent_eta);
        let exact_normal = self
            .orient
            .transform_vector(&ellipsoid_normal_body(&body_point, &axes));
        if area_vec.dot(&exact_normal) < 0.0 {
            area_vec = -area_vec;
        }

        let jacobian = area_vec.norm();
        SurfacePoint {
            position,
            normal: area_vec / jacobian,
            jacobian,
        }
    }

    pub fn semi_axes(&self) -> Vector3<f64> {
        self.semi_axes
    }

    pub fn centre(&self) -> Vector3<f64> {
        self.centre
    }

    pub fn orientation(&self) -> &UnitQuaternion<f64> {
        &self.orient
    }
}

pub fn semi_axes_from_solver_shape(req: f64, shape: &Vector3<f64>) -> Vector3<f64> {
    let (a, b, c) = (shape[0], shape[1], shape[2]);
    let boa = b / a;
    let coa = c / a;
    let scale = req / (boa * coa).powf(1.0 / 3.0);
    Vector3::new(scale, scale * boa, scale * coa)
}

pub fn ellipsoid_level_set_body(point_body: &Vector3<f64>, semi_axes: &Vector3<f64>) -> f64 {
    (point_body[0] / semi_axes[0]).powi(2)
        + (point_body[1] / semi_axes[1]).powi(2)
        + (point_body[2] / semi_axes[2]).powi(2)
        - 1.0
}

pub fn ellipsoid_normal_body(point_body: &Vector3<f64>, semi_axes: &Vector3<f64>) -> Vector3<f64> {
    Vector3::new(
        point_body[0] / semi_axes[0].powi(2),
        point_body[1] / semi_axes[1].powi(2),
        point_body[2] / semi_axes[2].powi(2),
    )
    .normalize()
}

pub fn quadratic_shape(
    alpha: f64,
    beta: f64,
    gamma: f64,
    xi: f64,
    eta: f64,
) -> ([f64; 6], [f64; 6], [f64; 6]) {
    let (alc, bec, gac) = (1.0 - alpha, 1.0 - beta, 1.0 - gamma);
    let (alalc, bebec, gagac) = (alpha * alc, beta * bec, gamma * gac);

    let ph2 = xi * (xi - alpha + eta * (alpha - gamma) / gac) / alc;
    let ph3 = eta * (eta - beta + xi * (beta + gamma - 1.0) / gamma) / bec;
    let ph4 = xi * (1.0 - xi - eta) / alalc;
    let ph5 = xi * eta / gagac;
    let ph6 = eta * (1.0 - xi - eta) / bebec;
    let ph1 = 1.0 - ph2 - ph3 - ph4 - ph5 - ph6;

    let dph2 = (2.0 * xi - alpha + eta * (alpha - gamma) / gac) / alc;
    let dph3 = eta * (beta + gamma - 1.0) / (gamma * bec);
    let dph4 = (1.0 - 2.0 * xi - eta) / alalc;
    let dph5 = eta / gagac;
    let dph6 = -eta / bebec;
    let dph1 = -dph2 - dph3 - dph4 - dph5 - dph6;

    let pph2 = xi * (alpha - gamma) / (alc * gac);
    let pph3 = (2.0 * eta - beta + xi * (beta + gamma - 1.0) / gamma) / bec;
    let pph4 = -xi / alalc;
    let pph5 = xi / gagac;
    let pph6 = (1.0 - xi - 2.0 * eta) / bebec;
    let pph1 = -pph2 - pph3 - pph4 - pph5 - pph6;

    (
        [ph1, ph2, ph3, ph4, ph5, ph6],
        [dph1, dph2, dph3, dph4, dph5, dph6],
        [pph1, pph2, pph3, pph4, pph5, pph6],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bem::geom::{abc, ellip_gridder, ellip_gridder_no_rotation};

    fn unit_element(ndiv: u32, elem: usize) -> ([Vector3<f64>; 6], (f64, f64, f64)) {
        let (_nelm, _npts, p, n) = ellip_gridder_no_rotation(ndiv);
        let nodes = std::array::from_fn(|i| {
            let idx = n[(elem, i)];
            Vector3::new(p[(idx, 0)], p[(idx, 1)], p[(idx, 2)])
        });
        let abc = abc(nodes[0], nodes[1], nodes[2], nodes[3], nodes[4], nodes[5]);
        (nodes, abc)
    }

    #[test]
    fn solver_shape_axes_match_grid_nodes() {
        let shape = Vector3::<f64>::new(1.0, 0.8, 0.6);
        let req = (shape[0] * shape[1] * shape[2]).powf(1.0_f64 / 3.0);
        let axes = semi_axes_from_solver_shape(req, &shape);
        let centre = Vector3::new(0.0, 0.0, 0.0);
        let orient = UnitQuaternion::identity();
        let (_nelm, npts, p, _n) = ellip_gridder(2, req, &shape, &centre, &orient);

        for i in 0..npts {
            let point = Vector3::new(p[(i, 0)], p[(i, 1)], p[(i, 2)]);
            assert!(ellipsoid_level_set_body(&point, &axes).abs() < 1.0e-12);
        }
    }

    #[test]
    fn exact_patch_points_and_normals_are_on_ellipsoid() {
        let shape = Vector3::<f64>::new(1.0, 0.8, 0.6);
        let req = (shape[0] * shape[1] * shape[2]).powf(1.0_f64 / 3.0);
        let axes = semi_axes_from_solver_shape(req, &shape);
        let centre = Vector3::new(0.2, -0.1, 0.4);
        let orient = UnitQuaternion::from_euler_angles(0.3, -0.2, 0.5);
        let (nodes, (alpha, beta, gamma)) = unit_element(1, 0);
        let patch = ExactEllipsoidPatch::new(nodes, axes, centre, orient, alpha, beta, gamma);

        for (xi, eta) in [(0.2, 0.2), (0.6, 0.1), (0.1, 0.7), (alpha, 0.0)] {
            let point = patch.evaluate(xi, eta);
            let body_point = patch
                .orientation()
                .inverse_transform_vector(&(point.position - patch.centre()));
            assert!(ellipsoid_level_set_body(&body_point, &axes).abs() < 1.0e-12);
            assert!(point.jacobian > 0.0);

            let normal_body = patch.orientation().inverse_transform_vector(&point.normal);
            let exact_body = ellipsoid_normal_body(&body_point, &axes);
            assert!(normal_body.dot(&exact_body) > 1.0 - 1.0e-12);
        }
    }

    #[test]
    fn unit_sphere_patch_normal_is_radial() {
        let axes = Vector3::<f64>::new(1.0, 1.0, 1.0);
        let centre = Vector3::zeros();
        let orient = UnitQuaternion::identity();
        let (nodes, (alpha, beta, gamma)) = unit_element(2, 3);
        let patch = ExactEllipsoidPatch::new(nodes, axes, centre, orient, alpha, beta, gamma);

        for (xi, eta) in [(0.25, 0.25), (0.5, 0.2), (0.1, 0.5)] {
            let point = patch.evaluate(xi, eta);
            assert!((point.position.norm() - 1.0).abs() < 1.0e-12);
            assert!(point.normal.dot(&point.position.normalize()) > 1.0 - 1.0e-12);
            assert!(point.jacobian > 0.0);
        }
    }
}
