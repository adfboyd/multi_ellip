use csv::*;
use nalgebra as na;
use serde::Serialize;

use crate::ellipsoids::body::Body;

type State = na::Vector3<f64>;
type Time = f64;

struct Ellipsoid {
    shape: na::Vector3<f64>,
}

impl Body {
    pub fn is_inside(&self, x :na::Vector3<f64>) -> bool {
        // Returns if a given point is inside the body
        let s = self.shape;
        let (a, b, c) = (s[0], s[1], s[2]);
        let (x1, x2, x3) = (x[0], x[1], x[2]);

        let res = (x1/a).powi(2) + (x2/b).powi(2) + (x3/c).powi(2);

        res < 1.0
    }

    pub fn norm_vec(&self, x :na::Vector3<f64>) -> na::Vector3<f64> {

        let s = self.shape;
        let (a, b, c) = (s[0], s[1], s[2]);
        let (x1, x2, x3) = (x[0], x[1], x[2]);

        let prefac = x1/(a*a).powi(2) + x2/(b*b).powi(2) + x3/(c*c).powi(2);
        let fac = 1.0 / prefac.sqrt();

        let n1 = fac * x1 / (a*a);
        let n2 = fac * x2 / (b*b);
        let n3 = fac * x3 / (c*c);

        na::Vector3::new(n1, n2, n3)

    }

    pub fn z_on_surface(&self, x :na::Vector2<f64>) -> f64 {

        let s = self.shape;
        let (a, b, c) = (s[0], s[1], s[2]);
        let (x1, x2) = (x[0], x[1]);

        let d1 = (x1/a).powi(2);
        let d2 =  (x2/b).powi(2);
        let u = c * (1.0 - d1 - d2).sqrt();
        u
    }

    pub fn sa_est(&self) -> f64 {
        let s = self.shape;
        let (a, b, c) = (s[0], s[1], s[2]);
        let p = 1.6;
        let num = (a*b).powf(p) + (b*c).powf(p) + (a*c).powf(p);
        let pi4 = std::f64::consts::PI * 4.0;
        let sa = pi4 * (num / 3.0).powf(1.0/p);
        sa
    }


    pub fn integ(&self, f: &dyn Fn(na::Vector3<f64>) -> f64) -> f64 {

        let s = self.shape;

        lambda = 1e-8;

        let outer = | x :f64 | -> f64 {

            let (a, b, c) = (s[0], s[1], s[2]);
            let delta = 1.0 - (c/a).powi(2);
            let eps = 1.0 - (c/b).powi(2);
            let s1 = x / a;
            let m  = eps * (1.0 - s1*s1) / (1.0 - delta * s1*s1);

            let inner = |q :f64| -> f64 {

                let y = b * q.sin() * (1.0 - s1*s1).sqrt();

                let alpha = 1.0 - (x/a).powi(2) - (y/b).powi(2);

                let u =
                    if alpha >= 0.0 {
                        c * alpha.sqrt()
                    }
                    else {
                        0.0
                    };

                let p1 = na::Vector3::new(x, y, u);
                let p2 = na::Vector3::new(x, y, -u);

                let sq_br = f(p1) + f(p2);
                let w = (1.0 - m * q.sin().powi(2));

                sq_br * w
            };

            let (status, result, abs_err, resabs) = rgsl::integration::qng(inner, 0.0, lambda, 1e-8, 1e-8);
            let fac = b * (1.0 - delta * s1 * s1).sqrt();
            result * fac
        };

        let (status, result, abs_err, resabs) = rgsl::integration::qng(outer, 0.0, lambda, 1e-8, 1e-8);

        result

    }

    pub fn phi(&self, x :na::Vector3<f64>) -> f64 {
        let s = self.shape;
        let v = self.linear_velocity();
        let omega = self.angular_velocity();

        let (x1, x2, x3) = (x[0], x[1], x[2]);
        let (a, b, c) = (s[0], s[1], s[2]);
        let (v1, v2, v3) = (v[0], v[1], v[2]);
        let (o1, o2, o3) = (omega[0], omega[1], omega[2]);
        let (aa, bb, cc) = (a*a, b*b, c*c);

        let lin_phi = -v1*x1 - v2*x2 - v3*x3;
        let lin_phi = -v.dot(&x);
        let rot_phi1 = - (bb - cc)/(bb + cc) * o1 * x2 * x3;
        let rot_phi2 = - (cc - aa)/(cc + aa) * o2 * x1 * x3;
        let rot_phi3 = - (aa - bb)/(aa + bb) * o3 * x1 * x2;

        let phi_val = lin_phi + rot_phi1 + rot_phi2 + rot_phi3;

        phi_val
    }

    pub fn gradphi(&self, x :na::Vector3<f64>) -> na::Vector3<f64> {
        let s = self.shape;
        let v = self.linear_velocity();
        let omega = self.angular_velocity();

        let (x1, x2, x3) = (x[0], x[1], x[2]);
        let (a, b, c) = (s[0], s[1], s[2]);
        let (v1, v2, v3) = (v[0], v[1], v[2]);
        let (o1, o2, o3) = (omega[0], omega[1], omega[2]);
        let (aa, bb, cc) = (a*a, b*b, c*c);

        let fac1 = (bb - cc)/(bb + cc) * o1;
        let fac2 = (cc - aa)/(cc + aa) * o2;
        let fac3 = (aa - bb)/(aa + bb) * o3;

        let u1 = -v1 - fac2 * x3 - fac3 * x2;
        let u2 = -v2 - fac1 * x3 - fac3 * x1;
        let u3 = -v3 - fac1 * x2 - fac2 * x1;

        let u = na::Vector3::new(u1, u2, u3);
        u
    }

}
//
// impl crate::ode::System<State> for Ellipsoid {
//     fn system(&self, x: Time, _y: &State, dy : &mut State) {
//
//         let (a, b, c) = (self.shape[0], self.shape[1], self.shape[2]);
//         let delta = 1.0 - (c/a).powi(2);
//         let eps = 1.0 - (c/b).powi(2);
//         let s1 = x / a;
//         let m  = eps * (1.0 - s1*s1) / (1.0 - delta * s1*s1);
//
//     }
// }