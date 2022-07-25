use std::ops::{Add, Sub};

use nalgebra as na;
use nalgebra::OMatrix;

pub type Vector7<T> = na::Matrix<T, na::U7, na::U1, na::ArrayStorage<T, 7, 1>>;

#[derive(Debug, Clone, Copy)]
pub struct State {
    pub(crate) v: na::Vector3<f64>,
    pub(crate) q: na::Quaternion<f64>,
}

impl State {
    pub fn new(v_new: na::Vector3<f64>, q_new: na::Quaternion<f64>) -> State {
        State { v: v_new, q: q_new }
    }

    pub fn from_vec(q_vec: OMatrix<f64, na::U7, na::U1>) -> State {
        State {
            v: na::Vector3::new(q_vec[0], q_vec[1], q_vec[2]),
            q: na::Quaternion::new(q_vec[3], q_vec[4], q_vec[5], q_vec[6]),
        }
    }

    pub fn to_vec(&self) -> Vector7<f64> {
        let v = na::vector![
            self.v[0], self.v[1], self.v[2], self.q[0], self.q[1], self.q[2], self.q[3]
        ];
        v
    }
}

impl Add for State {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            v: self.v + other.v,
            q: self.q + other.q,
        }
    }
}

impl Sub for State {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self {
            v: self.v - other.v,
            q: self.q - other.q,
        }
    }
}
