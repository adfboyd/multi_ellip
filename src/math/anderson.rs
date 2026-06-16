//! Anderson acceleration of a fixed-point iteration.

use nalgebra::{DMatrix, DVector};

/// One Anderson-accelerated update of a fixed-point iterate.
///
/// `w_k` is the current iterate and `f_k` the base-map correction, so the
/// un-accelerated next iterate would be `w_k + f_k`. Mix up to `m` previous
/// (iterate, correction) pairs via a small regularised least-squares to
/// extrapolate toward the fixed point, accelerating a slowly-contracting
/// coupling. Falls back to the plain update on the first call or a degenerate
/// system. This only changes the path to the fixed point, not the fixed point
/// itself (the caller's convergence test is unchanged).
pub fn anderson_next(
    w_hist: &mut Vec<DVector<f64>>,
    f_hist: &mut Vec<DVector<f64>>,
    w_k: &DVector<f64>,
    f_k: &DVector<f64>,
    m: usize,
) -> DVector<f64> {
    w_hist.push(w_k.clone());
    f_hist.push(f_k.clone());
    let k = f_hist.len();
    let plain = w_k + f_k;
    let next = if k < 2 {
        plain
    } else {
        let mk = m.min(k - 1);
        let d = f_k.len();
        let mut df = DMatrix::<f64>::zeros(d, mk);
        let mut dw = DMatrix::<f64>::zeros(d, mk);
        for j in 0..mk {
            let hi = k - 1 - j;
            let lo = k - 2 - j;
            df.set_column(j, &(&f_hist[hi] - &f_hist[lo]));
            dw.set_column(j, &(&w_hist[hi] - &w_hist[lo]));
        }
        // argmin_gamma || f_k - df gamma ||  via regularised normal equations.
        let ata = df.transpose() * &df;
        let scale = ata.diagonal().max().max(1e-300);
        let reg = &ata + DMatrix::<f64>::identity(mk, mk) * (1e-12 * scale);
        match reg.lu().solve(&(df.transpose() * f_k)) {
            Some(gamma) => &plain - (dw + df) * gamma,
            None => plain,
        }
    };
    while f_hist.len() > m + 1 {
        w_hist.remove(0);
        f_hist.remove(0);
    }
    next
}
