use nalgebra::{DMatrix, DVector};

/// Left-preconditioned GMRES (no restart) for `A x = b`.
///
/// `matvec(v)` returns `A v`; `precond(r)` returns `M^{-1} r` for the chosen
/// preconditioner `M`. Iterates on the preconditioned system `M^{-1} A x =
/// M^{-1} b`. Returns `(x, iterations, relative_residual)`, where the residual
/// is `||M^{-1}(b - A x)|| / ||M^{-1} b||`.
///
/// No restart: for the second-kind influence-matrix systems here (A = ½I + D)
/// with a block-diagonal self-block preconditioner, convergence takes only a
/// handful of iterations, so the Krylov basis stays small.
pub fn gmres<MV, PC>(
    matvec: MV,
    precond: PC,
    b: &DVector<f64>,
    x0: &DVector<f64>,
    tol: f64,
    max_iter: usize,
) -> (DVector<f64>, usize, f64)
where
    MV: Fn(&DVector<f64>) -> DVector<f64>,
    PC: Fn(&DVector<f64>) -> DVector<f64>,
{
    let n = b.len();
    let m = max_iter.min(n).max(1);

    // Relative-residual reference: ||M^{-1} b||.
    let bref = precond(b).norm().max(1e-300);

    // Preconditioned initial residual.
    let r0 = precond(&(b - matvec(x0)));
    let beta0 = r0.norm();
    if beta0 <= tol * bref {
        return (x0.clone(), 0, beta0 / bref);
    }

    let mut v: Vec<DVector<f64>> = Vec::with_capacity(m + 1);
    v.push(&r0 / beta0);

    let mut h = DMatrix::<f64>::zeros(m + 1, m);
    let mut cs = vec![0.0_f64; m];
    let mut sn = vec![0.0_f64; m];
    let mut g = DVector::<f64>::zeros(m + 1);
    g[0] = beta0;

    let mut used = 0;
    let mut resid = beta0 / bref;

    for j in 0..m {
        // Arnoldi step on the preconditioned operator: w = M^{-1} A v_j.
        let mut w = precond(&matvec(&v[j]));

        // Modified Gram-Schmidt against the existing basis.
        for i in 0..=j {
            let hij = w.dot(&v[i]);
            h[(i, j)] = hij;
            w -= hij * &v[i];
        }
        let hnext = w.norm();
        h[(j + 1, j)] = hnext;
        let breakdown = hnext <= 1e-14;
        if !breakdown {
            v.push(&w / hnext);
        }

        // Apply previous Givens rotations to the new Hessenberg column.
        for i in 0..j {
            let temp = cs[i] * h[(i, j)] + sn[i] * h[(i + 1, j)];
            h[(i + 1, j)] = -sn[i] * h[(i, j)] + cs[i] * h[(i + 1, j)];
            h[(i, j)] = temp;
        }

        // New Givens rotation to zero the sub-diagonal entry.
        let denom = (h[(j, j)].powi(2) + h[(j + 1, j)].powi(2)).sqrt();
        let (c, s) = if denom < 1e-300 {
            (1.0, 0.0)
        } else {
            (h[(j, j)] / denom, h[(j + 1, j)] / denom)
        };
        cs[j] = c;
        sn[j] = s;
        h[(j, j)] = c * h[(j, j)] + s * h[(j + 1, j)];
        h[(j + 1, j)] = 0.0;

        // Rotate the residual vector.
        let temp = c * g[j] + s * g[j + 1];
        g[j + 1] = -s * g[j] + c * g[j + 1];
        g[j] = temp;

        used = j + 1;
        resid = g[j + 1].abs() / bref;
        if resid <= tol || breakdown {
            break;
        }
    }

    // Back-substitute the upper-triangular least-squares system H y = g.
    let mut y = DVector::<f64>::zeros(used);
    for i in (0..used).rev() {
        let mut sum = g[i];
        for k in (i + 1)..used {
            sum -= h[(i, k)] * y[k];
        }
        y[i] = sum / h[(i, i)];
    }

    // x = x0 + V y.
    let mut x = x0.clone();
    for i in 0..used {
        x += y[i] * &v[i];
    }

    (x, used, resid)
}
