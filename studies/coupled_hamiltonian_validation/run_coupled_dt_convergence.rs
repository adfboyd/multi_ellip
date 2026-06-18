use std::collections::HashMap;
use std::env;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;
use std::process::{Command, Stdio};

#[derive(Clone)]
struct Case {
    name: String,
    dt: f64,
    tend: f64,
    rho: f64,
    ndiv: usize,
    sep: f64,
}

#[derive(Clone)]
struct Row {
    time: f64,
    ke_total: f64,
    pos: Vec<[f64; 3]>,
    marker: Vec<[f64; 3]>,
    pcon: [f64; 3],
    hcon: [f64; 3],
}

struct Summary {
    status: String,
    rows: usize,
    max_ke_drift_pct: Option<f64>,
    final_ke_drift_pct: Option<f64>,
    max_pos_err: Option<f64>,
    final_pos_err: Option<f64>,
    max_sep_err: Option<f64>,
    final_sep_err: Option<f64>,
    max_marker_err: Option<f64>,
    final_marker_err: Option<f64>,
    rel_dp: Option<f64>,
    rel_dh: Option<f64>,
    mean_step_s: Option<f64>,
    coupled_residual: Option<f64>,
    coupled_energy_error_rel: Option<f64>,
    jacobian_builds: Option<f64>,
    error: String,
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    let rerun = args.iter().any(|a| a == "--rerun");
    let summary_only = args.iter().any(|a| a == "--summary-only");
    let build = args.iter().any(|a| a == "--build");

    let root = env::current_dir()?;
    let study = root.join("coupled_dt_convergence_runs");
    let runs = study.join("runs");
    let summary_path = study.join("coupled_dt_convergence_summary.csv");
    let pairwise_path = study.join("coupled_dt_pairwise_summary.csv");
    let study_log = study.join("study.log");
    let exe = root.join("target").join("release").join("multi_ellip.exe");
    fs::create_dir_all(&runs)?;

    if build && !summary_only {
        log_line(&study_log, "cargo build --release --bin multi_ellip")?;
        let status = Command::new("cargo")
            .args(["build", "--release", "--bin", "multi_ellip"])
            .current_dir(&root)
            .status()?;
        if !status.success() {
            return Err(io::Error::new(io::ErrorKind::Other, "cargo build failed"));
        }
    }

    if !summary_only && !exe.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("missing executable {}", exe.display()),
        ));
    }

    let cases = vec![
        case(0.2),
        case(0.1),
        case(0.05),
        case(0.025),
        case(0.0125),
    ];
    log_line(
        &study_log,
        &format!(
            "dt convergence start cases={} rerun={} summary_only={}",
            cases.len(),
            rerun,
            summary_only
        ),
    )?;

    for case in &cases {
        run_case(&root, &exe, &runs, &study_log, case, rerun, summary_only)?;
    }

    let reference_case = cases
        .last()
        .map(|case| case.name.as_str())
        .unwrap_or("dt0p025");
    let reference = load_rows(
        &runs
            .join(reference_case)
            .join("out")
            .join("multiple_body_complete.dat"),
    )
    .unwrap_or_default();
    let mut rows = Vec::new();
    for case in &cases {
        let run_dir = runs.join(&case.name);
        rows.push((case.clone(), summarize_case(&run_dir, &reference)));
    }
    write_summary(&summary_path, &rows)?;
    write_pairwise_summary(&pairwise_path, &cases, &runs)?;
    log_line(
        &study_log,
        &format!("summary written {}", summary_path.display()),
    )?;
    log_line(
        &study_log,
        &format!("pairwise summary written {}", pairwise_path.display()),
    )?;

    println!(
        "{:<10} {:>8} {:>11} {:>11} {:>11} {:>11} {:>11} {:>10} {:>9}",
        "dt", "rows", "maxKE%", "maxPos", "maxSep", "maxMark", "rel_dH", "step_s", "Jbuilds"
    );
    for (case, s) in &rows {
        println!(
            "{:<10.4} {:>8} {:>11} {:>11} {:>11} {:>11} {:>11} {:>10} {:>9}",
            case.dt,
            s.rows,
            fmt_opt(s.max_ke_drift_pct),
            fmt_opt(s.max_pos_err),
            fmt_opt(s.max_sep_err),
            fmt_opt(s.max_marker_err),
            fmt_opt(s.rel_dh),
            fmt_opt(s.mean_step_s),
            fmt_opt(s.jacobian_builds)
        );
    }
    println!("pairwise summary: {}", pairwise_path.display());

    Ok(())
}

fn case(dt: f64) -> Case {
    Case {
        name: format!("dt{}", name_float(dt)),
        dt,
        tend: 2.0,
        rho: 0.1,
        ndiv: 2,
        sep: 3.0,
    }
}

fn run_case(
    root: &Path,
    exe: &Path,
    runs: &Path,
    study_log: &Path,
    case: &Case,
    rerun: bool,
    summary_only: bool,
) -> io::Result<()> {
    let run_dir = runs.join(&case.name);
    let input_path = run_dir.join("input.txt");
    let out_dir = run_dir.join("out");
    let data_path = out_dir.join("multiple_body_complete.dat");
    let log_path = run_dir.join("run.log");
    fs::create_dir_all(&run_dir)?;
    fs::write(&input_path, input_text(case))?;

    if summary_only {
        return Ok(());
    }
    if data_path.exists() && !rerun {
        log_line(study_log, &format!("skip existing {}", case.name))?;
        return Ok(());
    }
    if out_dir.exists() {
        fs::remove_dir_all(&out_dir)?;
    }
    fs::create_dir_all(&out_dir)?;

    log_line(study_log, &format!("run {}", case.name))?;
    let log_file = File::create(&log_path)?;
    let log_file_err = log_file.try_clone()?;
    let status = Command::new(exe)
        .arg(&input_path)
        .arg(&out_dir)
        .current_dir(root)
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(log_file_err))
        .status()?;
    if status.success() {
        log_line(study_log, &format!("done {}", case.name))?;
    } else {
        log_line(
            study_log,
            &format!("FAILED {} exit={:?}", case.name, status.code()),
        )?;
    }
    Ok(())
}

fn input_text(case: &Case) -> String {
    let x1 = -0.5 * case.sep;
    let x2 = 0.5 * case.sep;
    let lines = vec![
        format!("cex1={x1}"),
        "cey1=0.0".to_string(),
        "cez1=0.0".to_string(),
        "oriw1=1.0".to_string(),
        "orii1=2.0".to_string(),
        "orij1=0.0".to_string(),
        "orik1=0.0".to_string(),
        "lvx1=3.40056370237515".to_string(),
        "lvy1=-2.3933837676760783".to_string(),
        "lvz1=6.54486317259435".to_string(),
        "avx1=-4.8726808101148835".to_string(),
        "avy1=-3.94750142245162".to_string(),
        "avz1=0.3906812599012585".to_string(),
        "shx1=1.0".to_string(),
        "shy1=0.7".to_string(),
        "shz1=0.7".to_string(),
        "req1=1.0".to_string(),
        format!("rhos1={}", case.rho),
        format!("cex2={x2}"),
        "cey2=0.0".to_string(),
        "cez2=0.0".to_string(),
        "oriw2=1.0".to_string(),
        "orii2=0.0".to_string(),
        "orij2=1.0".to_string(),
        "orik2=0.0".to_string(),
        "lvx2=3.7391939347605".to_string(),
        "lvy2=-2.6317184005104504".to_string(),
        "lvz2=7.1966046869550295".to_string(),
        "avx2=-2.142982944119855".to_string(),
        "avy2=4.896730077049289".to_string(),
        "avz2=3.302737691384179".to_string(),
        "shx2=1.0".to_string(),
        "shy2=0.7".to_string(),
        "shz2=0.7".to_string(),
        "req2=1.0".to_string(),
        format!("rhos2={}", case.rho),
        "rhof=1.0".to_string(),
        format!("ndiv={}", case.ndiv),
        format!("tend={}", case.tend),
        format!("dt={}", case.dt),
        "tprint=1".to_string(),
        "logevery=10".to_string(),
        "nbody=2".to_string(),
        "impulse_scheme=1".to_string(),
        "energy_projection=0".to_string(),
        "fluid_energy_gradient=0".to_string(),
        "hamiltonian_midpoint_scheme=1".to_string(),
        "hamiltonian_coupled_solve=1".to_string(),
        "hamiltonian_coupled_iters=6".to_string(),
        "hamiltonian_coupled_eps=0.001".to_string(),
        "hamiltonian_coupled_max_shift=0.2".to_string(),
        "hamiltonian_coupled_jacobian_interval=6".to_string(),
        "hamiltonian_coupled_broyden_update=1".to_string(),
        "hamiltonian_coupled_endpoint_velocity=1".to_string(),
        "hamiltonian_adaptive_substeps=1".to_string(),
        "hamiltonian_max_substeps=8".to_string(),
        "hamiltonian_floor_tol=0.0001".to_string(),
    ];
    lines.join("\n") + "\n"
}

fn summarize_case(run_dir: &Path, reference: &[Row]) -> Summary {
    let data_path = run_dir.join("out").join("multiple_body_complete.dat");
    let log_path = run_dir.join("run.log");
    let rows = match load_rows(&data_path) {
        Ok(rows) if rows.len() >= 2 => rows,
        Ok(rows) => return Summary::missing("too_few_rows", rows.len(), ""),
        Err(e) => return Summary::missing("missing_output", 0, &e.to_string()),
    };
    let log_text = fs::read_to_string(&log_path).unwrap_or_default();
    let solver_error = log_text
        .lines()
        .find(|line| line.contains("Solver stopped with error:"))
        .map(|line| line.trim().to_string());

    let ke0 = rows[0].ke_total;
    let mut max_ke = 0.0_f64;
    let p0 = rows[0].pcon;
    let h0 = rows[0].hcon;
    let mut max_dp = 0.0_f64;
    let mut max_dh = 0.0_f64;
    for row in &rows {
        max_ke = max_ke.max((100.0 * (row.ke_total - ke0) / ke0).abs());
        max_dp = max_dp.max(norm(diff(row.pcon, p0)));
        max_dh = max_dh.max(norm(diff(row.hcon, h0)));
    }

    let mut max_pos_err = 0.0_f64;
    let mut final_pos_err = 0.0_f64;
    let mut max_sep_err = 0.0_f64;
    let mut final_sep_err = 0.0_f64;
    let mut max_marker_err = 0.0_f64;
    let mut final_marker_err = 0.0_f64;
    if !reference.is_empty() {
        for row in &rows {
            if let Some(r) = reference_at(reference, row.time) {
                let pos_err = stacked_error(&row.pos, &r.pos);
                let sep_err = (separation(row) - separation(r)).abs();
                let marker_err = stacked_error(&row.marker, &r.marker);
                max_pos_err = max_pos_err.max(pos_err);
                max_sep_err = max_sep_err.max(sep_err);
                max_marker_err = max_marker_err.max(marker_err);
                final_pos_err = pos_err;
                final_sep_err = sep_err;
                final_marker_err = marker_err;
            }
        }
    }

    let last = rows.last().unwrap();
    Summary {
        status: solver_error
            .as_ref()
            .map(|_| "solver_error".to_string())
            .unwrap_or_else(|| "ok".to_string()),
        rows: rows.len(),
        max_ke_drift_pct: Some(max_ke),
        final_ke_drift_pct: Some(100.0 * (last.ke_total - ke0) / ke0),
        max_pos_err: Some(max_pos_err),
        final_pos_err: Some(final_pos_err),
        max_sep_err: Some(max_sep_err),
        final_sep_err: Some(final_sep_err),
        max_marker_err: Some(max_marker_err),
        final_marker_err: Some(final_marker_err),
        rel_dp: (norm(p0) > 0.0).then_some(max_dp / norm(p0)),
        rel_dh: (norm(h0) > 0.0).then_some(max_dh / norm(h0)),
        mean_step_s: log_value(&log_text, "Mean time/step:"),
        coupled_residual: log_value(&log_text, "Coupled max residual norm:"),
        coupled_energy_error_rel: log_value(&log_text, "Coupled max energy error rel:"),
        jacobian_builds: log_value(&log_text, "Coupled Jacobian builds:"),
        error: solver_error.unwrap_or_default(),
    }
}

impl Summary {
    fn missing(status: &str, rows: usize, error: &str) -> Self {
        Self {
            status: status.to_string(),
            rows,
            max_ke_drift_pct: None,
            final_ke_drift_pct: None,
            max_pos_err: None,
            final_pos_err: None,
            max_sep_err: None,
            final_sep_err: None,
            max_marker_err: None,
            final_marker_err: None,
            rel_dp: None,
            rel_dh: None,
            mean_step_s: None,
            coupled_residual: None,
            coupled_energy_error_rel: None,
            jacobian_builds: None,
            error: error.to_string(),
        }
    }
}

fn load_rows(path: &Path) -> io::Result<Vec<Row>> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines();
    let header_line = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "empty csv"))?;
    let headers: Vec<String> = header_line
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    let index: HashMap<String, usize> = headers
        .iter()
        .enumerate()
        .map(|(i, h)| (h.clone(), i))
        .collect();
    let mut rows = Vec::new();
    for line in lines.filter(|l| !l.trim().is_empty()) {
        let fields: Vec<String> = line.split(',').map(|s| s.trim().to_string()).collect();
        let value = |name: &str| -> f64 {
            index
                .get(name)
                .and_then(|i| fields.get(*i))
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(0.0)
        };
        let mut pos = Vec::new();
        let mut marker = Vec::new();
        for b in 1..=2 {
            pos.push([
                value(&format!("px_{b}")),
                value(&format!("py_{b}")),
                value(&format!("pz_{b}")),
            ]);
            marker.push([
                value(&format!("ofix1_{b}")),
                value(&format!("ofix2_{b}")),
                value(&format!("ofix3_{b}")),
            ]);
        }
        rows.push(Row {
            time: value("time"),
            ke_total: value("ke_total"),
            pos,
            marker,
            pcon: [
                value("pcon_x_1") + value("pcon_x_2"),
                value("pcon_y_1") + value("pcon_y_2"),
                value("pcon_z_1") + value("pcon_z_2"),
            ],
            hcon: [
                value("hcon_x_1") + value("hcon_x_2"),
                value("hcon_y_1") + value("hcon_y_2"),
                value("hcon_z_1") + value("hcon_z_2"),
            ],
        });
    }
    Ok(rows)
}

fn write_summary(path: &Path, rows: &[(Case, Summary)]) -> io::Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "dt,status,rows,max_ke_drift_pct,final_ke_drift_pct,max_pos_err,final_pos_err,max_sep_err,final_sep_err,max_marker_err,final_marker_err,rel_dP,rel_dH,mean_step_s,coupled_residual,coupled_energy_error_rel,jacobian_builds,error")?;
    for (case, s) in rows {
        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            case.dt,
            s.status,
            s.rows,
            csv_opt(s.max_ke_drift_pct),
            csv_opt(s.final_ke_drift_pct),
            csv_opt(s.max_pos_err),
            csv_opt(s.final_pos_err),
            csv_opt(s.max_sep_err),
            csv_opt(s.final_sep_err),
            csv_opt(s.max_marker_err),
            csv_opt(s.final_marker_err),
            csv_opt(s.rel_dp),
            csv_opt(s.rel_dh),
            csv_opt(s.mean_step_s),
            csv_opt(s.coupled_residual),
            csv_opt(s.coupled_energy_error_rel),
            csv_opt(s.jacobian_builds),
            s.error.replace(',', ";")
        )?;
    }
    Ok(())
}

fn write_pairwise_summary(path: &Path, cases: &[Case], runs: &Path) -> io::Result<()> {
    let horizons = [0.25, 0.5, 1.0, 2.0];
    let mut loaded = Vec::new();
    for case in cases {
        let rows = load_rows(
            &runs
                .join(&case.name)
                .join("out")
                .join("multiple_body_complete.dat"),
        )
        .unwrap_or_default();
        loaded.push((case, rows));
    }

    let mut file = File::create(path)?;
    writeln!(
        file,
        "dt_coarse,dt_fine,horizon,ncompare,max_pos_err,final_pos_err,max_sep_err,final_sep_err,max_marker_err,final_marker_err"
    )?;
    for pair in loaded.windows(2) {
        let (coarse_case, coarse_rows) = &pair[0];
        let (fine_case, fine_rows) = &pair[1];
        if coarse_rows.is_empty() || fine_rows.is_empty() {
            continue;
        }
        for horizon in horizons {
            let mut ncompare = 0_usize;
            let mut max_pos_err = 0.0_f64;
            let mut final_pos_err = 0.0_f64;
            let mut max_sep_err = 0.0_f64;
            let mut final_sep_err = 0.0_f64;
            let mut max_marker_err = 0.0_f64;
            let mut final_marker_err = 0.0_f64;
            for row in coarse_rows.iter().filter(|row| row.time <= horizon + 1.0e-8) {
                if let Some(fine) = reference_at(fine_rows, row.time) {
                    ncompare += 1;
                    let pos_err = stacked_error(&row.pos, &fine.pos);
                    let sep_err = (separation(row) - separation(fine)).abs();
                    let marker_err = stacked_error(&row.marker, &fine.marker);
                    max_pos_err = max_pos_err.max(pos_err);
                    max_sep_err = max_sep_err.max(sep_err);
                    max_marker_err = max_marker_err.max(marker_err);
                    final_pos_err = pos_err;
                    final_sep_err = sep_err;
                    final_marker_err = marker_err;
                }
            }
            writeln!(
                file,
                "{},{},{},{},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}",
                coarse_case.dt,
                fine_case.dt,
                horizon,
                ncompare,
                max_pos_err,
                final_pos_err,
                max_sep_err,
                final_sep_err,
                max_marker_err,
                final_marker_err
            )?;
        }
    }
    Ok(())
}

fn reference_at(reference: &[Row], time: f64) -> Option<&Row> {
    reference
        .iter()
        .find(|row| (row.time - time).abs() < 1.0e-8)
}

fn separation(row: &Row) -> f64 {
    norm(diff(row.pos[1], row.pos[0]))
}

fn stacked_error(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    let sum = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = diff(*x, *y);
            d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
        })
        .sum::<f64>();
    sum.sqrt()
}

fn log_value(text: &str, label: &str) -> Option<f64> {
    text.lines().rev().find_map(|line| {
        let pos = line.find(label)?;
        line[pos + label.len()..]
            .split_whitespace()
            .next()?
            .parse::<f64>()
            .ok()
    })
}

fn log_line(path: &Path, message: &str) -> io::Result<()> {
    let mut file = fs::OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{}", message)
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn diff(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn name_float(v: f64) -> String {
    format!("{v:.4}")
        .trim_end_matches('0')
        .trim_end_matches('.')
        .replace('.', "p")
}

fn fmt_opt(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.4e}")).unwrap_or_default()
}

fn csv_opt(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.12e}")).unwrap_or_default()
}
