use std::collections::HashMap;
use std::env;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;
use std::process::{Command, Stdio};

#[derive(Clone)]
struct Case {
    name: String,
    scheme: &'static str,
    rho: f64,
    ndiv: usize,
    dt: f64,
    tend: f64,
    sep: f64,
}

struct Summary {
    status: String,
    rows: usize,
    max_ke_drift_pct: Option<f64>,
    final_ke_drift_pct: Option<f64>,
    max_dp: Option<f64>,
    rel_dp: Option<f64>,
    max_dh: Option<f64>,
    rel_dh: Option<f64>,
    mean_step_s: Option<f64>,
    coupled_residual: Option<f64>,
    coupled_impulse_residual: Option<f64>,
    coupled_energy_error_rel: Option<f64>,
    floor_excess: Option<f64>,
    floor_hits_fallbacks: Option<String>,
    error: String,
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    let rerun = args.iter().any(|a| a == "--rerun");
    let summary_only = args.iter().any(|a| a == "--summary-only");
    let build = args.iter().any(|a| a == "--build");

    let root = env::current_dir()?;
    let study = root.join("coupled_validation_runs");
    let runs = study.join("runs");
    let summary_path = study.join("coupled_hamiltonian_validation_summary.csv");
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

    let cases = cases();
    log_line(
        &study_log,
        &format!(
            "study start cases={} rerun={} summary_only={}",
            cases.len(),
            rerun,
            summary_only
        ),
    )?;

    for case in &cases {
        run_case(&root, &exe, &runs, &study_log, case, rerun, summary_only)?;
    }

    let mut rows = Vec::new();
    for case in &cases {
        let run_dir = runs.join(&case.name);
        rows.push((case.clone(), summarize_run(&run_dir)));
    }
    write_summary(&summary_path, &rows)?;
    log_line(
        &study_log,
        &format!("summary written {}", summary_path.display()),
    )?;

    println!(
        "{:<42} {:<8} {:<12} {:>12} {:>12} {:>10} {:>10} {:>10}",
        "name", "scheme", "status", "max_ke_%", "final_ke_%", "rel_dP", "rel_dH", "step_s"
    );
    for (case, s) in &rows {
        println!(
            "{:<42} {:<8} {:<12} {:>12} {:>12} {:>10} {:>10} {:>10}",
            case.name,
            case.scheme,
            s.status,
            fmt_opt(s.max_ke_drift_pct),
            fmt_opt(s.final_ke_drift_pct),
            fmt_opt(s.rel_dp),
            fmt_opt(s.rel_dh),
            fmt_opt(s.mean_step_s)
        );
    }

    Ok(())
}

fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    for rho in [1.0, 0.1, 0.01] {
        for scheme in ["impulse", "coupled"] {
            out.push(Case {
                name: format!("close_rho{}_nd2_dt0p1_t2_{}", rho_name(rho), scheme),
                scheme,
                rho,
                ndiv: 2,
                dt: 0.1,
                tend: 2.0,
                sep: 3.0,
            });
        }
    }

    for scheme in ["impulse", "coupled"] {
        out.push(Case {
            name: format!("close_rho0p1_nd2_dt0p05_t1_{}", scheme),
            scheme,
            rho: 0.1,
            ndiv: 2,
            dt: 0.05,
            tend: 1.0,
            sep: 3.0,
        });
        out.push(Case {
            name: format!("far_rho0p1_nd2_dt0p1_t2_{}", scheme),
            scheme,
            rho: 0.1,
            ndiv: 2,
            dt: 0.1,
            tend: 2.0,
            sep: 8.0,
        });
        out.push(Case {
            name: format!("close_rho0p1_nd3_dt0p1_t0p5_{}", scheme),
            scheme,
            rho: 0.1,
            ndiv: 3,
            dt: 0.1,
            tend: 0.5,
            sep: 3.0,
        });
    }
    out
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
    let mut lines = vec![
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
        "logevery=5".to_string(),
        "nbody=2".to_string(),
        "impulse_scheme=1".to_string(),
        "energy_projection=0".to_string(),
        "fluid_energy_gradient=0".to_string(),
    ];
    if case.scheme == "coupled" {
        lines.extend([
            "hamiltonian_midpoint_scheme=1".to_string(),
            "hamiltonian_coupled_solve=1".to_string(),
            "hamiltonian_coupled_iters=4".to_string(),
            "hamiltonian_coupled_eps=0.001".to_string(),
            "hamiltonian_coupled_max_shift=0.2".to_string(),
            "hamiltonian_floor_tol=0.001".to_string(),
        ]);
    }
    lines.join("\n") + "\n"
}

fn summarize_run(run_dir: &Path) -> Summary {
    let data_path = run_dir.join("out").join("multiple_body_complete.dat");
    let log_path = run_dir.join("run.log");
    if !data_path.exists() {
        return Summary::missing("missing_output", "no multiple_body_complete.dat");
    }

    let text = match fs::read_to_string(&data_path) {
        Ok(s) => s,
        Err(e) => return Summary::missing("read_failed", &e.to_string()),
    };
    let mut lines = text.lines();
    let Some(header_line) = lines.next() else {
        return Summary::missing("empty_output", "");
    };
    let headers: Vec<String> = header_line
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    let index: HashMap<String, usize> = headers
        .iter()
        .enumerate()
        .map(|(i, h)| (h.clone(), i))
        .collect();
    let rows: Vec<Vec<String>> = lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split(',').map(|s| s.trim().to_string()).collect())
        .collect();
    let log_text = fs::read_to_string(&log_path).unwrap_or_default();
    let solver_error = log_text
        .lines()
        .find(|line| line.contains("Solver stopped with error:"))
        .map(|line| line.trim().to_string());

    if rows.len() < 2 {
        return Summary::missing("too_few_rows", "");
    }

    let value = |row: &Vec<String>, name: &str| -> Option<f64> {
        let i = *index.get(name)?;
        row.get(i)?.parse::<f64>().ok()
    };

    let ke0 = match value(&rows[0], "ke_total") {
        Some(v) => v,
        None => return Summary::missing("parse_failed", "missing ke_total"),
    };
    let first = &rows[0];
    let p0 = [
        value(first, "pcon_x_1").unwrap_or(0.0) + value(first, "pcon_x_2").unwrap_or(0.0),
        value(first, "pcon_y_1").unwrap_or(0.0) + value(first, "pcon_y_2").unwrap_or(0.0),
        value(first, "pcon_z_1").unwrap_or(0.0) + value(first, "pcon_z_2").unwrap_or(0.0),
    ];
    let h0 = [
        value(first, "hcon_x_1").unwrap_or(0.0) + value(first, "hcon_x_2").unwrap_or(0.0),
        value(first, "hcon_y_1").unwrap_or(0.0) + value(first, "hcon_y_2").unwrap_or(0.0),
        value(first, "hcon_z_1").unwrap_or(0.0) + value(first, "hcon_z_2").unwrap_or(0.0),
    ];

    let mut max_ke_drift_pct: f64 = 0.0;
    let mut max_dp: f64 = 0.0;
    let mut max_dh: f64 = 0.0;
    for row in &rows {
        let ke = value(row, "ke_total").unwrap_or(ke0);
        max_ke_drift_pct = max_ke_drift_pct.max((100.0 * (ke - ke0) / ke0).abs());
        let p = [
            value(row, "pcon_x_1").unwrap_or(0.0) + value(row, "pcon_x_2").unwrap_or(0.0),
            value(row, "pcon_y_1").unwrap_or(0.0) + value(row, "pcon_y_2").unwrap_or(0.0),
            value(row, "pcon_z_1").unwrap_or(0.0) + value(row, "pcon_z_2").unwrap_or(0.0),
        ];
        let h = [
            value(row, "hcon_x_1").unwrap_or(0.0) + value(row, "hcon_x_2").unwrap_or(0.0),
            value(row, "hcon_y_1").unwrap_or(0.0) + value(row, "hcon_y_2").unwrap_or(0.0),
            value(row, "hcon_z_1").unwrap_or(0.0) + value(row, "hcon_z_2").unwrap_or(0.0),
        ];
        max_dp = max_dp.max(norm(diff(p, p0)));
        max_dh = max_dh.max(norm(diff(h, h0)));
    }
    let last = rows.last().unwrap();
    let final_ke_drift_pct = 100.0 * (value(last, "ke_total").unwrap_or(ke0) - ke0) / ke0;
    let p0_norm = norm(p0);
    let h0_norm = norm(h0);
    Summary {
        status: solver_error
            .as_ref()
            .map(|_| "solver_error".to_string())
            .unwrap_or_else(|| "ok".to_string()),
        rows: rows.len(),
        max_ke_drift_pct: Some(max_ke_drift_pct),
        final_ke_drift_pct: Some(final_ke_drift_pct),
        max_dp: Some(max_dp),
        rel_dp: (p0_norm > 0.0).then_some(max_dp / p0_norm),
        max_dh: Some(max_dh),
        rel_dh: (h0_norm > 0.0).then_some(max_dh / h0_norm),
        mean_step_s: log_value(&log_text, "Mean time/step:"),
        coupled_residual: log_value(&log_text, "Coupled max residual norm:"),
        coupled_impulse_residual: log_value(&log_text, "Coupled max impulse residual:"),
        coupled_energy_error_rel: log_value(&log_text, "Coupled max energy error rel:"),
        floor_excess: log_value(&log_text, "Projection max KE floor excess:"),
        floor_hits_fallbacks: log_floor_counts(&log_text),
        error: solver_error.unwrap_or_default(),
    }
}

impl Summary {
    fn missing(status: &str, error: &str) -> Self {
        Self {
            status: status.to_string(),
            rows: 0,
            max_ke_drift_pct: None,
            final_ke_drift_pct: None,
            max_dp: None,
            rel_dp: None,
            max_dh: None,
            rel_dh: None,
            mean_step_s: None,
            coupled_residual: None,
            coupled_impulse_residual: None,
            coupled_energy_error_rel: None,
            floor_excess: None,
            floor_hits_fallbacks: None,
            error: error.to_string(),
        }
    }
}

fn write_summary(path: &Path, rows: &[(Case, Summary)]) -> io::Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "name,scheme,rho,ndiv,dt,tend,sep,status,rows,max_ke_drift_pct,final_ke_drift_pct,max_dP,rel_dP,max_dH,rel_dH,mean_step_s,coupled_residual,coupled_impulse_residual,coupled_energy_error_rel,floor_excess,floor_hits_fallbacks,error")?;
    for (case, s) in rows {
        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            case.name,
            case.scheme,
            case.rho,
            case.ndiv,
            case.dt,
            case.tend,
            case.sep,
            s.status,
            s.rows,
            csv_opt(s.max_ke_drift_pct),
            csv_opt(s.final_ke_drift_pct),
            csv_opt(s.max_dp),
            csv_opt(s.rel_dp),
            csv_opt(s.max_dh),
            csv_opt(s.rel_dh),
            csv_opt(s.mean_step_s),
            csv_opt(s.coupled_residual),
            csv_opt(s.coupled_impulse_residual),
            csv_opt(s.coupled_energy_error_rel),
            csv_opt(s.floor_excess),
            s.floor_hits_fallbacks.clone().unwrap_or_default(),
            s.error.replace(',', ";")
        )?;
    }
    Ok(())
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

fn log_floor_counts(text: &str) -> Option<String> {
    text.lines().rev().find_map(|line| {
        let pos = line.find("Projection floor hits/fallbacks:")?;
        Some(line[pos + "Projection floor hits/fallbacks:".len()..].trim().to_string())
    })
}

fn log_line(path: &Path, message: &str) -> io::Result<()> {
    let mut file = fs::OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "[{}] {}", timestamp(), message)
}

fn timestamp() -> String {
    let output = Command::new("cmd")
        .args(["/C", "echo %DATE% %TIME%"])
        .output();
    match output {
        Ok(out) => String::from_utf8_lossy(&out.stdout).trim().to_string(),
        Err(_) => "unknown-time".to_string(),
    }
}

fn rho_name(rho: f64) -> String {
    if (rho - 1.0).abs() < 1.0e-12 {
        "1".to_string()
    } else if (rho - 0.1).abs() < 1.0e-12 {
        "0p1".to_string()
    } else if (rho - 0.01).abs() < 1.0e-12 {
        "0p01".to_string()
    } else {
        format!("{rho:.6}").trim_end_matches('0').trim_end_matches('.').replace('.', "p")
    }
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn diff(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn fmt_opt(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.6e}")).unwrap_or_default()
}

fn csv_opt(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.12e}")).unwrap_or_default()
}
