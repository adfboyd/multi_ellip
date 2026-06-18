import argparse
import csv
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "two_body_endpoint_subset_runs"
SOURCE_MANIFEST = ROOT / "studies" / "two_body_parameter_sweep" / "two_body_parameter_sweep_manifest.csv"
RUNS = STUDY / "runs"
SUMMARY = STUDY / "endpoint_subset_summary.csv"
BIN = ROOT / "target" / "release" / "multi_ellip.exe"

SELECTED = [
    "spheroid_1_0p7_0p7_rho1_E0p25_sep3_run01",
    "spheroid_1_0p7_0p7_rho0p1_E4_sep3_run01",
    "ellipsoid_1_0p8_0p6_rho0p1_E4_sep3_run01",
    "ellipsoid_1_0p8_0p6_rho0p03_E16_sep8_run01",
]

OVERRIDES = {
    "tend": "10.0",
    "dt": "0.05",
    "tprint": "1",
    "logevery": "20",
    "impulse_scheme": "1",
    "energy_projection": "0",
    "fluid_energy_gradient": "0",
    "hamiltonian_midpoint_scheme": "1",
    "hamiltonian_coupled_solve": "1",
    "hamiltonian_coupled_iters": "6",
    "hamiltonian_coupled_eps": "0.001",
    "hamiltonian_coupled_max_shift": "0.2",
    "hamiltonian_coupled_jacobian_interval": "6",
    "hamiltonian_coupled_broyden_update": "1",
    "hamiltonian_coupled_endpoint_velocity": "1",
    "hamiltonian_adaptive_substeps": "1",
    "hamiltonian_max_substeps": "8",
    "hamiltonian_floor_tol": "0.0001",
}


def load_manifest() -> dict[str, dict[str, str]]:
    with SOURCE_MANIFEST.open("r", encoding="utf-8", newline="") as f:
        return {row["name"]: row for row in csv.DictReader(f)}


def read_input(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text or text.startswith("#") or "=" not in text:
                continue
            key, value = text.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def write_input(path: Path, values: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for key, value in values.items():
            f.write(f"{key}={value}\n")


def prepare_case(source_row: dict[str, str]) -> tuple[Path, Path]:
    run_dir = RUNS / source_row["name"]
    input_path = run_dir / "input.txt"
    out_dir = run_dir / "out"
    values = read_input(Path(source_row["input"]))
    values.update(OVERRIDES)
    write_input(input_path, values)
    return input_path, out_dir


def output_path(run_dir: Path) -> Path:
    return run_dir / "out" / "multiple_body_complete.dat"


def output_complete(path: Path, tend: float, dt: float) -> bool:
    if not path.exists():
        return False
    expected_rows = int(round(tend / dt)) + 1
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f) >= expected_rows + 1


def log_value(text: str, label: str) -> str:
    for line in reversed(text.splitlines()):
        pos = line.find(label)
        if pos >= 0:
            parts = line[pos + len(label) :].split()
            if parts:
                return parts[0]
    return ""


def summarize_dat(path: Path) -> tuple[int, float, float]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        rows = list(reader)
    if not rows:
        return 0, float("nan"), float("nan")
    ke0 = float(rows[0]["ke_total"])
    drifts = [100.0 * (float(row["ke_total"]) / ke0 - 1.0) for row in rows]
    return len(rows), max(abs(x) for x in drifts), drifts[-1]


def run_case(source_row: dict[str, str], rerun: bool) -> dict[str, str]:
    input_path, out_dir = prepare_case(source_row)
    run_dir = input_path.parent
    log_path = run_dir / "run.log"
    data_path = output_path(run_dir)
    tend = float(OVERRIDES["tend"])
    dt = float(OVERRIDES["dt"])

    if rerun and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not output_complete(data_path, tend, dt):
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [str(BIN), str(input_path), str(out_dir)],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        status = "ok" if result.returncode == 0 else f"exit_{result.returncode}"
    else:
        status = "cached"

    log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    if "Solver stopped with error:" in log_text:
        status = "solver_error"
    rows, max_ke, final_ke = summarize_dat(data_path) if data_path.exists() else (0, float("nan"), float("nan"))

    return {
        "name": source_row["name"],
        "shape_name": source_row["shape_name"],
        "rho": source_row["rho"],
        "energy_ratio": source_row["energy_ratio"],
        "separation": source_row["separation"],
        "status": status,
        "rows": str(rows),
        "max_ke_drift_pct": f"{max_ke:.12e}",
        "final_ke_drift_pct": f"{final_ke:.12e}",
        "mean_step_s": log_value(log_text, "Mean time/step:"),
        "coupled_residual": log_value(log_text, "Coupled max residual norm:"),
        "coupled_scaled_impulse_residual": log_value(
            log_text, "Coupled max scaled impulse residual:"
        ),
        "coupled_raw_linear_impulse_residual": log_value(
            log_text, "Coupled max raw linear impulse residual:"
        ),
        "coupled_raw_angular_impulse_residual": log_value(
            log_text, "Coupled max raw angular impulse residual:"
        ),
        "coupled_true_energy_error_rel": log_value(
            log_text, "Coupled max true energy error rel:"
        ),
    }


def write_summary(rows: list[dict[str, str]]) -> None:
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "name",
        "shape_name",
        "rho",
        "energy_ratio",
        "separation",
        "status",
        "rows",
        "max_ke_drift_pct",
        "final_ke_drift_pct",
        "mean_step_s",
        "coupled_residual",
        "coupled_scaled_impulse_residual",
        "coupled_raw_linear_impulse_residual",
        "coupled_raw_angular_impulse_residual",
        "coupled_true_energy_error_rel",
    ]
    with SUMMARY.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small endpoint-velocity subset of the two-body sweep.")
    parser.add_argument("--rerun", action="store_true", help="remove existing outputs before running")
    args = parser.parse_args()

    if not BIN.exists():
        raise SystemExit(f"missing release binary: {BIN}")

    manifest = load_manifest()
    rows = []
    for name in SELECTED:
        if name not in manifest:
            raise SystemExit(f"missing selected case in manifest: {name}")
        print(f"running {name}", flush=True)
        row = run_case(manifest[name], args.rerun)
        rows.append(row)
        print(
            f"  {row['status']} maxKE={row['max_ke_drift_pct']} "
            f"step={row['mean_step_s']} residual={row['coupled_residual']}",
            flush=True,
        )
    write_summary(rows)
    print(f"summary: {SUMMARY}")


if __name__ == "__main__":
    main()
