from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "studies" / "single_body_endpoint_convergence"
RUNS = STUDY / "runs"
DOCS = ROOT / "docs" / "paper_figures" / "section3_clean"
BASE_INPUT = ROOT / "studies" / "ke_ratio_density" / "runs_impulse_nd2" / "ratio1_rho4_run01" / "input.txt"
TARGET_T = 10.0


@dataclass(frozen=True)
class Case:
    suite: str
    name: str
    ndiv: int
    dt: float
    reference: bool

    @property
    def run_dir(self) -> Path:
        return RUNS / self.name

    @property
    def output(self) -> Path:
        return self.run_dir / "single_body_complete.dat"

    @property
    def input(self) -> Path:
        return self.run_dir / "input.txt"

    @property
    def independent(self) -> float:
        if self.suite == "temporal":
            return self.dt
        # Each ndiv refinement halves the linear mesh scale.
        return 2.0 ** (-self.ndiv)


def token(value: float | int) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def cases() -> list[Case]:
    out: list[Case] = []
    for dt in (0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625):
        out.append(Case("temporal", f"temporal_nd3_dt{token(dt)}", 3, dt, math.isclose(dt, 0.00625)))
    for ndiv in (1, 2, 3, 4):
        out.append(Case("grid", f"grid_nd{ndiv}_dt0p05", ndiv, 0.05, ndiv == 4))
    return out


def parse_input(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def write_input(case: Case) -> None:
    values = parse_input(BASE_INPUT)
    values.update(
        {
            "ndiv": f"{case.ndiv}",
            "dt": f"{case.dt:g}",
            "tend": f"{TARGET_T:g}",
            "tprint": "1",
            "logevery": "50",
            "impulse_scheme": "1",
            "variational_scheme": "0",
            "hamiltonian_scheme": "0",
            "hamiltonian_midpoint_scheme": "0",
            "energy_projection": "0",
        }
    )
    case.run_dir.mkdir(parents=True, exist_ok=True)
    with case.input.open("w", encoding="utf-8", newline="\n") as f:
        for key, value in values.items():
            f.write(f"{key}={value}\n")


def output_complete(case: Case) -> bool:
    if not case.output.exists():
        return False
    expected = int(round(TARGET_T / case.dt)) + 1
    try:
        with case.output.open(encoding="utf-8") as f:
            rows = max(0, sum(1 for _ in f) - 1)
    except OSError:
        return False
    return rows >= expected


def fmt_hms(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def solver_exe() -> Path:
    exe = ROOT / "target" / "release" / "multi_ellip.exe"
    if exe.exists():
        return exe
    return ROOT / "target" / "release" / "multi_ellip"


def run_case(case: Case, rerun: bool) -> None:
    write_input(case)
    if output_complete(case) and not rerun:
        print(f"skip complete {case.name}", flush=True)
        return
    start = time.monotonic()
    log_path = case.run_dir / "run.log"
    cmd = [str(solver_exe()), str(case.input), str(case.run_dir)]
    print(f"$ {' '.join(cmd)}", flush=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)
    print(f"finished {case.name} in {fmt_hms(time.monotonic() - start)}", flush=True)


def read_output(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        cols: dict[str, list[float]] = {}
        for row in reader:
            for key, value in row.items():
                if key is None:
                    continue
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    number = math.nan
                cols.setdefault(key.strip(), []).append(number)
    return {key: np.asarray(values, dtype=float) for key, values in cols.items()}


def endpoint(case: Case) -> dict[str, np.ndarray]:
    data = read_output(case.output)
    time_col = data["time"]
    idx = int(np.argmin(np.abs(time_col - TARGET_T)))
    if abs(float(time_col[idx]) - TARGET_T) > 1.0e-9:
        raise ValueError(f"{case.output} has no endpoint at t={TARGET_T}")
    return {
        "centroid": np.array([data["px_1"][idx], data["py_1"][idx], data["pz_1"][idx]], dtype=float),
        "orientation_marker": np.array(
            [data["ofix1_1"][idx], data["ofix2_1"][idx], data["ofix3_1"][idx]], dtype=float
        ),
    }


def orientation_angle(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return math.nan
    cosang = float(np.dot(a, b) / (na * nb))
    return float(math.acos(max(-1.0, min(1.0, cosang))))


def fit_order(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if np.count_nonzero(mask) < 2:
        return math.nan
    slope, _intercept = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
    return float(slope)


def summarize(all_cases: list[Case]) -> list[dict[str, float | str]]:
    endpoints = {case.name: endpoint(case) for case in all_cases}
    refs = {
        suite: next(case for case in all_cases if case.suite == suite and case.reference)
        for suite in sorted({case.suite for case in all_cases})
    }
    rows: list[dict[str, float | str]] = []
    for case in all_cases:
        ref = refs[case.suite]
        item = endpoints[case.name]
        ref_item = endpoints[ref.name]
        centroid_error = float(np.linalg.norm(item["centroid"] - ref_item["centroid"]))
        marker_delta = item["orientation_marker"] - ref_item["orientation_marker"]
        rows.append(
            {
                "suite": case.suite,
                "case": case.name,
                "reference": case.reference,
                "ref_case": "" if case.reference else ref.name,
                "ndiv": case.ndiv,
                "dt": case.dt,
                "t": TARGET_T,
                "independent": "dt" if case.suite == "temporal" else "h",
                "independent_value": case.independent,
                "centroid_x": item["centroid"][0],
                "centroid_y": item["centroid"][1],
                "centroid_z": item["centroid"][2],
                "orientation_marker_x": item["orientation_marker"][0],
                "orientation_marker_y": item["orientation_marker"][1],
                "orientation_marker_z": item["orientation_marker"][2],
                "centroid_error": math.nan if case.reference else centroid_error,
                "orientation_marker_error": math.nan if case.reference else float(np.linalg.norm(marker_delta)),
                "orientation_angle_error_rad": math.nan
                if case.reference
                else orientation_angle(item["orientation_marker"], ref_item["orientation_marker"]),
            }
        )
    return rows


def write_rows(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, float | str]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.4), constrained_layout=True)
    specs = [
        ("temporal", "centroid_error", "Centroid temporal convergence", "time step $\\Delta t$", "centroid error"),
        (
            "temporal",
            "orientation_angle_error_rad",
            "Orientation temporal convergence",
            "time step $\\Delta t$",
            "orientation error (rad)",
        ),
        ("grid", "centroid_error", "Centroid grid convergence", "mesh scale $h$", "centroid error"),
        (
            "grid",
            "orientation_angle_error_rad",
            "Orientation grid convergence",
            "mesh scale $h$",
            "orientation error (rad)",
        ),
    ]
    order_rows: list[dict[str, float | str]] = []
    for axis, (suite, metric, title, xlabel, ylabel) in zip(axes.flat, specs):
        subset = [row for row in rows if row["suite"] == suite and str(row["reference"]) != "True"]
        x = np.asarray([float(row["independent_value"]) for row in subset], dtype=float)
        y = np.asarray([float(row[metric]) for row in subset], dtype=float)
        axis.loglog(x, y, "o-", color="#1f77b4")
        axis.invert_xaxis()
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25, which="both")
        order = fit_order(x, y)
        order_rows.append({"suite": suite, "metric": metric, "order": order})
        if math.isfinite(order):
            axis.text(0.04, 0.08, f"order {order:.2f}", transform=axis.transAxes, fontsize=9)

    fig.suptitle(r"Single ellipsoid impulse convergence at $t=10$ ($\rho=4$, $E=1$)", fontsize=12)
    STUDY.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    study_png = STUDY / "single_body_endpoint_convergence.png"
    docs_png = DOCS / "section3_single_body_endpoint_convergence.png"
    fig.savefig(study_png, dpi=240)
    plt.close(fig)
    write_rows(STUDY / "single_body_endpoint_convergence_orders.csv", order_rows)
    try:
        shutil.copyfile(study_png, docs_png)
    except PermissionError as exc:
        print(f"warning: could not copy figure to docs: {exc}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run endpoint convergence for a periodic single-body impulse case.")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    all_cases = cases()
    if not args.plot_only:
        if not solver_exe().exists():
            subprocess.run(["cargo", "build", "--release"], cwd=ROOT, check=True)
        for case in all_cases:
            run_case(case, args.rerun)
    rows = summarize(all_cases)
    write_rows(STUDY / "single_body_endpoint_convergence_summary.csv", rows)
    plot(rows)


if __name__ == "__main__":
    main()
