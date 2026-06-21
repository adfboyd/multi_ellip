from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "studies" / "exact_singular_validation"
ENERGY_RUNS = STUDY / "energy_runs"
CONV = STUDY / "convergence"


BASE_TWO_BODY = {
    "cex1": -1.5,
    "cey1": 0.0,
    "cez1": 0.0,
    "oriw1": 1.0,
    "orii1": 2.0,
    "orij1": 0.0,
    "orik1": 0.0,
    "lvx1": -1.0,
    "lvy1": 0.2,
    "lvz1": 0.1,
    "avx1": 0.4,
    "avy1": -0.7,
    "avz1": 0.2,
    "shx1": 1.0,
    "shy1": 0.8,
    "shz1": 0.6,
    "req1": 1.0,
    "rhos1": 1.0,
    "cex2": 1.5,
    "cey2": 0.0,
    "cez2": 0.0,
    "oriw2": 1.0,
    "orii2": 0.0,
    "orij2": 1.0,
    "orik2": 0.0,
    "lvx2": -0.9,
    "lvy2": 0.1,
    "lvz2": 0.2,
    "avx2": -0.6,
    "avy2": -0.1,
    "avz2": -0.3,
    "shx2": 1.0,
    "shy2": 0.8,
    "shz2": 0.6,
    "req2": 1.0,
    "rhos2": 1.0,
    "rhof": 1.0,
    "tprint": 1,
    "logevery": 10,
    "nbody": 2,
    "impulse_scheme": 1,
    "energy_projection": 1,
}


@dataclass(frozen=True)
class EnergyCase:
    name: str
    ndiv: int
    dt: float
    tend: float
    exact: bool
    projection: bool

    @property
    def run_dir(self) -> Path:
        return ENERGY_RUNS / self.name


ENERGY_CASES = [
    EnergyCase("default_nd2_dt0p1_t2", 2, 0.1, 2.0, False, True),
    EnergyCase("exact_nd2_dt0p2_t2", 2, 0.2, 2.0, True, True),
    EnergyCase("exact_nd2_dt0p1_t2", 2, 0.1, 2.0, True, True),
    EnergyCase("exact_nd2_dt0p05_t2", 2, 0.05, 2.0, True, True),
    EnergyCase("exact_nd3_dt0p1_t1", 3, 0.1, 1.0, True, True),
    EnergyCase("default_nd2_dt0p1_t2_noproj", 2, 0.1, 2.0, False, False),
    EnergyCase("exact_nd2_dt0p2_t2_noproj", 2, 0.2, 2.0, True, False),
    EnergyCase("exact_nd2_dt0p1_t2_noproj", 2, 0.1, 2.0, True, False),
    EnergyCase("exact_nd2_dt0p05_t2_noproj", 2, 0.05, 2.0, True, False),
    EnergyCase("exact_nd3_dt0p1_t1_noproj", 3, 0.1, 1.0, True, False),
]


def fmt_hms(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def run(cmd: list[str], log_path: Path | None = None) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    start = time.monotonic()
    log = log_path.open("w", encoding="utf-8", newline="\n") if log_path else None
    try:
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
            if log:
                log.write(line)
                log.flush()
        rc = proc.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)
    finally:
        if log:
            log.close()
    print(f"Finished in {fmt_hms(time.monotonic() - start)}", flush=True)


def solver_exe() -> Path:
    exe = ROOT / "target" / "release" / "multi_ellip.exe"
    if not exe.exists():
        exe = ROOT / "target" / "release" / "multi_ellip"
    return exe


def write_energy_input(case: EnergyCase) -> Path:
    case.run_dir.mkdir(parents=True, exist_ok=True)
    values = dict(BASE_TWO_BODY)
    values.update(
        {
            "ndiv": case.ndiv,
            "dt": case.dt,
            "tend": case.tend,
            "exact_ellipsoid_geometry": 1 if case.exact else 0,
            "exact_singular_geometry": 1 if case.exact else 0,
            "energy_projection": 1 if case.projection else 0,
        }
    )
    path = case.run_dir / "input.txt"
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for key, value in values.items():
            f.write(f"{key}={value}\n")
    return path


def energy_output(case: EnergyCase) -> Path:
    return case.run_dir / "multiple_body_complete.dat"


def energy_complete(case: EnergyCase) -> bool:
    path = energy_output(case)
    if not path.exists():
        return False
    expected = int(round(case.tend / case.dt)) + 2
    return sum(1 for _ in path.open(encoding="utf-8")) >= expected


def run_energy_cases(rerun: bool) -> None:
    run(["cargo", "build", "--release"])
    exe = solver_exe()
    for i, case in enumerate(ENERGY_CASES, start=1):
        write_energy_input(case)
        print()
        print("=" * 76)
        print(
            f"Energy case {i}/{len(ENERGY_CASES)}: {case.name} "
            f"(ndiv={case.ndiv}, dt={case.dt}, t={case.tend}, "
            f"exact={case.exact}, projection={case.projection})"
        )
        print("=" * 76)
        if energy_complete(case) and not rerun:
            print("Output already complete; skipping. Use --rerun to regenerate.")
            continue
        run([str(exe), str(case.run_dir / "input.txt"), str(case.run_dir)], case.run_dir / "run.log")


def read_dat(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    return data


def parse_wall_seconds(log_path: Path) -> float:
    if not log_path.exists():
        return math.nan
    text = log_path.read_text(encoding="utf-8", errors="replace")
    marker = "Total wall time:"
    for line in text.splitlines():
        if marker not in line:
            continue
        value = line.split(marker, 1)[1].strip()
        total = 0.0
        for token in value.split():
            try:
                if token.endswith("h"):
                    total += 3600.0 * float(token[:-1])
                elif token.endswith("m"):
                    total += 60.0 * float(token[:-1])
                elif token.endswith("s"):
                    total += float(token[:-1])
            except ValueError:
                pass
        return total
    return math.nan


def summarize_energy() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in ENERGY_CASES:
        data = read_dat(energy_output(case))
        drift = 100.0 * (data["ke_total"] - data["ke_total"][0]) / data["ke_total"][0]
        rows.append(
            {
                "case": case.name,
                "exact_singular": case.exact,
                "energy_projection": case.projection,
                "ndiv": case.ndiv,
                "dt": case.dt,
                "tend": case.tend,
                "rows": len(data),
                "ke0": float(data["ke_total"][0]),
                "ke_end": float(data["ke_total"][-1]),
                "ke_end_drift_pct": float(drift[-1]),
                "ke_max_abs_drift_pct": float(np.max(np.abs(drift))),
                "wall_seconds": parse_wall_seconds(case.run_dir / "run.log"),
            }
        )
    out = STUDY / "energy_summary.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def run_convergence(rerun: bool) -> None:
    CONV.mkdir(parents=True, exist_ok=True)
    run(["cargo", "build", "--release", "--bin", "bem_operator_probe", "--bin", "density_collocation_probe"])
    target = ROOT / "target" / "release"
    op_csv = CONV / "sphere_operator.csv"
    sphere_density = CONV / "sphere_density_exact_singular.csv"
    ellipsoid_density = CONV / "ellipsoid_density_exact_singular.csv"
    if rerun or not op_csv.exists():
        run([str(target / "bem_operator_probe.exe"), str(op_csv), "4"])
    if rerun or not sphere_density.exists():
        run([str(target / "density_collocation_probe.exe"), str(sphere_density), "4", "0", "sphere", "exact_singular"])
    if rerun or not ellipsoid_density.exists():
        run([str(target / "density_collocation_probe.exe"), str(ellipsoid_density), "3", "0", "ellipsoid", "exact_singular"])


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fit_order(points: list[tuple[int, float]]) -> float:
    filtered = [(2.0 ** (-n), y) for n, y in points if math.isfinite(y) and y > 0.0]
    if len(filtered) < 2:
        return math.nan
    x = np.log([p[0] for p in filtered])
    y = np.log([p[1] for p in filtered])
    return float(np.polyfit(x, y, 1)[0])


def convergence_summary() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    op_rows = [r for r in read_csv(CONV / "sphere_operator.csv") if r["mode"] == "exact_singular"]
    for metric in ("rhs_rel_error", "a_phi_rel_error", "solved_phi_rel_error"):
        points = [(int(r["ndiv"]), float(r[metric])) for r in op_rows]
        rows.append({"source": "sphere_operator", "metric": metric, "order": fit_order(points)})

    for path, source in (
        (CONV / "sphere_density_exact_singular.csv", "sphere_density"),
        (CONV / "ellipsoid_density_exact_singular.csv", "ellipsoid_density"),
    ):
        data = read_csv(path)
        for metric in (
            "solved_added_rel_error",
            "analytic_phi_added_rel_error",
            "phi_rel_error",
            "analytic_residual_rel",
        ):
            by_ndiv: dict[int, list[float]] = {}
            for row in data:
                by_ndiv.setdefault(int(row["ndiv"]), []).append(float(row[metric]))
            points = [(n, float(np.mean(vals))) for n, vals in sorted(by_ndiv.items())]
            rows.append({"source": source, "metric": metric, "order": fit_order(points)})

    out = STUDY / "convergence_orders.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_energy() -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for case in ENERGY_CASES:
        data = read_dat(energy_output(case))
        drift = 100.0 * (data["ke_total"] - data["ke_total"][0]) / data["ke_total"][0]
        label = f"{case.name}"
        ax.plot(data["time"], drift, marker="o", markersize=3, linewidth=1.2, label=label)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("time")
    ax.set_ylabel("total KE drift (%)")
    ax.set_title("Two-body energy drift")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(STUDY / "energy_drift.png", dpi=160)
    plt.close(fig)


def plot_convergence() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    op_rows = [r for r in read_csv(CONV / "sphere_operator.csv") if r["mode"] == "exact_singular"]
    for metric, label in (
        ("rhs_rel_error", "RHS"),
        ("a_phi_rel_error", "A phi"),
        ("solved_phi_rel_error", "solved phi"),
    ):
        pts = sorted((int(r["ndiv"]), float(r[metric])) for r in op_rows)
        axes[0].semilogy([p[0] for p in pts], [p[1] for p in pts], "o-", label=label)
    axes[0].set_title("Unit sphere operator")

    for ax, path, title in (
        (axes[1], CONV / "sphere_density_exact_singular.csv", "Sphere added inertia"),
        (axes[2], CONV / "ellipsoid_density_exact_singular.csv", "Ellipsoid added inertia"),
    ):
        data = read_csv(path)
        for metric, label in (
            ("solved_added_rel_error", "solved added"),
            ("phi_rel_error", "phi"),
            ("analytic_residual_rel", "analytic residual"),
        ):
            by_ndiv: dict[int, list[float]] = {}
            for row in data:
                by_ndiv.setdefault(int(row["ndiv"]), []).append(float(row[metric]))
            pts = sorted((n, float(np.mean(vals))) for n, vals in by_ndiv.items())
            ax.semilogy([p[0] for p in pts], [p[1] for p in pts], "o-", label=label)
        ax.set_title(title)

    for ax in axes:
        ax.set_xlabel("ndiv")
        ax.set_ylabel("relative error")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)

    fig.savefig(STUDY / "convergence.png", dpi=160)
    plt.close(fig)


def print_summary(energy_rows: list[dict[str, object]], order_rows: list[dict[str, object]]) -> None:
    print()
    print("Energy summary")
    print("case                         max drift %      end drift %      wall s")
    for row in energy_rows:
        print(
            f"{str(row['case']):<28} "
            f"{float(row['ke_max_abs_drift_pct']):>11.6f} "
            f"{float(row['ke_end_drift_pct']):>15.6f} "
            f"{float(row['wall_seconds']):>11.2f}"
        )
    print()
    print("Convergence orders")
    for row in order_rows:
        print(f"{row['source']:<20} {row['metric']:<32} {float(row['order']):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--energy-only", action="store_true")
    parser.add_argument("--convergence-only", action="store_true")
    args = parser.parse_args()

    STUDY.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        if not args.convergence_only:
            run_energy_cases(args.rerun)
        if not args.energy_only:
            run_convergence(args.rerun)

    energy_rows = summarize_energy()
    order_rows = convergence_summary()
    plot_energy()
    plot_convergence()
    print_summary(energy_rows, order_rows)
    print(f"\nSaved outputs under {STUDY}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}: {' '.join(exc.cmd)}", file=sys.stderr)
        raise
