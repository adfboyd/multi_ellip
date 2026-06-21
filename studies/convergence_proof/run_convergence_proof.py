from __future__ import annotations

import argparse
import csv
import math
import re
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
STUDY = ROOT / "studies" / "convergence_proof"
RUNS = STUDY / "runs"


SINGLE_BASE = {
    "cex1": 0.0,
    "cey1": 0.0,
    "cez1": 0.0,
    "oriw1": 1.0,
    "orii1": 2.0,
    "orij1": 0.0,
    "orik1": 0.0,
    "lvx1": -1.0,
    "lvy1": 0.0,
    "lvz1": 0.0,
    "avx1": 1.0,
    "avy1": 1.0,
    "avz1": 0.0,
    "shx1": 1.0,
    "shy1": 0.8,
    "shz1": 0.6,
    "req1": 1.0,
    "rhos1": 1.0,
    "rhof": 1.0,
    "tprint": 1,
    "logevery": 100,
    "nbody": 1,
    "impulse_scheme": 1,
}


MULTI_BASE = {
    "cex1": 4.0,
    "cey1": 0.0,
    "cez1": 0.0,
    "oriw1": 1.0,
    "orii1": 2.0,
    "orij1": 0.0,
    "orik1": 0.0,
    "lvx1": -0.4,
    "lvy1": 0.1,
    "lvz1": 0.0,
    "avx1": 1.0,
    "avy1": 1.0,
    "avz1": 0.0,
    "shx1": 1.0,
    "shy1": 0.8,
    "shz1": 0.6,
    "req1": 1.0,
    "rhos1": 1.0,
    "cex2": -4.0,
    "cey2": 1.0,
    "cez2": 0.0,
    "oriw2": 1.0,
    "orii2": 0.0,
    "orij2": 1.0,
    "orik2": 0.0,
    "lvx2": -0.4,
    "lvy2": 0.1,
    "lvz2": 0.0,
    "avx2": 0.0,
    "avy2": 1.0,
    "avz2": 1.0,
    "shx2": 1.0,
    "shy2": 0.8,
    "shz2": 0.6,
    "req2": 1.0,
    "rhos2": 1.0,
    "rhof": 1.0,
    "tprint": 1,
    "logevery": 100,
    "nbody": 2,
    "impulse_scheme": 1,
}


@dataclass(frozen=True)
class Case:
    suite: str
    name: str
    nbody: int
    ndiv: int
    dt: float
    tend: float
    independent: str
    is_reference: bool = False

    @property
    def output_name(self) -> str:
        return "single_body_complete.dat" if self.nbody == 1 else "multiple_body_complete.dat"

    @property
    def independent_value(self) -> float:
        if self.independent == "dt":
            return self.dt
        if self.independent == "h":
            # Each ndiv refinement halves the linear surface spacing.
            return 2.0 ** (-self.ndiv)
        raise ValueError(self.independent)

    @property
    def run_dir(self) -> Path:
        return RUNS / self.name


def label_float(x: float) -> str:
    text = f"{x:g}".replace("-", "m").replace(".", "p")
    return text.replace("+", "")


def cases() -> list[Case]:
    out: list[Case] = []

    # Time convergence: fixed mesh, decreasing dt, finest dt as reference.
    for system, nbody, ndiv in (("single", 1, 3), ("multi", 2, 2)):
        suite = f"{system}_time"
        for dt in (0.2, 0.1, 0.05, 0.025, 0.0125):
            out.append(
                Case(
                    suite=suite,
                    name=f"{suite}_nd{ndiv}_dt{label_float(dt)}",
                    nbody=nbody,
                    ndiv=ndiv,
                    dt=dt,
                    tend=10.0,
                    independent="dt",
                    is_reference=math.isclose(dt, 0.0125),
                )
            )

    # Grid convergence: fixed small dt, increasing ndiv, finest ndiv as reference.
    for ndiv in (1, 2, 3, 4):
        out.append(
            Case(
                suite="single_grid",
                name=f"single_grid_nd{ndiv}_dt0p0125",
                nbody=1,
                ndiv=ndiv,
                dt=0.0125,
                tend=5.0,
                independent="h",
                is_reference=ndiv == 4,
            )
        )

    # Two-body grid convergence is deliberately shorter and uses ndiv=3 as the
    # reference. ndiv=4 two-body runs are too expensive for a routine proof run.
    for ndiv in (1, 2, 3):
        out.append(
            Case(
                suite="multi_grid",
                name=f"multi_grid_nd{ndiv}_dt0p0125",
                nbody=2,
                ndiv=ndiv,
                dt=0.0125,
                tend=5.0,
                independent="h",
                is_reference=ndiv == 3,
            )
        )

    return out


def fmt_hms(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def input_values(case: Case) -> dict[str, float]:
    values = dict(SINGLE_BASE if case.nbody == 1 else MULTI_BASE)
    values["ndiv"] = case.ndiv
    values["dt"] = case.dt
    values["tend"] = case.tend
    return values


def write_input(case: Case) -> Path:
    case.run_dir.mkdir(parents=True, exist_ok=True)
    path = case.run_dir / "input.txt"
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for key, value in input_values(case).items():
            f.write(f"{key}={value}\n")
    return path


def output_file(case: Case) -> Path:
    return case.run_dir / case.output_name


def output_complete(case: Case) -> bool:
    path = output_file(case)
    if not path.exists():
        return False
    expected_lines = int(round(case.tend / case.dt)) + 2
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f) >= expected_lines


def stream_run(cmd: list[str], log_path: Path) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    started = time.monotonic()
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
    print(f"Finished in {fmt_hms(time.monotonic() - started)}", flush=True)


def load(case: Case) -> np.ndarray:
    data = np.genfromtxt(output_file(case), delimiter=",", names=True)
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    return data


def vec(data: np.ndarray, prefix: str, body: int) -> np.ndarray:
    return np.column_stack(
        [data[f"{prefix}_x_{body}"], data[f"{prefix}_y_{body}"], data[f"{prefix}_z_{body}"]]
    )


def pos(data: np.ndarray, body: int) -> np.ndarray:
    return np.column_stack([data[f"px_{body}"], data[f"py_{body}"], data[f"pz_{body}"]])


def vel(data: np.ndarray, body: int) -> np.ndarray:
    return np.column_stack([data[f"vx_{body}"], data[f"vy_{body}"], data[f"vz_{body}"]])


def omega(data: np.ndarray, body: int) -> np.ndarray:
    return np.column_stack([data[f"w1_{body}"], data[f"w2_{body}"], data[f"w3_{body}"]])


def quat(data: np.ndarray, body: int) -> np.ndarray:
    return np.column_stack(
        [data[f"q0_{body}"], data[f"q1_{body}"], data[f"q2_{body}"], data[f"q3_{body}"]]
    )


def stack_by_body(fn, data: np.ndarray, nbody: int) -> np.ndarray:
    return np.stack([fn(data, b) for b in range(1, nbody + 1)], axis=1)


def time_indices(ref: np.ndarray, sample: np.ndarray) -> np.ndarray:
    ref_map = {round(float(t), 10): i for i, t in enumerate(ref["time"])}
    out = []
    missing = []
    for t in sample["time"]:
        key = round(float(t), 10)
        idx = ref_map.get(key)
        if idx is None:
            missing.append(float(t))
        else:
            out.append(idx)
    if missing:
        preview = ", ".join(f"{t:g}" for t in missing[:5])
        raise RuntimeError(f"reference is missing {len(missing)} sample times: {preview}")
    return np.asarray(out, dtype=int)


def rms_norm(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(x * x, axis=tuple(range(1, x.ndim))))))


def max_norm(x: np.ndarray) -> float:
    return float(np.max(np.sqrt(np.sum(x * x, axis=tuple(range(1, x.ndim))))))


def orientation_error(sample_q: np.ndarray, ref_q: np.ndarray) -> np.ndarray:
    dots = np.abs(np.sum(sample_q * ref_q, axis=2))
    dots = np.clip(dots, 0.0, 1.0)
    return 2.0 * np.arccos(dots)


def kinetic_drift(data: np.ndarray) -> tuple[float, float]:
    drift = 100.0 * (data["ke_total"] - data["ke_total"][0]) / data["ke_total"][0]
    return float(drift[-1]), float(np.max(np.abs(drift)))


def parse_wall_seconds(log_path: Path) -> float:
    if not log_path.exists():
        return math.nan
    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Total wall time:\s+(.+)", text)
    if not match:
        return math.nan
    value = match.group(1).strip()
    total = 0.0
    for amount, unit in re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*([hms])", value):
        scale = {"h": 3600.0, "m": 60.0, "s": 1.0}[unit]
        total += float(amount) * scale
    return total if total > 0.0 else math.nan


def summarize_case(case: Case, data: np.ndarray, ref_case: Case | None, ref_data: np.ndarray | None) -> dict[str, object]:
    end_drift, max_drift = kinetic_drift(data)
    base = {
        "suite": case.suite,
        "case": case.name,
        "reference": case.is_reference,
        "ref_case": ref_case.name if ref_case else "",
        "nbody": case.nbody,
        "ndiv": case.ndiv,
        "dt": case.dt,
        "tend": case.tend,
        "independent": case.independent,
        "independent_value": case.independent_value,
        "rows": len(data),
        "common_rows": len(data),
        "ke_end_drift_pct": end_drift,
        "ke_max_abs_drift_pct": max_drift,
        "wall_seconds": parse_wall_seconds(case.run_dir / "run.log"),
    }
    error_fields = {
        "pos_rms": math.nan,
        "pos_max": math.nan,
        "vel_rms": math.nan,
        "vel_max": math.nan,
        "omega_rms": math.nan,
        "omega_max": math.nan,
        "orient_rms_rad": math.nan,
        "orient_max_rad": math.nan,
        "combined_rms": math.nan,
        "combined_max": math.nan,
    }
    if ref_data is None:
        base.update(error_fields)
        return base

    idx = time_indices(ref_data, data)
    ref_at = ref_data[idx]

    p = stack_by_body(pos, data, case.nbody)
    p_ref = stack_by_body(pos, ref_at, case.nbody)
    v = stack_by_body(vel, data, case.nbody)
    v_ref = stack_by_body(vel, ref_at, case.nbody)
    w = stack_by_body(omega, data, case.nbody)
    w_ref = stack_by_body(omega, ref_at, case.nbody)
    q = stack_by_body(quat, data, case.nbody)
    q_ref = stack_by_body(quat, ref_at, case.nbody)

    dp = p - p_ref
    dv = v - v_ref
    dw = w - w_ref
    dq = orientation_error(q, q_ref)

    p_ref_all = stack_by_body(pos, ref_data, case.nbody)
    v_ref_all = stack_by_body(vel, ref_data, case.nbody)
    w_ref_all = stack_by_body(omega, ref_data, case.nbody)
    pos_scale = max(1.0, rms_norm(p_ref_all - p_ref_all[0]))
    vel_scale = max(1.0, rms_norm(v_ref_all))
    omega_scale = max(1.0, rms_norm(w_ref_all))

    pos_rms = rms_norm(dp)
    pos_max = max_norm(dp)
    vel_rms = rms_norm(dv)
    vel_max = max_norm(dv)
    omega_rms = rms_norm(dw)
    omega_max = max_norm(dw)
    orient_rms = float(np.sqrt(np.mean(np.sum(dq * dq, axis=1))))
    orient_max = float(np.max(np.sqrt(np.sum(dq * dq, axis=1))))

    combined_series = np.sqrt(
        np.sum((dp / pos_scale) ** 2, axis=(1, 2))
        + np.sum((dv / vel_scale) ** 2, axis=(1, 2))
        + np.sum((dw / omega_scale) ** 2, axis=(1, 2))
        + np.sum(dq * dq, axis=1)
    )

    base.update(
        {
            "common_rows": len(idx),
            "pos_rms": pos_rms,
            "pos_max": pos_max,
            "vel_rms": vel_rms,
            "vel_max": vel_max,
            "omega_rms": omega_rms,
            "omega_max": omega_max,
            "orient_rms_rad": orient_rms,
            "orient_max_rad": orient_max,
            "combined_rms": float(np.sqrt(np.mean(combined_series * combined_series))),
            "combined_max": float(np.max(combined_series)),
        }
    )
    return base


def fit_order(rows: list[dict[str, object]], metric: str) -> float:
    xs = []
    ys = []
    for row in rows:
        if row["reference"]:
            continue
        y = float(row[metric])
        if math.isfinite(y) and y > 0.0:
            xs.append(float(row["independent_value"]))
            ys.append(y)
    if len(xs) < 2:
        return math.nan
    coeff = np.polyfit(np.log(xs), np.log(ys), 1)
    return float(coeff[0])


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    all_cases = cases()
    data = {case.name: load(case) for case in all_cases}
    refs = {
        suite: next(case for case in all_cases if case.suite == suite and case.is_reference)
        for suite in sorted({case.suite for case in all_cases})
    }
    rows = []
    for case in all_cases:
        ref_case = refs[case.suite]
        rows.append(
            summarize_case(
                case,
                data[case.name],
                None if case.is_reference else ref_case,
                None if case.is_reference else data[ref_case.name],
            )
        )

    order_rows = []
    metrics = ["combined_rms", "pos_rms", "vel_rms", "omega_rms", "orient_rms_rad", "ke_max_abs_drift_pct"]
    for suite in sorted(refs):
        suite_rows = [row for row in rows if row["suite"] == suite]
        for metric in metrics:
            order_rows.append({"suite": suite, "metric": metric, "order": fit_order(suite_rows, metric)})

    write_csv(STUDY / "convergence_summary.csv", rows)
    write_csv(STUDY / "convergence_orders.csv", order_rows)
    return rows, order_rows


def plot(rows: list[dict[str, object]], orders: list[dict[str, object]]) -> None:
    order_map = {(row["suite"], row["metric"]): float(row["order"]) for row in orders}
    suites = ["single_time", "multi_time", "single_grid", "multi_grid"]
    titles = {
        "single_time": "Single body time convergence",
        "multi_time": "Two body time convergence",
        "single_grid": "Single body grid convergence",
        "multi_grid": "Two body grid convergence",
    }
    xlabels = {
        "single_time": "dt",
        "multi_time": "dt",
        "single_grid": "h ~ 2^-ndiv",
        "multi_grid": "h ~ 2^-ndiv",
    }
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    axes = axes.ravel()

    for ax, suite in zip(axes[:4], suites):
        subset = [row for row in rows if row["suite"] == suite and not row["reference"]]
        subset.sort(key=lambda row: float(row["independent_value"]))
        x = np.array([float(row["independent_value"]) for row in subset])
        combined = np.array([float(row["combined_rms"]) for row in subset])
        pos_e = np.array([float(row["pos_rms"]) for row in subset])
        vel_e = np.array([float(row["vel_rms"]) for row in subset])
        omg_e = np.array([float(row["omega_rms"]) for row in subset])
        ori_e = np.array([float(row["orient_rms_rad"]) for row in subset])

        ax.loglog(x, combined, "o-", label=f"state p={order_map[(suite, 'combined_rms')]:.2f}")
        ax.loglog(x, pos_e, "s--", label=f"pos p={order_map[(suite, 'pos_rms')]:.2f}")
        ax.loglog(x, vel_e, "^--", label=f"vel p={order_map[(suite, 'vel_rms')]:.2f}")
        ax.loglog(x, omg_e, "v--", label=f"omega p={order_map[(suite, 'omega_rms')]:.2f}")
        ax.loglog(x, ori_e, "d--", label=f"orient p={order_map[(suite, 'orient_rms_rad')]:.2f}")
        ax.invert_xaxis()
        ax.set_title(titles[suite])
        ax.set_xlabel(xlabels[suite])
        ax.set_ylabel("RMS error vs reference")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)

    for ax, suite_group, title, xlabel in (
        (axes[4], ["single_time", "multi_time"], "Energy drift vs dt", "dt"),
        (axes[5], ["single_grid", "multi_grid"], "Energy drift vs grid spacing", "h ~ 2^-ndiv"),
    ):
        for suite in suite_group:
            subset = [row for row in rows if row["suite"] == suite and not row["reference"]]
            subset.sort(key=lambda row: float(row["independent_value"]))
            x = np.array([float(row["independent_value"]) for row in subset])
            y = np.array([float(row["ke_max_abs_drift_pct"]) for row in subset])
            ax.loglog(x, y, "o-", label=f"{suite} p={order_map[(suite, 'ke_max_abs_drift_pct')]:.2f}")
        ax.invert_xaxis()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("max |KE drift| (%)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)

    fig.suptitle("Impulse scheme convergence against numerical references", fontsize=14)
    fig.tight_layout()
    out = STUDY / "convergence_dashboard.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out.relative_to(ROOT)}")


def print_summary(rows: list[dict[str, object]], orders: list[dict[str, object]]) -> None:
    print()
    print("Convergence orders from RMS errors vs numerical reference")
    print("suite           state    pos      vel      omega    orient   KE drift")
    for suite in ("single_time", "multi_time", "single_grid", "multi_grid"):
        vals = {
            row["metric"]: float(row["order"])
            for row in orders
            if row["suite"] == suite
        }
        print(
            f"{suite:<14} "
            f"{vals['combined_rms']:>7.3f} "
            f"{vals['pos_rms']:>8.3f} "
            f"{vals['vel_rms']:>8.3f} "
            f"{vals['omega_rms']:>8.3f} "
            f"{vals['orient_rms_rad']:>8.3f} "
            f"{vals['ke_max_abs_drift_pct']:>9.3f}"
        )
    print()
    print("Non-reference cases")
    print("case                              x            state_rms      KE max drift %")
    for row in rows:
        if row["reference"]:
            continue
        print(
            f"{row['case']:<33} "
            f"{float(row['independent_value']):>10.5g} "
            f"{float(row['combined_rms']):>14.6e} "
            f"{float(row['ke_max_abs_drift_pct']):>15.6f}"
        )


def run_cases(args: argparse.Namespace) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    exe = Path(args.exe) if args.exe else ROOT / "target" / "release" / "multi_ellip.exe"
    if not exe.exists():
        alt = ROOT / "target" / "release" / "multi_ellip"
        exe = alt if alt.exists() else exe
    if not exe.exists():
        raise FileNotFoundError(f"solver executable not found: {exe}")

    selected = [case for case in cases() if not args.suite or case.suite in args.suite]
    started = time.monotonic()
    for idx, case in enumerate(selected, start=1):
        write_input(case)
        print()
        print("=" * 76)
        print(
            f"Case {idx}/{len(selected)}: {case.name} "
            f"(suite={case.suite}, nbody={case.nbody}, ndiv={case.ndiv}, dt={case.dt}, t={case.tend})"
        )
        print(f"Elapsed study time: {fmt_hms(time.monotonic() - started)}")
        print("=" * 76)
        if output_complete(case) and not args.rerun:
            print("Output already complete; skipping. Use --rerun to regenerate.", flush=True)
            continue
        stream_run([str(exe), str(case.run_dir / "input.txt"), str(case.run_dir)], case.run_dir / "run.log")
    print(f"Run phase elapsed: {fmt_hms(time.monotonic() - started)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trajectory-based grid/time convergence proof.")
    parser.add_argument("--rerun", action="store_true", help="rerun cases even if output exists")
    parser.add_argument("--plot-only", action="store_true", help="only summarize and plot existing outputs")
    parser.add_argument("--exe", default="", help="solver executable; defaults to target/release/multi_ellip.exe")
    parser.add_argument(
        "--suite",
        action="append",
        choices=["single_time", "multi_time", "single_grid", "multi_grid"],
        help="run only selected suite(s); may be passed multiple times",
    )
    args = parser.parse_args()

    STUDY.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        run_cases(args)
    rows, orders = summarize()
    plot(rows, orders)
    print(f"Saved {(STUDY / 'convergence_summary.csv').relative_to(ROOT)}")
    print(f"Saved {(STUDY / 'convergence_orders.csv').relative_to(ROOT)}")
    print_summary(rows, orders)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}: {' '.join(exc.cmd)}", file=sys.stderr)
        raise
