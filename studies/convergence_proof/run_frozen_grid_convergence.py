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

from run_convergence_proof import (
    MULTI_BASE,
    ROOT,
    SINGLE_BASE,
    STUDY,
    fmt_hms,
    label_float,
)


RUNS = STUDY / "frozen_grid_runs"


@dataclass(frozen=True)
class FrozenCase:
    system: str
    sample_time: float
    sample_label: str
    nbody: int
    ndiv: int
    reference: bool
    state: dict[str, float]

    @property
    def name(self) -> str:
        return f"{self.system}_t{self.sample_label}_nd{self.ndiv}"

    @property
    def run_dir(self) -> Path:
        return RUNS / self.name

    @property
    def output_name(self) -> str:
        return "single_body_complete.dat" if self.nbody == 1 else "multiple_body_complete.dat"

    @property
    def h(self) -> float:
        return 2.0 ** (-self.ndiv)


def load_data(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    return data


def row_at_time(data: np.ndarray, t: float) -> np.void:
    idx = int(np.argmin(np.abs(data["time"] - t)))
    if abs(float(data["time"][idx]) - t) > 1.0e-9:
        raise RuntimeError(f"no row at t={t:g}; nearest is {data['time'][idx]:g}")
    return data[idx]


def state_from_row(row: np.void, nbody: int) -> dict[str, float]:
    values: dict[str, float] = {}
    for b in range(1, nbody + 1):
        values[f"cex{b}"] = float(row[f"px_{b}"])
        values[f"cey{b}"] = float(row[f"py_{b}"])
        values[f"cez{b}"] = float(row[f"pz_{b}"])
        values[f"lvx{b}"] = float(row[f"vx_{b}"])
        values[f"lvy{b}"] = float(row[f"vy_{b}"])
        values[f"lvz{b}"] = float(row[f"vz_{b}"])
        values[f"avx{b}"] = float(row[f"w1_{b}"])
        values[f"avy{b}"] = float(row[f"w2_{b}"])
        values[f"avz{b}"] = float(row[f"w3_{b}"])
        values[f"oriw{b}"] = float(row[f"q0_{b}"])
        values[f"orii{b}"] = float(row[f"q1_{b}"])
        values[f"orij{b}"] = float(row[f"q2_{b}"])
        values[f"orik{b}"] = float(row[f"q3_{b}"])
    return values


def build_cases() -> list[FrozenCase]:
    source = {
        "single": (
            STUDY / "runs" / "single_time_nd3_dt0p0125" / "single_body_complete.dat",
            1,
            (1, 2, 3, 4),
            4,
        ),
        "multi": (
            STUDY / "runs" / "multi_time_nd2_dt0p0125" / "multiple_body_complete.dat",
            2,
            (1, 2, 3, 4),
            4,
        ),
    }
    sample_times = (0.0, 2.5, 5.0, 7.5, 10.0)
    out: list[FrozenCase] = []
    for system, (path, nbody, ndivs, ref_ndiv) in source.items():
        data = load_data(path)
        for t in sample_times:
            state = state_from_row(row_at_time(data, t), nbody)
            sample_label = label_float(t)
            for ndiv in ndivs:
                out.append(
                    FrozenCase(
                        system=system,
                        sample_time=t,
                        sample_label=sample_label,
                        nbody=nbody,
                        ndiv=ndiv,
                        reference=ndiv == ref_ndiv,
                        state=state,
                    )
                )
    return out


def write_input(case: FrozenCase) -> Path:
    case.run_dir.mkdir(parents=True, exist_ok=True)
    values = dict(SINGLE_BASE if case.nbody == 1 else MULTI_BASE)
    values.update(case.state)
    values["ndiv"] = case.ndiv
    values["dt"] = 0.001
    values["tend"] = 0.001
    values["tprint"] = 1
    values["logevery"] = 1
    path = case.run_dir / "input.txt"
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for key, value in values.items():
            f.write(f"{key}={value}\n")
    return path


def output_complete(case: FrozenCase) -> bool:
    path = case.run_dir / case.output_name
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f) >= 3


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


def initial_output(case: FrozenCase) -> np.void:
    data = load_data(case.run_dir / case.output_name)
    return data[0]


def stacked(row: np.void, prefix: str, nbody: int) -> np.ndarray:
    vals = []
    for b in range(1, nbody + 1):
        vals.extend([float(row[f"{prefix}_x_{b}"]), float(row[f"{prefix}_y_{b}"]), float(row[f"{prefix}_z_{b}"])])
    return np.asarray(vals)


def quantities(row: np.void, nbody: int) -> dict[str, np.ndarray | float]:
    return {
        "ke_fluid": float(row["ke_fluid"]),
        "lfluid": stacked(row, "lfluid", nbody),
        "lambda": stacked(row, "lambdafluid", nbody),
        "pcon": stacked(row, "pcon", nbody),
        "hcon": stacked(row, "hcon", nbody),
    }


def rel_err(value: np.ndarray | float, ref: np.ndarray | float) -> float:
    if isinstance(value, float):
        return abs(value - float(ref)) / max(abs(float(ref)), 1.0)
    diff = np.asarray(value) - np.asarray(ref)
    return float(np.linalg.norm(diff) / max(np.linalg.norm(ref), 1.0))


def summarize() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    cases = build_cases()
    by_key = {(c.system, c.sample_time, c.ndiv): c for c in cases}
    ref_ndiv = {
        system: next(c.ndiv for c in cases if c.system == system and c.reference)
        for system in sorted({c.system for c in cases})
    }
    rows = []
    for case in cases:
        row = initial_output(case)
        q = quantities(row, case.nbody)
        ref_case = by_key[(case.system, case.sample_time, ref_ndiv[case.system])]
        ref_q = quantities(initial_output(ref_case), case.nbody)
        rows.append(
            {
                "system": case.system,
                "sample_time": case.sample_time,
                "case": case.name,
                "reference": case.reference,
                "ref_case": "" if case.reference else ref_case.name,
                "nbody": case.nbody,
                "ndiv": case.ndiv,
                "h": case.h,
                "ke_fluid": q["ke_fluid"],
                "ke_fluid_rel_error": math.nan if case.reference else rel_err(q["ke_fluid"], ref_q["ke_fluid"]),
                "lfluid_rel_error": math.nan if case.reference else rel_err(q["lfluid"], ref_q["lfluid"]),
                "lambda_rel_error": math.nan if case.reference else rel_err(q["lambda"], ref_q["lambda"]),
                "pcon_rel_error": math.nan if case.reference else rel_err(q["pcon"], ref_q["pcon"]),
                "hcon_rel_error": math.nan if case.reference else rel_err(q["hcon"], ref_q["hcon"]),
            }
        )

    agg_rows = []
    for system in ("single", "multi"):
        for ndiv in sorted({c.ndiv for c in cases if c.system == system}):
            subset = [r for r in rows if r["system"] == system and int(r["ndiv"]) == ndiv]
            ref = all(bool(r["reference"]) for r in subset)
            agg = {
                "system": system,
                "ndiv": ndiv,
                "h": 2.0 ** (-ndiv),
                "reference": ref,
            }
            for metric in (
                "ke_fluid_rel_error",
                "lfluid_rel_error",
                "lambda_rel_error",
                "pcon_rel_error",
                "hcon_rel_error",
            ):
                vals = [float(r[metric]) for r in subset if math.isfinite(float(r[metric]))]
                agg[f"{metric}_rms_over_states"] = math.nan if not vals else float(np.sqrt(np.mean(np.square(vals))))
                agg[f"{metric}_max_over_states"] = math.nan if not vals else float(np.max(vals))
            agg_rows.append(agg)

    write_csv(STUDY / "frozen_grid_summary.csv", rows)
    write_csv(STUDY / "frozen_grid_aggregate.csv", agg_rows)
    return rows, agg_rows


def fit_order(rows: list[dict[str, object]], metric: str) -> float:
    xs = []
    ys = []
    for row in rows:
        if row["reference"]:
            continue
        y = float(row[metric])
        if math.isfinite(y) and y > 0.0:
            xs.append(float(row["h"]))
            ys.append(y)
    if len(xs) < 2:
        return math.nan
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot(agg_rows: list[dict[str, object]]) -> None:
    metrics = [
        ("ke_fluid_rel_error_rms_over_states", "fluid KE"),
        ("lfluid_rel_error_rms_over_states", "linear impulse"),
        ("lambda_rel_error_rms_over_states", "angular impulse"),
        ("pcon_rel_error_rms_over_states", "conserved P"),
        ("hcon_rel_error_rms_over_states", "conserved H"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, system in zip(axes, ("single", "multi")):
        subset = [r for r in agg_rows if r["system"] == system and not r["reference"]]
        subset.sort(key=lambda r: float(r["h"]))
        for metric, label in metrics:
            h = np.asarray([float(r["h"]) for r in subset])
            y = np.asarray([float(r[metric]) for r in subset])
            order = fit_order([r for r in agg_rows if r["system"] == system], metric)
            ax.loglog(h, y, "o-", label=f"{label} p={order:.2f}")
        ax.invert_xaxis()
        ax.set_title(f"{system.capitalize()} frozen-state BEM grid convergence")
        ax.set_xlabel("h ~ 2^-ndiv")
        ax.set_ylabel("RMS relative error over sampled states")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = STUDY / "frozen_grid_dashboard.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out.relative_to(ROOT)}")


def print_summary(agg_rows: list[dict[str, object]]) -> None:
    metrics = [
        "ke_fluid_rel_error_rms_over_states",
        "lfluid_rel_error_rms_over_states",
        "lambda_rel_error_rms_over_states",
        "pcon_rel_error_rms_over_states",
        "hcon_rel_error_rms_over_states",
    ]
    print()
    print("Frozen-state BEM grid orders")
    print("system    KE       L        Lambda   P        H")
    for system in ("single", "multi"):
        rows = [r for r in agg_rows if r["system"] == system]
        vals = [fit_order(rows, metric) for metric in metrics]
        print(f"{system:<8} " + " ".join(f"{v:>8.3f}" for v in vals))
    print()
    print("RMS relative errors over sampled states")
    print("system    ndiv   h          KE          L          Lambda     P          H")
    for row in agg_rows:
        if row["reference"]:
            continue
        print(
            f"{row['system']:<8} {int(row['ndiv']):>4} {float(row['h']):>9.5g} "
            f"{float(row['ke_fluid_rel_error_rms_over_states']):>10.3e} "
            f"{float(row['lfluid_rel_error_rms_over_states']):>10.3e} "
            f"{float(row['lambda_rel_error_rms_over_states']):>10.3e} "
            f"{float(row['pcon_rel_error_rms_over_states']):>10.3e} "
            f"{float(row['hcon_rel_error_rms_over_states']):>10.3e}"
        )


def run_cases(args: argparse.Namespace) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    exe = Path(args.exe) if args.exe else ROOT / "target" / "release" / "multi_ellip.exe"
    if not exe.exists():
        alt = ROOT / "target" / "release" / "multi_ellip"
        exe = alt if alt.exists() else exe
    if not exe.exists():
        raise FileNotFoundError(f"solver executable not found: {exe}")

    selected = build_cases()
    if args.system:
        selected = [case for case in selected if case.system in args.system]
    started = time.monotonic()
    for idx, case in enumerate(selected, start=1):
        write_input(case)
        print()
        print("=" * 76)
        print(
            f"Case {idx}/{len(selected)}: {case.name} "
            f"(system={case.system}, t_sample={case.sample_time:g}, ndiv={case.ndiv})"
        )
        print(f"Elapsed study time: {fmt_hms(time.monotonic() - started)}")
        print("=" * 76)
        if output_complete(case) and not args.rerun:
            print("Output already complete; skipping. Use --rerun to regenerate.", flush=True)
            continue
        stream_run([str(exe), str(case.run_dir / "input.txt"), str(case.run_dir)], case.run_dir / "run.log")
    print(f"Run phase elapsed: {fmt_hms(time.monotonic() - started)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen-state BEM grid convergence proof.")
    parser.add_argument("--rerun", action="store_true", help="rerun cases even if output exists")
    parser.add_argument("--plot-only", action="store_true", help="only summarize/plot existing outputs")
    parser.add_argument("--exe", default="", help="solver executable; defaults to target/release/multi_ellip.exe")
    parser.add_argument("--system", action="append", choices=["single", "multi"])
    args = parser.parse_args()

    STUDY.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        run_cases(args)
    _rows, agg_rows = summarize()
    plot(agg_rows)
    print(f"Saved {(STUDY / 'frozen_grid_summary.csv').relative_to(ROOT)}")
    print(f"Saved {(STUDY / 'frozen_grid_aggregate.csv').relative_to(ROOT)}")
    print_summary(agg_rows)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}: {' '.join(exc.cmd)}", file=sys.stderr)
        raise
