from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "studies" / "multibody_energy_balance"
RUNS = STUDY / "runs"


BASE_INPUT = {
    "cex1": -1.5,
    "cey1": 0.0,
    "cez1": 0.0,
    "oriw1": 1.0,
    "orii1": 2.0,
    "orij1": 0.0,
    "orik1": 0.0,
    "lvx1": -7.907019248080818,
    "lvy1": 1.7102878772347652,
    "lvz1": 2.212574692365578,
    "avx1": 2.86061355541103,
    "avy1": -5.563329540982053,
    "avz1": 0.5870878208517969,
    "shx1": 1.0,
    "shy1": 0.7,
    "shz1": 0.7,
    "req1": 1.0,
    "rhos1": 1.0,
    "cex2": 1.5,
    "cey2": 0.0,
    "cez2": 0.0,
    "oriw2": 1.0,
    "orii2": 0.0,
    "orij2": 1.0,
    "orik2": 0.0,
    "lvx2": -6.856048818520789,
    "lvy2": 1.4829630246431886,
    "lvz2": 1.918487818170412,
    "avx2": -5.897074624758821,
    "avy2": -0.3860381663849962,
    "avz2": -2.1339875839557165,
    "shx2": 1.0,
    "shy2": 0.7,
    "shz2": 0.7,
    "req2": 1.0,
    "rhos2": 1.0,
    "rhof": 1.0,
    "ndiv": 2,
    "tend": 5.0,
    "dt": 0.025,
    "tprint": 4,
    "logevery": 100,
    "nbody": 2,
    "impulse_scheme": 1,
    "energy_projection": 0,
    "impulse_pair_metric_correction": 1,
    "impulse_pair_metric_mode": 1,
    "impulse_pair_metric_linear_scale": 1.0,
    "impulse_pair_metric_angular_scale": 0.0,
    "impulse_pair_metric_cutoff": 0,
    "impulse_pair_metric_inner_cutoff": 4.0,
    "impulse_pair_metric_outer_cutoff": 4.0,
    "impulse_pair_metric_eps": 0.001,
    "impulse_internal_load_constraint": 1,
}


def token(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def cargo_command() -> list[str]:
    cargo = os.environ.get("CARGO")
    if cargo:
        return [cargo]
    userprofile = os.environ.get("USERPROFILE")
    if userprofile:
        candidate = Path(userprofile) / ".cargo" / "bin" / "cargo.exe"
        if candidate.exists():
            return [str(candidate)]
    return ["cargo"]


def run_command(cmd: list[str], log_path: Path | None = None) -> None:
    print("$ " + " ".join(cmd), flush=True)
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


def write_input(path: Path, tend: float, dt: float, ndiv: int) -> None:
    cfg = dict(BASE_INPUT)
    cfg["tend"] = tend
    cfg["dt"] = dt
    cfg["ndiv"] = ndiv
    cfg["logevery"] = max(1, int(round(tend / dt)) // 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for key, value in cfg.items():
            f.write(f"{key}={value}\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize_data(data_path: Path) -> dict[str, float | int | str]:
    with data_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, skipinitialspace=True))
    if not rows:
        raise RuntimeError(f"no rows in {data_path}")

    ke0 = float(rows[0]["ke_total"])
    drifts = [100.0 * (float(r["ke_total"]) - ke0) / ke0 for r in rows]
    last = rows[-1]
    p1 = [float(last[f"px_1"]), float(last[f"py_1"]), float(last[f"pz_1"])]
    p2 = [float(last[f"px_2"]), float(last[f"py_2"]), float(last[f"pz_2"])]
    active_rows = sum(1 for r in rows if float(r["impulse_pair_metric_pairs"]) > 0.5)
    return {
        "rows": len(rows),
        "t_final": float(last["time"]),
        "max_abs_ke_drift_pct": max(abs(d) for d in drifts),
        "final_ke_drift_pct": drifts[-1],
        "final_separation": math.dist(p1, p2),
        "active_output_rows": active_rows,
        "sha256": sha256(data_path),
    }


def summarize_log(log_path: Path) -> dict[str, str]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    patterns = {
        "mean_time_per_step": r"Mean time/step:\s+([0-9.]+) s",
        "cache_hits_direct": r"Impulse start cache hits/direct:\s+([0-9]+ / [0-9]+)",
        "fp_iters": r"Impulse FP iters last/mean/max:\s+([^\n]+)",
        "pairs_last_max": r"Pair metric pairs last/max:\s+([0-9]+ / [0-9]+)",
    }
    out: dict[str, str] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        out[key] = match.group(1).strip() if match else ""
    return out


def parse_pair_count(value: str) -> tuple[int, int]:
    left, right = value.split("/")
    return int(left.strip()), int(right.strip())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the close two-body pair-metric benchmark and summarize byte-level regression data."
    )
    parser.add_argument("--name", default=None, help="run directory name under studies/multibody_energy_balance/runs")
    parser.add_argument("--tend", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=0.025)
    parser.add_argument("--ndiv", type=int, default=2)
    parser.add_argument("--baseline", type=Path, default=None, help="optional baseline multiple_body_complete.dat")
    parser.add_argument("--expect-cache-hits", type=int, default=None)
    parser.add_argument("--expect-direct-start-solves", type=int, default=None)
    parser.add_argument("--expect-active-output-rows", type=int, default=None)
    parser.add_argument("--expect-pairs-max", type=int, default=None)
    parser.add_argument("--max-mean-time", type=float, default=None, help="fail if reported mean step time exceeds this")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    name = args.name or f"pair_metric_regression_nd{args.ndiv}_dt{token(args.dt)}_t{token(args.tend)}"
    run_dir = RUNS / name
    input_path = run_dir / "input.txt"
    log_path = run_dir / "run.log"
    data_path = run_dir / "multiple_body_complete.dat"

    write_input(input_path, args.tend, args.dt, args.ndiv)
    if not args.skip_build:
        run_command(cargo_command() + ["build", "--release", "--bin", "multi_ellip"])
    exe = ROOT / "target" / "release" / ("multi_ellip.exe" if sys.platform == "win32" else "multi_ellip")
    run_command([str(exe), str(input_path), str(run_dir)], log_path)

    summary = summarize_data(data_path)
    summary.update(summarize_log(log_path))
    print("\nSummary")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    if args.baseline:
        baseline_hash = sha256(args.baseline)
        print(f"  baseline_sha256: {baseline_hash}")
        if summary["sha256"] != baseline_hash:
            raise SystemExit("output hash differs from baseline")
        print("  baseline_match: yes")

    if args.expect_cache_hits is not None or args.expect_direct_start_solves is not None:
        if not summary["cache_hits_direct"]:
            raise SystemExit("missing cache hit/direct summary")
        cache_hits, direct_solves = parse_pair_count(str(summary["cache_hits_direct"]))
        if args.expect_cache_hits is not None and cache_hits != args.expect_cache_hits:
            raise SystemExit(f"cache hits {cache_hits} != expected {args.expect_cache_hits}")
        if args.expect_direct_start_solves is not None and direct_solves != args.expect_direct_start_solves:
            raise SystemExit(
                f"direct start solves {direct_solves} != expected {args.expect_direct_start_solves}"
            )

    if args.expect_active_output_rows is not None:
        active_rows = int(summary["active_output_rows"])
        if active_rows != args.expect_active_output_rows:
            raise SystemExit(f"active output rows {active_rows} != expected {args.expect_active_output_rows}")

    if args.expect_pairs_max is not None:
        if not summary["pairs_last_max"]:
            raise SystemExit("missing pair last/max summary")
        _pairs_last, pairs_max = parse_pair_count(str(summary["pairs_last_max"]))
        if pairs_max != args.expect_pairs_max:
            raise SystemExit(f"max active pairs {pairs_max} != expected {args.expect_pairs_max}")

    if args.max_mean_time is not None:
        mean_time = float(summary["mean_time_per_step"])
        if mean_time > args.max_mean_time:
            raise SystemExit(f"mean step time {mean_time:.4f}s exceeds limit {args.max_mean_time:.4f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
