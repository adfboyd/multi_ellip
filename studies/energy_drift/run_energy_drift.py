from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "studies" / "energy_drift"
RUNS = STUDY / "runs"


BASE_INPUT = {
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
    "tend": 50.0,
    "tprint": 1,
    "logevery": 50,
    "nbody": 1,
    "impulse_scheme": 1,
}

CASES = [
    {"name": "nd2_dt0p2", "ndiv": 2, "dt": 0.2, "group": "mesh"},
    {"name": "nd3_dt0p2", "ndiv": 3, "dt": 0.2, "group": "mesh"},
    {"name": "nd4_dt0p2", "ndiv": 4, "dt": 0.2, "group": "mesh"},
    {"name": "nd2_dt0p1", "ndiv": 2, "dt": 0.1, "group": "dt_check"},
    {"name": "nd2_dt0p05", "ndiv": 2, "dt": 0.05, "group": "dt_check"},
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


def write_input(case: dict[str, object]) -> Path:
    case_dir = RUNS / str(case["name"])
    case_dir.mkdir(parents=True, exist_ok=True)
    values = dict(BASE_INPUT)
    values["ndiv"] = case["ndiv"]
    values["dt"] = case["dt"]
    input_path = case_dir / "input.txt"
    with input_path.open("w", encoding="utf-8", newline="\n") as f:
        for key, value in values.items():
            f.write(f"{key}={value}\n")
    return input_path


def stream_run(cmd: list[str], log_path: Path | None = None) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    log = log_path.open("w", encoding="utf-8", newline="\n") if log_path else None
    started = time.monotonic()
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
    print(f"Finished in {fmt_hms(time.monotonic() - started)}", flush=True)


def output_complete(case: dict[str, object]) -> bool:
    output = RUNS / str(case["name"]) / "single_body_complete.dat"
    if not output.exists():
        return False
    # Header + initial row + one row for each sampled output step.
    expected_rows = int(float(BASE_INPUT["tend"]) / float(case["dt"])) + 2
    with output.open("r", encoding="utf-8") as f:
        actual_rows = sum(1 for _ in f)
    return actual_rows >= expected_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the t=0..50 single-body energy-drift study.")
    parser.add_argument("--rerun", action="store_true", help="rerun cases even if output already exists")
    args = parser.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)
    stream_run(["cargo", "build", "--release"])

    exe = ROOT / "target" / "release" / "multi_ellip.exe"
    if not exe.exists():
        exe = ROOT / "target" / "release" / "multi_ellip"

    study_started = time.monotonic()
    for idx, case in enumerate(CASES, start=1):
        input_path = write_input(case)
        output_dir = RUNS / str(case["name"])
        steps = int(float(BASE_INPUT["tend"]) / float(case["dt"]))
        print()
        print("=" * 72)
        print(
            f"Case {idx}/{len(CASES)}: {case['name']} "
            f"(ndiv={case['ndiv']}, dt={case['dt']}, ~{steps} steps)"
        )
        print(f"Elapsed study time: {fmt_hms(time.monotonic() - study_started)}")
        print("=" * 72)
        if output_complete(case) and not args.rerun:
            print("Output already complete; skipping. Use --rerun to regenerate.", flush=True)
            continue
        stream_run([str(exe), str(input_path), str(output_dir)], output_dir / "run.log")

    print()
    print(f"Study runs saved under {RUNS}")
    print(f"Total elapsed: {fmt_hms(time.monotonic() - study_started)}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}: {' '.join(exc.cmd)}", file=sys.stderr)
        raise
