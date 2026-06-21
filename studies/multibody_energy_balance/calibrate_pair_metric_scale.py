from __future__ import annotations

import argparse
import csv
import math
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "studies" / "multibody_energy_balance"
DEFAULT_RUNS = STUDY / "runs" / "pair_metric_scale_calibration"
DEFAULT_OUT = STUDY / "pair_metric_scale_calibration.csv"


def token(value: float | str) -> str:
    return str(value).replace("\\", "_").replace("/", "_").replace(".", "p").replace("-", "m")


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


def parse_scalar(raw: str) -> float | int | str:
    text = raw.strip()
    try:
        value = float(text)
    except ValueError:
        return text
    if value.is_integer():
        return int(value)
    return value


def read_input(path: Path) -> dict[str, float | int | str]:
    values: dict[str, float | int | str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = parse_scalar(value)
    return values


def write_input(
    path: Path, base: dict[str, float | int | str], tend: float, scale: float, angular_scale: float
) -> float:
    cfg = dict(base)
    dt = float(cfg.get("dt", 0.025))
    nsteps = max(1, int(round(tend / dt)))
    tend_snapped = nsteps * dt
    cfg.update(
        {
            "tend": tend_snapped,
            "tprint": 1,
            "logevery": max(1, nsteps // 4),
            "impulse_scheme": 1,
            "energy_projection": 0,
            "impulse_pair_metric_correction": 1,
            "impulse_pair_metric_mode": 1,
            "impulse_pair_metric_linear_scale": scale,
            "impulse_pair_metric_angular_scale": angular_scale,
            "impulse_pair_metric_cutoff": 0,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for key, value in cfg.items():
            f.write(f"{key}={value}\n")
    return tend_snapped


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, skipinitialspace=True))
    if not rows:
        raise RuntimeError(f"no rows in {path}")
    return rows


def f(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"{key} is not finite")
    return value


def body_ids(row: dict[str, str]) -> list[int]:
    ids = []
    for key in row:
        match = re.fullmatch(r"px_(\d+)", key)
        if match:
            ids.append(int(match.group(1)))
    return sorted(ids)


def vec(row: dict[str, str], prefix: str, body: int) -> tuple[float, float, float]:
    if prefix in {"p", "v", "w"}:
        names = {
            "p": (f"px_{body}", f"py_{body}", f"pz_{body}"),
            "v": (f"vx_{body}", f"vy_{body}", f"vz_{body}"),
            "w": (f"w1_{body}", f"w2_{body}", f"w3_{body}"),
        }[prefix]
    else:
        names = (f"{prefix}_x_{body}", f"{prefix}_y_{body}", f"{prefix}_z_{body}")
    return (f(row, names[0]), f(row, names[1]), f(row, names[2]))


def dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def rms(errors: list[float]) -> float:
    return math.sqrt(sum(e * e for e in errors) / max(1, len(errors)))


def separation(row: dict[str, str], ids: list[int]) -> float:
    if len(ids) < 2:
        return float("nan")
    seps = [dist(vec(row, "p", a), vec(row, "p", b)) for i, a in enumerate(ids) for b in ids[i + 1 :]]
    return min(seps)


def max_ke_drift(rows: list[dict[str, str]]) -> tuple[float, float]:
    ke0 = f(rows[0], "ke_total")
    drifts = [100.0 * (f(row, "ke_total") - ke0) / ke0 for row in rows]
    return max(abs(d) for d in drifts), drifts[-1]


def max_column(rows: list[dict[str, str]], key: str) -> float:
    if key not in rows[0]:
        return float("nan")
    values = []
    for row in rows:
        try:
            values.append(abs(float(row[key])))
        except ValueError:
            pass
    return max(values) if values else float("nan")


def max_body_h_span(rows: list[dict[str, str]], ids: list[int]) -> float:
    spans = []
    first = rows[0]
    for body in ids:
        h0 = vec(first, "hcon", body)
        scale = max(math.sqrt(sum(x * x for x in h0)), sys.float_info.epsilon)
        spans.extend(dist(vec(row, "hcon", body), h0) / scale for row in rows)
    return max(spans) if spans else float("nan")


def max_global_span(rows: list[dict[str, str]], ids: list[int], prefix: str) -> float:
    def total(row: dict[str, str]) -> tuple[float, float, float]:
        out = [0.0, 0.0, 0.0]
        for body in ids:
            v = vec(row, prefix, body)
            out[0] += v[0]
            out[1] += v[1]
            out[2] += v[2]
        return (out[0], out[1], out[2])

    start = total(rows[0])
    scale = max(math.sqrt(sum(x * x for x in start)), sys.float_info.epsilon)
    return max(dist(total(row), start) / scale for row in rows)


def summarize_log(log_path: Path) -> dict[str, str | float]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    patterns = {
        "mean_time_per_step": r"Mean time/step:\s+([0-9.]+) s",
        "cache_hits_direct": r"Impulse start cache hits/direct:\s+([0-9]+ / [0-9]+)",
        "pairs_last_max": r"Pair metric pairs last/max:\s+([0-9]+ / [0-9]+)",
    }
    out: dict[str, str | float] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if not match:
            out[key] = float("nan") if key == "mean_time_per_step" else ""
        elif key == "mean_time_per_step":
            out[key] = float(match.group(1))
        else:
            out[key] = match.group(1).strip()
    return out


def parse_reference(arg: str) -> tuple[str, Path]:
    if "=" not in arg:
        path = Path(arg)
        return path.parent.name, path
    label, raw_path = arg.split("=", 1)
    return label, Path(raw_path)


def default_references() -> list[tuple[str, Path]]:
    candidates = [
        ("t0p25", ROOT / "scratch_variational_reference" / "variational" / "multiple_body_complete.dat"),
        ("t0p5", ROOT / "scratch_variational_reference_t0p5" / "variational" / "multiple_body_complete.dat"),
    ]
    return [(label, path) for label, path in candidates if path.exists()]


def compare_run(
    label: str,
    scale: float,
    angular_scale: float,
    run_rows: list[dict[str, str]],
    ref_rows: list[dict[str, str]],
    log_path: Path,
    run_dir: Path,
) -> dict[str, object]:
    run_final = run_rows[-1]
    ref_final = ref_rows[-1]
    ids = body_ids(ref_final)
    pos_errors = [dist(vec(run_final, "p", body), vec(ref_final, "p", body)) for body in ids]
    vel_errors = [dist(vec(run_final, "v", body), vec(ref_final, "v", body)) for body in ids]
    omega_errors = [dist(vec(run_final, "w", body), vec(ref_final, "w", body)) for body in ids]
    max_abs_ke, final_ke = max_ke_drift(run_rows)
    sep = separation(run_final, ids)
    sep_ref = separation(ref_final, ids)
    out: dict[str, object] = {
        "label": label,
        "scale": scale,
        "angular_scale": angular_scale,
        "t_final": f(run_final, "time"),
        "pos_rms": rms(pos_errors),
        "vel_rms": rms(vel_errors),
        "omega_rms": rms(omega_errors),
        "sep": sep,
        "sep_ref": sep_ref,
        "sep_err": sep - sep_ref,
        "max_abs_ke_drift_pct": max_abs_ke,
        "final_ke_drift_pct": final_ke,
        "max_global_p_span_rel": max_global_span(run_rows, ids, "pcon"),
        "max_global_h_span_rel": max_global_span(run_rows, ids, "hcon"),
        "max_body_h_span_rel": max_body_h_span(run_rows, ids),
        "max_body_h_column": max_column(run_rows, "impulse_body_h_drift_max"),
        "run_dir": str(run_dir.relative_to(ROOT)),
    }
    out.update(summarize_log(log_path))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate the cheap impulse pair-metric correction against variational reference trajectories."
    )
    parser.add_argument(
        "--base-input",
        type=Path,
        default=ROOT / "scratch_variational_reference_t0p5" / "pair_dg" / "input.txt",
        help="input template for the pair-metric impulse run",
    )
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        help="variational reference as LABEL=PATH; may be repeated",
    )
    parser.add_argument("--scales", nargs="+", type=float, default=[1.0, 1.3, 1.4, 1.5])
    parser.add_argument("--angular-scale", type=float, default=0.0)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    base_input = args.base_input if args.base_input.is_absolute() else ROOT / args.base_input
    runs_dir = args.runs_dir if args.runs_dir.is_absolute() else ROOT / args.runs_dir
    out_path = args.out if args.out.is_absolute() else ROOT / args.out

    refs = [parse_reference(raw) for raw in args.reference] if args.reference else default_references()
    if not refs:
        raise SystemExit("no references supplied and no default scratch references found")
    for label, path in refs:
        resolved = path if path.is_absolute() else ROOT / path
        if not resolved.exists():
            raise SystemExit(f"reference {label} does not exist: {resolved}")

    if not args.skip_build:
        run_command(cargo_command() + ["build", "--release", "--bin", "multi_ellip"])
    exe = ROOT / "target" / "release" / ("multi_ellip.exe" if sys.platform == "win32" else "multi_ellip")
    if not exe.exists():
        raise SystemExit(f"release binary not found: {exe}")

    base = read_input(base_input)
    rows: list[dict[str, object]] = []
    for label, ref_path_raw in refs:
        ref_path = ref_path_raw if ref_path_raw.is_absolute() else ROOT / ref_path_raw
        ref_rows = read_rows(ref_path)
        tend = f(ref_rows[-1], "time")
        for scale in args.scales:
            run_dir = runs_dir / f"{token(label)}_lin{token(scale)}_ang{token(args.angular_scale)}"
            input_path = run_dir / "input.txt"
            log_path = run_dir / "run.log"
            data_path = run_dir / "multiple_body_complete.dat"
            tend_run = write_input(input_path, base, tend, scale, args.angular_scale)
            run_command([str(exe), str(input_path), str(run_dir)], log_path)
            run_rows = read_rows(data_path)
            if abs(f(run_rows[-1], "time") - tend_run) > 10.0 * sys.float_info.epsilon * max(1.0, tend_run):
                raise RuntimeError(
                    f"{run_dir} ended at t={f(run_rows[-1], 'time')}, expected snapped t={tend_run}"
                )
            rows.append(compare_run(label, scale, args.angular_scale, run_rows, ref_rows, log_path, run_dir))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {out_path.relative_to(ROOT)}")
    print(
        f"{'label':<8}{'scale':>8}{'pos rms':>12}{'vel rms':>12}"
        f"{'sep err':>12}{'max KE%':>10}{'mean step':>11}"
    )
    for row in rows:
        print(
            f"{str(row['label']):<8}{float(row['scale']):>8.3g}"
            f"{float(row['pos_rms']):>12.4g}{float(row['vel_rms']):>12.4g}"
            f"{float(row['sep_err']):>12.4g}{float(row['max_abs_ke_drift_pct']):>10.4g}"
            f"{float(row['mean_time_per_step']):>11.4g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
