from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "studies" / "ke_ratio_density"
DEFAULT_SOURCE_ROOT = STUDY / "runs"
DEFAULT_OUT_ROOT = ROOT / "single_body_dynamics_check_after_two_body"
DEFAULT_REFERENCE = STUDY / "ke_ratio_density_summary.csv"
BIN = ROOT / "target" / "release" / "multi_ellip.exe"


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def fmt_hms(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def fmt_finish_time(seconds_from_now: float) -> str:
    return (datetime.now() + timedelta(seconds=max(0.0, seconds_from_now))).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(message + "\n")
    print(message, flush=True)


def parse_input(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        try:
            values[key.strip()] = float(value.strip())
        except ValueError:
            pass
    return values


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt_metric(value: float) -> str:
    if not np.isfinite(value):
        return ""
    return f"{float(value):.12g}"


def load_output(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    return data


def vector_error_pct(data: np.ndarray, prefix: str) -> np.ndarray:
    values = np.column_stack([data[f"{prefix}_x_1"], data[f"{prefix}_y_1"], data[f"{prefix}_z_1"]])
    initial = values[0]
    scale = max(float(np.linalg.norm(initial)), 1.0)
    return 100.0 * np.linalg.norm(values - initial, axis=1) / scale


def span(data: np.ndarray, cols: tuple[str, str, str]) -> float:
    values = np.column_stack([data[col] for col in cols])
    return float(np.max(np.linalg.norm(values - values[0], axis=1)))


def complete_output(path: Path, params: dict[str, float]) -> bool:
    if not path.exists():
        return False
    tend = params.get("tend")
    dt = params.get("dt")
    if tend is None or dt is None or dt <= 0:
        return False
    expected_rows = int(round(tend / dt)) + 1
    with path.open("r", encoding="utf-8") as f:
        lines = sum(1 for _ in f)
    return lines >= expected_rows + 1


def lin_rot_input_ratio(params: dict[str, float]) -> float:
    v2 = params.get("lvx1", 0.0) ** 2 + params.get("lvy1", 0.0) ** 2 + params.get("lvz1", 0.0) ** 2
    w2 = params.get("avx1", 0.0) ** 2 + params.get("avy1", 0.0) ** 2 + params.get("avz1", 0.0) ** 2
    return v2 / w2 if w2 > 0 else float("nan")


def analyse_output(case: str, output: Path, params: dict[str, float]) -> dict[str, str]:
    if not output.exists():
        return {"case": case, "status": "missing-output"}
    data = load_output(output)
    finite_cols = ["time", "ke_total", "ke_fluid", "ke_solid", "pcon_x_1", "hcon_x_1"]
    if len(data) < 2 or any(not np.all(np.isfinite(data[col])) for col in finite_cols):
        return {"case": case, "status": "nonfinite-or-short"}

    ke0 = float(data["ke_total"][0])
    drift = 100.0 * (data["ke_total"] - ke0) / ke0
    p_err = vector_error_pct(data, "pcon")
    h_err = vector_error_pct(data, "hcon")
    return {
        "case": case,
        "status": "OK",
        "density": fmt_metric(params.get("rhos1", float("nan"))),
        "input_lin_rot_speed_ratio": fmt_metric(lin_rot_input_ratio(params)),
        "ndiv": fmt_metric(params.get("ndiv", float("nan"))),
        "dt": fmt_metric(params.get("dt", float("nan"))),
        "tend": fmt_metric(params.get("tend", float("nan"))),
        "ke0": fmt_metric(ke0),
        "ke_end": fmt_metric(float(data["ke_total"][-1])),
        "final_drift_pct": fmt_metric(float(drift[-1])),
        "max_abs_drift_pct": fmt_metric(float(np.max(np.abs(drift)))),
        "max_pcon_drift_pct": fmt_metric(float(np.max(p_err))),
        "max_hcon_drift_pct": fmt_metric(float(np.max(h_err))),
        "path_span": fmt_metric(span(data, ("px_1", "py_1", "pz_1"))),
        "ofix_span": fmt_metric(span(data, ("ofix1_1", "ofix2_1", "ofix3_1"))),
        "output": str(output),
    }


def with_reference(row: dict[str, str], reference: dict[str, dict[str, str]]) -> dict[str, str]:
    ref = reference.get(row["case"])
    if not ref:
        return row
    merged = dict(row)
    for key in ("final_drift_pct", "max_abs_drift_pct", "path_span", "ofix_span"):
        ref_value = ref.get(key, "")
        merged[f"reference_{key}"] = ref_value
        try:
            merged[f"delta_{key}"] = fmt_metric(float(row[key]) - float(ref_value))
        except (KeyError, ValueError):
            merged[f"delta_{key}"] = ""
    return merged


def run_case(case_dir: Path, out_root: Path, rerun: bool, study_log: Path, index: int, total: int) -> dict[str, str]:
    case = case_dir.name
    src_input = case_dir / "input.txt"
    params = parse_input(src_input)
    out_dir = out_root / "runs" / case
    output = out_dir / "single_body_complete.dat"
    log_path = out_dir / "run.log"
    input_path = out_dir / "input.txt"
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_input, input_path)

    if not rerun and complete_output(output, params):
        row = analyse_output(case, output, params)
        return {**row, "returncode": "", "wall_seconds": "", "log": str(log_path), "message": "reused"}

    append_log(study_log, f"[{index:02d}/{total:02d}] START {case} at {now()}")
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        proc = subprocess.Popen(
            [str(BIN), str(input_path), str(out_dir)],
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
    wall = time.monotonic() - started
    row = analyse_output(case, output, params)
    status = row.get("status", "")
    if rc != 0 and status == "OK":
        status = "bad-returncode"
    row = {**row, "status": status, "returncode": str(rc), "wall_seconds": f"{wall:.3f}", "log": str(log_path)}
    append_log(study_log, f"[{index:02d}/{total:02d}] {status:<18} {case} wall={fmt_hms(wall)} rc={rc}")
    return row


def plot_dashboard(rows: list[dict[str, str]], out: Path) -> None:
    ok = [row for row in rows if row.get("status") == "OK"]
    if not ok:
        return
    names = [row["case"] for row in ok]
    short = [name.replace("ratio", "r").replace("_run", "_") for name in names]

    def values(key: str) -> np.ndarray:
        return np.array([float(row.get(key, "nan") or "nan") for row in ok], dtype=float)

    x = np.arange(len(ok))
    fig, ax = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)

    ax[0, 0].bar(x, values("max_abs_drift_pct"), label="current", color="#4078a8")
    ref = values("reference_max_abs_drift_pct")
    if np.any(np.isfinite(ref)):
        ax[0, 0].plot(x, ref, "o", ms=4, color="#b23b3b", label="old summary")
    ax[0, 0].set(title="Max absolute KE drift", ylabel="%")
    ax[0, 0].legend(fontsize=8)

    ax[0, 1].bar(x - 0.18, values("max_pcon_drift_pct"), width=0.36, label="linear", color="#56945f")
    ax[0, 1].bar(x + 0.18, values("max_hcon_drift_pct"), width=0.36, label="angular", color="#a66a3f")
    ax[0, 1].set(title="Momentum conservation drift", ylabel="%")
    ax[0, 1].legend(fontsize=8)

    ax[1, 0].bar(x, values("delta_max_abs_drift_pct"), color="#6f5aa8")
    ax[1, 0].axhline(0.0, lw=0.8, color="black")
    ax[1, 0].set(title="Change in max KE drift vs old summary", ylabel="percentage points")

    ax[1, 1].bar(x - 0.18, values("path_span"), width=0.36, label="trajectory", color="#4078a8")
    ax[1, 1].bar(x + 0.18, values("ofix_span"), width=0.36, label="orientation marker", color="#d19b3d")
    ax[1, 1].set(title="Motion spans", ylabel="span")
    ax[1, 1].legend(fontsize=8)

    for axis in ax.flat:
        axis.set_xticks(x)
        axis.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.25)

    fig.suptitle("Single-body dynamics validation rerun", fontsize=14)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun the ten single-body KE-ratio dynamics cases.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--reference-summary", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    source_root = args.source_root if args.source_root.is_absolute() else ROOT / args.source_root
    out_root = args.out_root if args.out_root.is_absolute() else ROOT / args.out_root
    reference_path = args.reference_summary if args.reference_summary.is_absolute() else ROOT / args.reference_summary
    study_log = out_root / "study.log"
    summary = out_root / "single_body_dynamics_check_summary.csv"

    if not BIN.exists():
        raise SystemExit(f"Missing release binary: {BIN}. Run cargo build --release first.")

    cases = sorted(path for path in source_root.iterdir() if (path / "input.txt").exists())
    if args.limit is not None:
        cases = cases[: args.limit]
    reference = {row["case"]: row for row in read_rows(reference_path)}

    append_log(study_log, "=" * 72)
    append_log(study_log, f"Single-body dynamics check started at {now()}")
    append_log(study_log, f"Source root: {source_root}")
    append_log(study_log, f"Output root: {out_root}")
    append_log(study_log, f"Reference summary: {reference_path}")
    append_log(study_log, f"Cases: {len(cases)}")
    append_log(study_log, f"Rerun completed outputs: {args.rerun}")

    rows: list[dict[str, str]] = []
    started = time.monotonic()
    for i, case_dir in enumerate(cases, start=1):
        row = run_case(case_dir, out_root, args.rerun, study_log, i, len(cases))
        rows.append(with_reference(row, reference))
        write_rows(summary, rows)
        elapsed = time.monotonic() - started
        avg_case = elapsed / i
        eta = avg_case * (len(cases) - i)
        append_log(
            study_log,
            f"Elapsed={fmt_hms(elapsed)} | avg/case={fmt_hms(avg_case)} | "
            f"remaining={len(cases) - i} | ETA={fmt_hms(eta)} | finish~{fmt_finish_time(eta)}",
        )

    dashboard = out_root / "single_body_dynamics_check_dashboard.png"
    plot_dashboard(rows, dashboard)
    append_log(study_log, f"Summary: {summary}")
    append_log(study_log, f"Dashboard: {dashboard}")
    append_log(study_log, f"Single-body dynamics check finished at {now()} wall={fmt_hms(time.monotonic() - started)}")


if __name__ == "__main__":
    main()
