from __future__ import annotations

import argparse
import csv
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "studies" / "two_body_parameter_sweep"
MANIFEST = STUDY / "two_body_parameter_sweep_manifest.csv"
STUDY_LOG = STUDY / "study.log"
SUMMARY = STUDY / "two_body_parameter_sweep_run_summary.csv"
BIN = ROOT / "target" / "release" / "multi_ellip.exe"

RECURRENCE_DIM = 3
RECURRENCE_TAU = 10
RECURRENCE_RATE = 0.03
RECURRENCE_THEILER = 50


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


def append_study_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(message + "\n")
    print(message, flush=True)


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def output_complete(row: dict[str, str]) -> bool:
    output = Path(row["output"])
    if not output.exists():
        return False
    try:
        tend = float(row["tend"])
        dt = float(row["dt"])
    except (KeyError, ValueError):
        return False
    expected_rows = int(round(tend / dt)) + 1
    with output.open("r", encoding="utf-8") as f:
        line_count = sum(1 for _ in f)
    return line_count >= expected_rows + 1


def load_output(row: dict[str, str]) -> np.ndarray:
    data = np.genfromtxt(Path(row["output"]), delimiter=",", names=True)
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    return data


def colvec(data: np.ndarray, prefix: str, body: int) -> np.ndarray:
    return np.column_stack(
        [data[f"{prefix}_x_{body}"], data[f"{prefix}_y_{body}"], data[f"{prefix}_z_{body}"]]
    )


def position(data: np.ndarray, body: int) -> np.ndarray:
    return np.column_stack([data[f"px_{body}"], data[f"py_{body}"], data[f"pz_{body}"]])


def marker(data: np.ndarray, body: int) -> np.ndarray:
    return np.column_stack([data[f"ofix1_{body}"], data[f"ofix2_{body}"], data[f"ofix3_{body}"]])


def vector_error_pct(values: np.ndarray) -> np.ndarray:
    initial = values[0]
    scale = max(float(np.linalg.norm(initial)), 1.0)
    return 100.0 * np.linalg.norm(values - initial, axis=1) / scale


def total_conserved_vector(data: np.ndarray, prefix: str) -> np.ndarray:
    cols = []
    body = 1
    while f"{prefix}_x_{body}" in data.dtype.names:
        cols.append(colvec(data, prefix, body))
        body += 1
    if not cols:
        raise KeyError(f"no columns found for {prefix}")
    return np.sum(cols, axis=0)


def plot_marker_path(ax, marker_values: np.ndarray, t: np.ndarray, title: str) -> None:
    start = len(t) // 2
    path = marker_values[start:]
    times = t[start:]
    stride = max(1, len(path) // 900)
    path = path[::stride]
    times = times[::stride]

    if len(path) > 1:
        segments = np.stack([path[:-1], path[1:]], axis=1)
        lc = Line3DCollection(segments, cmap="viridis", linewidth=0.28, alpha=0.55)
        lc.set_array(times[:-1])
        ax.add_collection3d(lc)
    ax.scatter(
        path[:, 0],
        path[:, 1],
        path[:, 2],
        c=times,
        cmap="viridis",
        s=1.6,
        alpha=0.7,
        depthshade=False,
        rasterized=True,
    )
    ax.set_xlim(-1.03, 1.03)
    ax.set_ylim(-1.03, 1.03)
    ax.set_zlim(-1.03, 1.03)
    ax.set_box_aspect((1, 1, 1))
    ax.set(title=f"{title} (second half)", xlabel="ofix1", ylabel="ofix2", zlabel="ofix3")


def has_finite_output(data: np.ndarray) -> bool:
    cols = ["time", "ke_total", "ke_fluid", "ke_solid"]
    return all(np.all(np.isfinite(data[col])) for col in cols)


def drift_pct(data: np.ndarray) -> np.ndarray:
    e0 = float(data["ke_total"][0])
    return 100.0 * (data["ke_total"] - e0) / e0


def save_dashboard(row: dict[str, str], data: np.ndarray) -> Path:
    out_dir = Path(row["output"]).parent
    out = out_dir / "dashboard.png"
    t = data["time"]
    p1 = position(data, 1)
    p2 = position(data, 2)
    sep = np.linalg.norm(p1 - p2, axis=1)
    m1 = marker(data, 1)
    m2 = marker(data, 2)
    p_err = vector_error_pct(total_conserved_vector(data, "pcon"))
    h_err = vector_error_pct(total_conserved_vector(data, "hcon"))

    fig = plt.figure(figsize=(15, 11), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, drift_pct(data), lw=1.2)
    ax.set(title="Total KE drift", xlabel="t", ylabel="drift (%)")
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, data["ke_total"], lw=1.2, label="total")
    ax.plot(t, data["ke_fluid"], lw=1.0, label="fluid")
    ax.plot(t, data["ke_solid"], lw=1.0, label="solid")
    ax.plot(t, data["ke_lin_solid"], lw=0.9, ls="--", label="solid lin")
    ax.plot(t, data["ke_rot_solid"], lw=0.9, ls="--", label="solid rot")
    ax.set(title="KE composition", xlabel="t", ylabel="KE")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(t, sep, lw=1.2)
    ax.set(title="Body separation", xlabel="t", ylabel="|x1 - x2|")
    ax.grid(alpha=0.25)

    ax3 = fig.add_subplot(gs[1, 0], projection="3d")
    ax3.plot(p1[:, 0], p1[:, 1], p1[:, 2], lw=1.1, label="body 1")
    ax3.plot(p2[:, 0], p2[:, 1], p2[:, 2], lw=1.1, label="body 2")
    ax3.scatter([p1[0, 0], p2[0, 0]], [p1[0, 1], p2[0, 1]], [p1[0, 2], p2[0, 2]], s=18)
    ax3.set(title="Body trajectories", xlabel="x", ylabel="y", zlabel="z")
    ax3.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    for i, label in enumerate(("x", "y", "z")):
        ax.plot(t, m1[:, i], lw=0.55, alpha=0.8, label=label, rasterized=True)
    ax.set(title="Orientation marker body 1", xlabel="t", ylabel="component")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[1, 2])
    for i, label in enumerate(("x", "y", "z")):
        ax.plot(t, m2[:, i], lw=0.55, alpha=0.8, label=label, rasterized=True)
    ax.set(title="Orientation marker body 2", xlabel="t", ylabel="component")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax3 = fig.add_subplot(gs[2, 0], projection="3d")
    plot_marker_path(ax3, m1, t, "Marker path body 1")

    ax3 = fig.add_subplot(gs[2, 1], projection="3d")
    plot_marker_path(ax3, m2, t, "Marker path body 2")

    ax = fig.add_subplot(gs[2, 2])
    ax.plot(t, p_err, lw=1.0, label="linear momentum")
    ax.plot(t, h_err, lw=1.0, label="angular momentum")
    ax.set(title="Momentum conservation error", xlabel="t", ylabel="relative drift (%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.suptitle(row["name"], fontsize=13)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def delay_embed(series: np.ndarray, dim: int, tau: int) -> np.ndarray:
    n = len(series) - (dim - 1) * tau
    if n <= 1:
        raise ValueError("not enough samples for recurrence embedding")
    return np.hstack([series[i * tau : i * tau + n] for i in range(dim)])


def recurrence_matrix(
    series: np.ndarray,
    dim: int = RECURRENCE_DIM,
    tau: int = RECURRENCE_TAU,
    target_rate: float = RECURRENCE_RATE,
    theiler: int = RECURRENCE_THEILER,
) -> tuple[np.ndarray, float, float]:
    embedded = delay_embed(series, dim, tau)
    diff = embedded[:, None, :] - embedded[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    mask = np.ones(dist.shape, dtype=bool)
    idx = np.arange(dist.shape[0])
    mask[np.abs(idx[:, None] - idx[None, :]) <= theiler] = False
    values = dist[mask]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("no finite recurrence distances")
    eps = float(np.quantile(finite, target_rate))
    rec = dist <= eps
    rec[~mask] = False
    achieved = float(rec[mask].mean()) if np.any(mask) else float("nan")
    return rec, eps, achieved


def save_recurrence(row: dict[str, str], data: np.ndarray, body: int) -> dict[str, str]:
    out_dir = Path(row["output"]).parent
    rec, eps, achieved = recurrence_matrix(marker(data, body))
    out = out_dir / f"recurrence_body{body}.png"
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    ax.imshow(rec, origin="lower", cmap="binary", interpolation="nearest")
    ax.set(
        title=(
            f"Recurrence body {body}: d={RECURRENCE_DIM}, tau={RECURRENCE_TAU}, "
            f"RR={achieved:.3f}"
        ),
        xlabel="sample",
        ylabel="sample",
    )
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return {
        "body": str(body),
        "recurrence_png": str(out),
        "embedding_dim": str(RECURRENCE_DIM),
        "tau": str(RECURRENCE_TAU),
        "target_rr": f"{RECURRENCE_RATE:.6g}",
        "theiler": str(RECURRENCE_THEILER),
        "epsilon": f"{eps:.12g}",
        "achieved_rr": f"{achieved:.12g}",
        "matrix_size": str(rec.shape[0]),
    }


def postprocess_case(row: dict[str, str]) -> dict[str, str]:
    output = Path(row["output"])
    if not output.exists():
        return {"dashboard": "", "postprocess_message": "missing output"}
    data = load_output(row)
    if len(data) < 2 or not has_finite_output(data):
        return {"dashboard": "", "postprocess_message": "non-finite or too-short output"}
    dashboard = save_dashboard(row, data)
    rec_rows = [save_recurrence(row, data, 1), save_recurrence(row, data, 2)]
    rec_csv = output.parent / "recurrence_metrics.csv"
    with rec_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rec_rows[0].keys()))
        writer.writeheader()
        writer.writerows(rec_rows)
    return {
        "dashboard": str(dashboard),
        "recurrence_metrics": str(rec_csv),
        "postprocess_message": "ok",
    }


def last_error(log_path: Path) -> str:
    if not log_path.exists():
        return ""
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in reversed(lines):
        if "Solver stopped with error:" in line:
            return line.strip()
        if "panicked at" in line:
            return line.strip()
    return ""


def stream_run(
    row: dict[str, str],
    index: int,
    total: int,
    study_log: Path,
) -> dict[str, str]:
    name = row["name"]
    input_path = Path(row["input"])
    output_dir = Path(row["output"]).parent
    log_path = output_dir / "run.log"
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    append_study_log(study_log, f"[{index:03d}/{total:03d}] START {name} at {now()}")
    cmd = [str(BIN), str(input_path), str(output_dir)]
    rc = 0
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

    wall = time.monotonic() - started
    err = last_error(log_path)
    complete = output_complete(row)
    status = "OK" if rc == 0 and complete and not err else "FAIL"
    if rc == 0 and err:
        status = "STOPPED"
    post = postprocess_case(row) if status == "OK" else {"dashboard": "", "postprocess_message": "skipped"}
    append_study_log(
        study_log,
        f"[{index:03d}/{total:03d}] {status:<7} {name} wall={fmt_hms(wall)} rc={rc}"
        + (f" msg={err}" if err else "")
    )
    return {
        **row,
        "status": status,
        "returncode": str(rc),
        "wall_seconds": f"{wall:.3f}",
        "log": str(log_path),
        "message": err,
        **post,
    }


def append_summary(path: Path, row: dict[str, str], write_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the two-body parameter sweep serially.")
    parser.add_argument("--rerun", action="store_true", help="rerun completed outputs")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N pending cases")
    parser.add_argument("--postprocess-only", action="store_true", help="generate dashboards/recurrences for completed outputs")
    parser.add_argument("--manifest", type=Path, default=MANIFEST, help="manifest CSV to run")
    parser.add_argument("--summary", type=Path, default=SUMMARY, help="summary CSV to append/write")
    parser.add_argument(
        "--study-log",
        type=Path,
        default=None,
        help="study log path; defaults to study.log beside the selected summary",
    )
    args = parser.parse_args()
    study_log = args.study_log or args.summary.parent / "study.log"

    if not BIN.exists():
        raise SystemExit(f"Missing release binary: {BIN}. Run cargo build --release first.")
    rows = load_manifest(args.manifest)
    if args.postprocess_only:
        pending = [row for row in rows if output_complete(row)]
    else:
        pending = [row for row in rows if args.rerun or not output_complete(row)]
    if args.limit is not None:
        pending = pending[: args.limit]

    append_study_log(study_log, "=" * 72)
    append_study_log(study_log, f"Study run started at {now()}")
    append_study_log(study_log, f"Manifest: {args.manifest}")
    append_study_log(study_log, f"Summary: {args.summary}")
    append_study_log(study_log, f"Total manifest cases: {len(rows)}")
    append_study_log(study_log, f"Pending this invocation: {len(pending)}")
    append_study_log(study_log, f"Rerun completed outputs: {args.rerun}")
    append_study_log(study_log, f"Postprocess only: {args.postprocess_only}")

    write_header = not args.summary.exists() or args.rerun
    if args.rerun and args.summary.exists():
        args.summary.unlink()

    started = time.monotonic()
    for i, row in enumerate(pending, start=1):
        if args.postprocess_only:
            append_study_log(
                study_log,
                f"[{i:03d}/{len(pending):03d}] POST {row['name']} at {now()}",
            )
            result = {**row, "status": "POST", "returncode": "", "wall_seconds": "", "log": "", "message": "", **postprocess_case(row)}
        else:
            result = stream_run(row, i, len(pending), study_log)
        append_summary(args.summary, result, write_header)
        write_header = False
        elapsed = time.monotonic() - started
        remaining_cases = len(pending) - i
        avg_case = elapsed / i
        eta_seconds = avg_case * remaining_cases
        append_study_log(
            study_log,
            "Elapsed study time: "
            f"{fmt_hms(elapsed)} | avg/case={fmt_hms(avg_case)} | "
            f"remaining={remaining_cases} | ETA={fmt_hms(eta_seconds)} | "
            f"finish~{fmt_finish_time(eta_seconds)}"
        )

    append_study_log(
        study_log,
        f"Study invocation finished at {now()} wall={fmt_hms(time.monotonic() - started)}",
    )


if __name__ == "__main__":
    main()
