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
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "studies" / "ke_ratio_density"
DEFAULT_BINARY = ROOT / "target" / "release" / "multi_ellip.exe"


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def token(value: float | int) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def fmt_hms(seconds: float) -> str:
    seconds = int(round(max(0.0, seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def finish_time(seconds_from_now: float) -> str:
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
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        try:
            values[key.strip()] = float(value.strip())
        except ValueError:
            pass
    return values


def normalised_shape(values: dict[str, float]) -> np.ndarray:
    raw = np.array(
        [values.get("shx1", 1.0), values.get("shy1", 1.0), values.get("shz1", 1.0)],
        dtype=float,
    )
    req = values.get("req1", 1.0)
    return raw * (req / float(np.prod(raw) ** (1.0 / 3.0)))


def solid_mass(shape: np.ndarray, rho: float) -> float:
    return float((4.0 / 3.0) * np.pi * rho * np.prod(shape))


def solid_inertia(shape: np.ndarray, rho: float) -> np.ndarray:
    mass = solid_mass(shape, rho)
    a, b, c = shape
    return 0.2 * mass * np.array([b * b + c * c, a * a + c * c, a * a + b * b])


def lin_rot_ke_ratio(values: dict[str, float]) -> float:
    shape = normalised_shape(values)
    rho = values.get("rhos1", 1.0)
    mass = solid_mass(shape, rho)
    inertia = solid_inertia(shape, rho)
    velocity = np.array([values.get("lvx1", 0.0), values.get("lvy1", 0.0), values.get("lvz1", 0.0)])
    omega = np.array([values.get("avx1", 0.0), values.get("avy1", 0.0), values.get("avz1", 0.0)])
    lin_ke = 0.5 * mass * float(np.dot(velocity, velocity))
    rot_ke = 0.5 * float(np.dot(inertia * omega, omega))
    return lin_ke / rot_ke if rot_ke > 0.0 else float("nan")


def apply_shape_variant(values: dict[str, float], shape_ratio: tuple[float, float, float] | None) -> dict[str, float]:
    out = dict(values)
    if shape_ratio is None:
        return out

    target_ratio = lin_rot_ke_ratio(values)
    out["shx1"], out["shy1"], out["shz1"] = shape_ratio

    shape = normalised_shape(out)
    rho = out.get("rhos1", 1.0)
    mass = solid_mass(shape, rho)
    inertia = solid_inertia(shape, rho)
    velocity = np.array([out.get("lvx1", 0.0), out.get("lvy1", 0.0), out.get("lvz1", 0.0)])
    omega = np.array([out.get("avx1", 0.0), out.get("avy1", 0.0), out.get("avz1", 0.0)])
    lin_ke = 0.5 * mass * float(np.dot(velocity, velocity))
    rot_ke_unscaled = 0.5 * float(np.dot(inertia * omega, omega))
    if target_ratio > 0.0 and rot_ke_unscaled > 0.0:
        scale = np.sqrt((lin_ke / target_ratio) / rot_ke_unscaled)
        omega *= scale
        out["avx1"], out["avy1"], out["avz1"] = [float(x) for x in omega]
    return out


def write_input(
    path: Path,
    values: dict[str, float],
    ndiv: int,
    shape_ratio: tuple[float, float, float] | None,
) -> dict[str, float]:
    out = apply_shape_variant(values, shape_ratio)
    out["ndiv"] = float(ndiv)
    out["impulse_scheme"] = 1.0
    out["variational_scheme"] = 0.0
    out["hamiltonian_scheme"] = 0.0
    out["hamiltonian_midpoint_scheme"] = 0.0
    out.setdefault("tprint", 1.0)
    out.setdefault("logevery", 100.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for key, value in out.items():
            f.write(f"{key}={value:g}\n")
    return out


def source_cases(source_root: Path) -> list[Path]:
    cases = [path for path in source_root.iterdir() if (path / "input.txt").exists()]
    return sorted(cases, key=lambda p: p.name)


def complete_output(path: Path, params: dict[str, float]) -> bool:
    if not path.exists():
        return False
    tend = params.get("tend", float("nan"))
    dt = params.get("dt", float("nan"))
    if not np.isfinite(tend) or not np.isfinite(dt) or dt <= 0.0:
        return False
    expected_rows = int(round(tend / dt)) + 1
    with path.open("r", encoding="utf-8") as f:
        line_count = sum(1 for _ in f)
    return line_count >= expected_rows + 1


def load_output(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    return data


def vector(data: np.ndarray, *cols: str) -> np.ndarray:
    return np.column_stack([data[col] for col in cols])


def vector_drift_pct(data: np.ndarray, prefix: str) -> np.ndarray:
    values = vector(data, f"{prefix}_x_1", f"{prefix}_y_1", f"{prefix}_z_1")
    scale = max(float(np.linalg.norm(values[0])), 1.0)
    return 100.0 * np.linalg.norm(values - values[0], axis=1) / scale


def span(data: np.ndarray, *cols: str) -> float:
    values = vector(data, *cols)
    return float(np.max(np.linalg.norm(values - values[0], axis=1)))


def lin_rot_ratio(params: dict[str, float]) -> float:
    v2 = params.get("lvx1", 0.0) ** 2 + params.get("lvy1", 0.0) ** 2 + params.get("lvz1", 0.0) ** 2
    w2 = params.get("avx1", 0.0) ** 2 + params.get("avy1", 0.0) ** 2 + params.get("avz1", 0.0) ** 2
    return v2 / w2 if w2 > 0.0 else float("nan")


def fmt(value: float) -> str:
    return "" if not np.isfinite(value) else f"{float(value):.12g}"


def analyse(case: str, params: dict[str, float], output: Path) -> dict[str, str]:
    if not output.exists():
        return {"case": case, "status": "missing-output"}
    data = load_output(output)
    required = ["time", "ke_total", "ke_fluid", "ke_solid", "pcon_x_1", "hcon_x_1"]
    if len(data) < 2 or any(not np.all(np.isfinite(data[col])) for col in required):
        return {"case": case, "status": "nonfinite-or-short", "output": str(output)}

    ke0 = float(data["ke_total"][0])
    drift = 100.0 * (data["ke_total"] - ke0) / ke0
    return {
        "case": case,
        "status": "OK",
        "density": fmt(params.get("rhos1", float("nan"))),
        "target_linKE_rotKE": fmt(lin_rot_ke_ratio(params)),
        "input_lin_rot_speed_ratio": fmt(lin_rot_ratio(params)),
        "shape_x": fmt(params.get("shx1", float("nan"))),
        "shape_y": fmt(params.get("shy1", float("nan"))),
        "shape_z": fmt(params.get("shz1", float("nan"))),
        "ndiv": fmt(params.get("ndiv", float("nan"))),
        "dt": fmt(params.get("dt", float("nan"))),
        "tend": fmt(params.get("tend", float("nan"))),
        "ke0": fmt(ke0),
        "solid_ke0": fmt(float(data["ke_solid"][0])),
        "fluid_ke0": fmt(float(data["ke_fluid"][0])),
        "end_drift_pct": fmt(float(drift[-1])),
        "max_abs_drift_pct": fmt(float(np.max(np.abs(drift)))),
        "max_pcon_drift_pct": fmt(float(np.max(vector_drift_pct(data, "pcon")))),
        "max_hcon_drift_pct": fmt(float(np.max(vector_drift_pct(data, "hcon")))),
        "path_span": fmt(span(data, "px_1", "py_1", "pz_1")),
        "ofix_span": fmt(span(data, "ofix1_1", "ofix2_1", "ofix3_1")),
        "output": str(output),
    }


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_case(
    binary: Path,
    source_case: Path,
    run_root: Path,
    ndiv: int,
    rerun: bool,
    study_log: Path,
    index: int,
    total: int,
    shape_ratio: tuple[float, float, float] | None,
) -> dict[str, str]:
    case = source_case.name
    params = parse_input(source_case / "input.txt")
    run_dir = run_root / case
    input_path = run_dir / "input.txt"
    output = run_dir / "single_body_complete.dat"
    log_path = run_dir / "run.log"
    params = write_input(input_path, params, ndiv, shape_ratio)

    if not rerun and complete_output(output, params):
        row = analyse(case, params, output)
        return {**row, "returncode": "", "wall_seconds": "", "log": str(log_path), "message": "reused"}

    append_log(study_log, f"[{index:02d}/{total:02d}] START {case} ndiv={ndiv} at {now()}")
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        proc = subprocess.Popen(
            [str(binary), str(input_path), str(run_dir)],
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
    row = analyse(case, params, output)
    status = row.get("status", "unknown")
    if rc != 0 and status == "OK":
        status = "bad-returncode"
    row = {**row, "status": status, "returncode": str(rc), "wall_seconds": f"{wall:.3f}", "log": str(log_path)}
    append_log(study_log, f"[{index:02d}/{total:02d}] {status:<18} {case} wall={fmt_hms(wall)} rc={rc}")
    return row


def group_label(case: str) -> str:
    return "ratio1_rho4" if case.startswith("ratio1_") else "ratio20_rho0p25"


def run_number(case: str) -> int:
    tail = case.rsplit("_run", 1)[-1]
    try:
        return int(tail)
    except ValueError:
        return 0


def plot_dashboard(rows: list[dict[str, str]], run_root: Path, out: Path, ndiv: int, title_suffix: str) -> None:
    ok = [row for row in rows if row.get("status") == "OK"]
    if not ok:
        return
    fig, ax = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    colors = {"ratio1_rho4": "#1f77b4", "ratio20_rho0p25": "#d62728"}
    labels_done: set[tuple[str, str]] = set()

    for row in sorted(ok, key=lambda r: (group_label(r["case"]), run_number(r["case"]))):
        data = load_output(run_root / row["case"] / "single_body_complete.dat")
        t = data["time"]
        drift = 100.0 * (data["ke_total"] - data["ke_total"][0]) / data["ke_total"][0]
        group = group_label(row["case"])
        color = colors[group]
        group_short = group.replace("ratio", "ratio ")

        label = group_short if ("drift", group) not in labels_done else None
        labels_done.add(("drift", group))
        ax[0, 0].plot(t, drift, lw=1.0, alpha=0.55, color=color, label=label)

        label_solid = f"{group_short} solid" if ("solid", group) not in labels_done else None
        label_fluid = f"{group_short} fluid" if ("fluid", group) not in labels_done else None
        labels_done.add(("solid", group))
        labels_done.add(("fluid", group))
        ax[0, 1].plot(t, 100.0 * data["ke_solid"] / data["ke_total"][0], lw=0.9, alpha=0.45, color=color, label=label_solid)
        ax[0, 1].plot(
            t,
            100.0 * data["ke_fluid"] / data["ke_total"][0],
            lw=0.9,
            ls="--",
            alpha=0.45,
            color=color,
            label=label_fluid,
        )

        ax[1, 0].plot(data["px_1"], data["py_1"], lw=0.9, alpha=0.45, color=color)
        ax[1, 1].plot(data["ofix1_1"], data["ofix2_1"], lw=0.75, alpha=0.35, color=color)

    ax[0, 0].axhline(0.0, lw=0.8, color="black")
    ax[0, 0].set(title="Total KE drift", xlabel="t", ylabel="drift (%)")
    ax[0, 1].set(title="Energy exchange", xlabel="t", ylabel="% initial total KE")
    ax[1, 0].set(title="Centre path projection", xlabel="x", ylabel="y")
    ax[1, 1].set(title="Orientation marker projection", xlabel="ofix x", ylabel="ofix y")
    for axis in ax.flat:
        axis.grid(alpha=0.25)
    ax[0, 0].legend(fontsize=8)
    ax[0, 1].legend(fontsize=8)
    fig.suptitle(
        f"Random direction single-body impulse reruns, {title_suffix}, ndiv={ndiv}, dt=0.05, t=0..100",
        fontsize=14,
    )
    fig.savefig(out, dpi=170)
    plt.close(fig)


def plot_orientation_panels(rows: list[dict[str, str]], run_root: Path, out: Path, ndiv: int, title_suffix: str) -> None:
    ok = [row for row in rows if row.get("status") == "OK"]
    if not ok:
        return
    ordered = sorted(ok, key=lambda r: (group_label(r["case"]), run_number(r["case"])))
    fig, axes = plt.subplots(2, 5, figsize=(18, 8), sharex=True, sharey=True, constrained_layout=True)
    norm = plt.Normalize(0.0, 100.0)
    cmap = plt.get_cmap("viridis")

    for axis, row in zip(axes.flat, ordered):
        data = load_output(run_root / row["case"] / "single_body_complete.dat")
        t = data["time"]
        axis.scatter(data["ofix1_1"], data["ofix2_1"], c=t, cmap=cmap, norm=norm, s=3.0, linewidths=0.0)
        axis.plot(data["ofix1_1"], data["ofix2_1"], lw=0.35, alpha=0.28, color="#303030")
        axis.scatter(data["ofix1_1"][0], data["ofix2_1"][0], s=22, c="black", marker="o", zorder=3)
        axis.scatter(data["ofix1_1"][-1], data["ofix2_1"][-1], s=28, c="#d62728", marker="x", zorder=3)
        density = float(row.get("density", "nan") or "nan")
        ratio = float(row.get("target_linKE_rotKE", "nan") or "nan")
        axis.set_title(f"run {run_number(row['case'])}: rho={density:g}, ratio={ratio:g}", fontsize=10)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-1.1, 1.1)
        axis.set_ylim(-1.1, 1.1)
        axis.grid(alpha=0.2)

    for axis in axes[:, 0]:
        axis.set_ylabel("ofix y")
    for axis in axes[-1, :]:
        axis.set_xlabel("ofix x")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.78, pad=0.012)
    cbar.set_label("time")
    fig.suptitle(
        f"Orientation marker projections by run, impulse {title_suffix}, ndiv={ndiv}, dt=0.05, t=0..100",
        fontsize=14,
    )
    fig.savefig(out, dpi=170)
    plt.close(fig)


def postprocess(rows: list[dict[str, str]], run_root: Path, study: Path, ndiv: int, suffix: str, title_suffix: str) -> None:
    plot_dashboard(rows, run_root, study / f"ke_ratio_density_{suffix}_dashboard.png", ndiv, title_suffix)
    plot_orientation_panels(rows, run_root, study / f"orientation_marker_panels_{suffix}.png", ndiv, title_suffix)


def shape_label(shape_ratio: tuple[float, float, float] | None) -> str:
    if shape_ratio is None:
        return ""
    return "shape" + "_".join(token(v) for v in shape_ratio)


def title_label(label: str, shape_ratio: tuple[float, float, float] | None) -> str:
    if shape_ratio is not None:
        return "shape " + ":".join(f"{v:g}" for v in shape_ratio)
    if label:
        return label.replace("_", " ")
    return "shape 1:0.8:0.6"


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun the ten single-body KE-ratio cases with the impulse method.")
    parser.add_argument("--ndiv", type=int, required=True)
    parser.add_argument("--source-root", type=Path, default=STUDY / "runs")
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--shape", nargs=3, type=float, metavar=("SHX", "SHY", "SHZ"))
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--rerun", action="store_true", help="rerun completed outputs in the new run root")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    source_root = args.source_root if args.source_root.is_absolute() else ROOT / args.source_root
    shape_ratio = tuple(args.shape) if args.shape is not None else None
    variant_label = args.label or shape_label(shape_ratio)
    suffix = f"impulse_{variant_label}_nd{args.ndiv}" if variant_label else f"impulse_nd{args.ndiv}"
    title_suffix = title_label(variant_label, shape_ratio)
    run_root = args.run_root or (STUDY / f"runs_{suffix}")
    run_root = run_root if run_root.is_absolute() else ROOT / run_root
    binary = args.binary if args.binary.is_absolute() else ROOT / args.binary
    summary = STUDY / f"ke_ratio_density_{suffix}_summary.csv"
    study_log = STUDY / f"{suffix}_study.log"

    if not binary.exists():
        raise SystemExit(f"Missing binary: {binary}")

    cases = source_cases(source_root)
    if args.limit is not None:
        cases = cases[: args.limit]

    append_log(study_log, "=" * 72)
    append_log(study_log, f"Impulse KE-ratio rerun started at {now()}")
    append_log(study_log, f"ndiv: {args.ndiv}")
    append_log(study_log, f"Shape ratio override: {shape_ratio if shape_ratio is not None else 'none'}")
    append_log(study_log, f"Output suffix: {suffix}")
    append_log(study_log, f"Source root: {source_root}")
    append_log(study_log, f"Run root: {run_root}")
    append_log(study_log, f"Binary: {binary}")
    append_log(study_log, f"Cases: {len(cases)}")
    append_log(study_log, f"Rerun completed outputs: {args.rerun}")
    append_log(study_log, f"Plot only: {args.plot_only}")

    rows: list[dict[str, str]] = []
    started = time.monotonic()
    for i, case_dir in enumerate(cases, start=1):
        if args.plot_only:
            params = parse_input(case_dir / "input.txt")
            params = apply_shape_variant(params, shape_ratio)
            params["ndiv"] = float(args.ndiv)
            row = analyse(case_dir.name, params, run_root / case_dir.name / "single_body_complete.dat")
        else:
            row = run_case(
                binary,
                case_dir,
                run_root,
                args.ndiv,
                args.rerun,
                study_log,
                i,
                len(cases),
                shape_ratio,
            )
        rows.append(row)
        write_rows(summary, rows)
        elapsed = time.monotonic() - started
        avg = elapsed / i
        eta = avg * (len(cases) - i)
        append_log(
            study_log,
            f"Elapsed={fmt_hms(elapsed)} | avg/case={fmt_hms(avg)} | "
            f"remaining={len(cases) - i} | ETA={fmt_hms(eta)} | finish~{finish_time(eta)}",
        )

    postprocess(rows, run_root, STUDY, args.ndiv, suffix, title_suffix)
    append_log(study_log, f"Summary: {summary}")
    append_log(study_log, f"Dashboard: {STUDY / f'ke_ratio_density_{suffix}_dashboard.png'}")
    append_log(study_log, f"Orientation panels: {STUDY / f'orientation_marker_panels_{suffix}.png'}")
    append_log(study_log, f"Finished at {now()} wall={fmt_hms(time.monotonic() - started)}")


if __name__ == "__main__":
    main()
