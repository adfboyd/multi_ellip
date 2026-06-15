from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "studies" / "density_sweep"
RUNS = STUDY / "runs"

DENSITIES = [4.0, 2.0, 1.0, 0.5, 0.25, 0.1]
T_END = 50.0
DT = 0.05
NDIV = 2


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
    "ndiv": NDIV,
    "tend": T_END,
    "dt": DT,
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
    "ndiv": NDIV,
    "tend": T_END,
    "dt": DT,
    "tprint": 1,
    "logevery": 100,
    "nbody": 2,
    "impulse_scheme": 1,
}


def density_label(rho: float) -> str:
    return f"rho{str(rho).replace('.', 'p')}"


def cases() -> list[dict[str, object]]:
    out = []
    for system in ("single", "multi"):
        for rho in DENSITIES:
            out.append(
                {
                    "system": system,
                    "density": rho,
                    "name": f"{system}_{density_label(rho)}",
                }
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


def input_values(case: dict[str, object]) -> dict[str, float]:
    values = dict(SINGLE_BASE if case["system"] == "single" else MULTI_BASE)
    rho = float(case["density"])
    values["rhos1"] = rho
    if case["system"] == "multi":
        values["rhos2"] = rho
    return values


def output_file(case: dict[str, object]) -> Path:
    filename = "single_body_complete.dat" if case["system"] == "single" else "multiple_body_complete.dat"
    return RUNS / str(case["name"]) / filename


def write_input(case: dict[str, object]) -> Path:
    run_dir = RUNS / str(case["name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "input.txt"
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for key, value in input_values(case).items():
            f.write(f"{key}={value}\n")
    return path


def output_complete(case: dict[str, object]) -> bool:
    path = output_file(case)
    if not path.exists():
        return False
    expected_rows = int(T_END / DT) + 2
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f) >= expected_rows


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


def load(case: dict[str, object]) -> np.ndarray:
    data = np.genfromtxt(output_file(case), delimiter=",", names=True)
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    return data


def vec(data: np.ndarray, prefix: str, body: int) -> np.ndarray:
    return np.column_stack(
        [data[f"{prefix}_x_{body}"], data[f"{prefix}_y_{body}"], data[f"{prefix}_z_{body}"]]
    )


def position(data: np.ndarray, body: int) -> np.ndarray:
    return np.column_stack(
        [data[f"px_{body}"], data[f"py_{body}"], data[f"pz_{body}"]]
    )


def drift_pct(data: np.ndarray) -> np.ndarray:
    ke = data["ke_total"]
    return 100.0 * (ke - ke[0]) / ke[0]


def span(values: np.ndarray) -> float:
    return float(np.max(np.linalg.norm(values - values[0], axis=1)))


def min_separation(data: np.ndarray) -> tuple[float, float]:
    if "px_2" not in data.dtype.names:
        return float("nan"), float("nan")
    rel = position(data, 1) - position(data, 2)
    sep = np.linalg.norm(rel, axis=1)
    idx = int(np.argmin(sep))
    return float(sep[idx]), float(data["time"][idx])


def summarize_case(case: dict[str, object], data: np.ndarray) -> dict[str, object]:
    drift = drift_pct(data)
    p_total = vec(data, "pcon", 1)
    h_total = vec(data, "hcon", 1)
    if case["system"] == "multi":
        p_total = p_total + vec(data, "pcon", 2)
        h_total = h_total + vec(data, "hcon", 2)
    min_sep, min_sep_time = min_separation(data)
    return {
        "case": case["name"],
        "system": case["system"],
        "density": case["density"],
        "ndiv": NDIV,
        "dt": DT,
        "tend": T_END,
        "rows": len(data),
        "ke0": data["ke_total"][0],
        "ke_end": data["ke_total"][-1],
        "drift_end_pct": drift[-1],
        "drift_max_abs_pct": np.max(np.abs(drift)),
        "p_total_span": span(p_total),
        "h_total_span": span(h_total),
        "min_separation": min_sep,
        "min_separation_time": min_sep_time,
    }


def write_summary(rows: list[dict[str, object]]) -> None:
    out = STUDY / "density_sweep_summary.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out.relative_to(ROOT)}")
    print("system  density  end drift %  max |drift| %    P span       H span")
    for row in rows:
        print(
            f"{row['system']:<6} {row['density']:>7} {row['drift_end_pct']:>12.5f}"
            f" {row['drift_max_abs_pct']:>14.5f}"
            f" {row['p_total_span']:>11.3e} {row['h_total_span']:>11.3e}"
        )


def plot(results: dict[str, np.ndarray], rows: list[dict[str, object]]) -> None:
    by_system = {"single": [], "multi": []}
    for row in rows:
        by_system[str(row["system"])].append(row)
    for system_rows in by_system.values():
        system_rows.sort(key=lambda row: float(row["density"]))

    fig, ax = plt.subplots(3, 3, figsize=(18, 14))
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(DENSITIES)))
    color_by_density = dict(zip(DENSITIES, colors))

    for system, axis in (("single", ax[0, 0]), ("multi", ax[0, 1])):
        for case in cases():
            if case["system"] != system:
                continue
            data = results[str(case["name"])]
            rho = float(case["density"])
            axis.plot(data["time"], drift_pct(data), lw=1.1, color=color_by_density[rho], label=f"rho={rho:g}")
        axis.axhline(0, color="k", lw=0.8)
        axis.set(title=f"{system} total KE drift", xlabel="t", ylabel="drift (%)")
        axis.legend(fontsize=7, ncol=2)
        axis.grid(alpha=0.3)

    for system, marker in (("single", "o"), ("multi", "s")):
        rows_s = by_system[system]
        rho = [float(row["density"]) for row in rows_s]
        max_drift = [float(row["drift_max_abs_pct"]) for row in rows_s]
        end_drift = [float(row["drift_end_pct"]) for row in rows_s]
        ax[0, 2].plot(rho, max_drift, marker=marker, lw=1.4, label=f"{system} max")
        ax[0, 2].plot(rho, np.abs(end_drift), marker=marker, lw=1.0, ls="--", label=f"{system} |end|")
    ax[0, 2].set_xscale("log")
    ax[0, 2].set(title="Drift summary vs density", xlabel="solid density", ylabel="percent")
    ax[0, 2].invert_xaxis()
    ax[0, 2].legend(fontsize=8)
    ax[0, 2].grid(alpha=0.3)

    for system, marker in (("single", "o"), ("multi", "s")):
        rows_s = by_system[system]
        rho = [float(row["density"]) for row in rows_s]
        p_span = [float(row["p_total_span"]) for row in rows_s]
        h_span = [float(row["h_total_span"]) for row in rows_s]
        ax[1, 0].plot(rho, p_span, marker=marker, lw=1.4, label=f"{system} P")
        ax[1, 1].plot(rho, h_span, marker=marker, lw=1.4, label=f"{system} H")
    for axis, title in ((ax[1, 0], "Total P invariant span"), (ax[1, 1], "Total H invariant span")):
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.invert_xaxis()
        axis.set(title=title, xlabel="solid density", ylabel="span")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)

    multi_rows = by_system["multi"]
    rho = [float(row["density"]) for row in multi_rows]
    min_sep = [float(row["min_separation"]) for row in multi_rows]
    ax[1, 2].plot(rho, min_sep, marker="s", lw=1.4)
    ax[1, 2].set_xscale("log")
    ax[1, 2].invert_xaxis()
    ax[1, 2].set(title="Multi-body minimum centre separation", xlabel="solid density", ylabel="distance")
    ax[1, 2].grid(alpha=0.3)

    for rho in (4.0, 1.0, 0.1):
        for system, axis in (("single", ax[2, 0]), ("multi", ax[2, 1])):
            case = next(c for c in cases() if c["system"] == system and float(c["density"]) == rho)
            data = results[str(case["name"])]
            ke0 = data["ke_total"][0]
            fluid_pct = 100.0 * data["ke_fluid"] / ke0
            solid_pct = 100.0 * data["ke_solid"] / ke0
            axis.plot(data["time"], fluid_pct, lw=1.0, color=color_by_density[rho], label=f"rho={rho:g} fluid")
            axis.plot(data["time"], solid_pct, lw=1.0, ls="--", color=color_by_density[rho], label=f"rho={rho:g} solid")
    ax[2, 0].set(title="Single energy exchange", xlabel="t", ylabel="% initial total KE")
    ax[2, 1].set(title="Multi energy exchange", xlabel="t", ylabel="% initial total KE")
    for axis in (ax[2, 0], ax[2, 1]):
        axis.legend(fontsize=6, ncol=2)
        axis.grid(alpha=0.3)

    for case in cases():
        if case["system"] != "multi":
            continue
        data = results[str(case["name"])]
        rel = position(data, 1) - position(data, 2)
        sep = np.linalg.norm(rel, axis=1)
        rho = float(case["density"])
        ax[2, 2].plot(data["time"], sep, lw=1.0, color=color_by_density[rho], label=f"rho={rho:g}")
    ax[2, 2].set(title="Multi-body centre separation", xlabel="t", ylabel="distance")
    ax[2, 2].legend(fontsize=7, ncol=2)
    ax[2, 2].grid(alpha=0.3)

    fig.suptitle(f"Density sweep: impulse scheme, ndiv={NDIV}, dt={DT}, t=0..{T_END}", fontsize=14)
    fig.tight_layout()
    out = STUDY / "density_sweep_dashboard.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved {out.relative_to(ROOT)}")


def run_cases(args: argparse.Namespace) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    stream_run(["cargo", "build", "--release"])
    exe = ROOT / "target" / "release" / "multi_ellip.exe"
    if not exe.exists():
        exe = ROOT / "target" / "release" / "multi_ellip"

    all_cases = cases()
    started = time.monotonic()
    for idx, case in enumerate(all_cases, start=1):
        input_path = write_input(case)
        out_dir = RUNS / str(case["name"])
        print()
        print("=" * 72)
        print(
            f"Case {idx}/{len(all_cases)}: {case['name']} "
            f"({case['system']}, rho={case['density']}, ndiv={NDIV}, dt={DT})"
        )
        print(f"Elapsed study time: {fmt_hms(time.monotonic() - started)}")
        print("=" * 72)
        if output_complete(case) and not args.rerun:
            print("Output already complete; skipping. Use --rerun to regenerate.", flush=True)
            continue
        stream_run([str(exe), str(input_path), str(out_dir)], out_dir / "run.log")
    print(f"Run phase elapsed: {fmt_hms(time.monotonic() - started)}")


def summarize_and_plot() -> None:
    all_cases = cases()
    results = {str(case["name"]): load(case) for case in all_cases}
    rows = [summarize_case(case, results[str(case["name"])]) for case in all_cases]
    write_summary(rows)
    plot(results, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Density sweep energy-conservation study.")
    parser.add_argument("--rerun", action="store_true", help="rerun cases even if output already exists")
    parser.add_argument("--plot-only", action="store_true", help="only summarize/plot existing outputs")
    args = parser.parse_args()

    STUDY.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        run_cases(args)
    summarize_and_plot()


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}: {' '.join(exc.cmd)}", file=sys.stderr)
        raise
