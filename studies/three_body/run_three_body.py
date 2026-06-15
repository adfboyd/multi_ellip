from __future__ import annotations

import argparse
import csv
import itertools
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "studies" / "three_body"
RUNS = STUDY / "runs"

# 2 x 2 x 2 factorial: density x dt x ndiv.
DENSITIES = [1.0, 0.1]
DTS = [0.1, 0.05]
NDIVS = [2, 3]
T_END = 25.0
NBODY = 3

# Three bodies on a ~5-radius triangle in the z=0 plane with tangential
# (orbiting) velocities plus small out-of-plane components and distinct spins.
# Chosen so the bodies interact strongly but stay clear of contact over t=25.
BODY_GEOMETRY = [
    # cx, cy, cz, ori(w,i,j,k), lv(x,y,z), av(x,y,z)
    (5.0, 0.0, 0.0, (1.0, 2.0, 0.0, 0.0), (0.0, 0.4, 0.1), (1.0, 1.0, 0.0)),
    (-2.5, 4.33, 0.0, (1.0, 0.0, 1.0, 0.0), (-0.35, -0.2, 0.1), (0.0, 1.0, 1.0)),
    (-2.5, -4.33, 0.0, (1.0, 0.0, 0.0, 1.0), (0.35, -0.2, -0.2), (1.0, 0.0, 1.0)),
]
SHAPE = (1.0, 0.8, 0.6)  # shx, shy, shz
REQ = 1.0


def base_input(density: float, dt: float, ndiv: int) -> dict[str, object]:
    values: dict[str, object] = {}
    for idx, (cx, cy, cz, ori, lv, av) in enumerate(BODY_GEOMETRY, start=1):
        values[f"cex{idx}"] = cx
        values[f"cey{idx}"] = cy
        values[f"cez{idx}"] = cz
        values[f"oriw{idx}"], values[f"orii{idx}"], values[f"orij{idx}"], values[f"orik{idx}"] = ori
        values[f"lvx{idx}"], values[f"lvy{idx}"], values[f"lvz{idx}"] = lv
        values[f"avx{idx}"], values[f"avy{idx}"], values[f"avz{idx}"] = av
        values[f"shx{idx}"], values[f"shy{idx}"], values[f"shz{idx}"] = SHAPE
        values[f"req{idx}"] = REQ
        values[f"rhos{idx}"] = density
    values.update(
        {
            "rhof": 1.0,
            "ndiv": ndiv,
            "tend": T_END,
            "dt": dt,
            "tprint": 1,
            "logevery": 50,
            "nbody": NBODY,
            "impulse_scheme": 1,
        }
    )
    return values


def label(density: float, dt: float, ndiv: int) -> str:
    d = str(density).replace(".", "p")
    s = str(dt).replace(".", "p")
    return f"rho{d}_nd{ndiv}_dt{s}"


def cases() -> list[dict[str, object]]:
    out = []
    for density, dt, ndiv in itertools.product(DENSITIES, DTS, NDIVS):
        out.append(
            {
                "density": density,
                "dt": dt,
                "ndiv": ndiv,
                "name": label(density, dt, ndiv),
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


def output_file(case: dict[str, object]) -> Path:
    return RUNS / str(case["name"]) / "multiple_body_complete.dat"


def write_input(case: dict[str, object]) -> Path:
    run_dir = RUNS / str(case["name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "input.txt"
    values = base_input(float(case["density"]), float(case["dt"]), int(case["ndiv"]))
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for key, value in values.items():
            f.write(f"{key}={value}\n")
    return path


def output_complete(case: dict[str, object]) -> bool:
    path = output_file(case)
    if not path.exists():
        return False
    expected_rows = int(T_END / float(case["dt"])) + 1
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f) >= expected_rows


def stream_run(cmd: list[str], log_path: Path | None = None) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    log = log_path.open("w", encoding="utf-8", newline="\n") if log_path else None
    try:
        proc = subprocess.Popen(
            cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
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
    return np.column_stack([data[f"px_{body}"], data[f"py_{body}"], data[f"pz_{body}"]])


def velocity(data: np.ndarray, body: int) -> np.ndarray:
    return np.column_stack([data[f"vx_{body}"], data[f"vy_{body}"], data[f"vz_{body}"]])


def drift_pct(data: np.ndarray) -> np.ndarray:
    ke = data["ke_total"]
    return 100.0 * (ke - ke[0]) / ke[0]


def span(values: np.ndarray) -> float:
    return float(np.max(np.linalg.norm(values - values[0], axis=1)))


def min_separation(data: np.ndarray) -> tuple[float, float]:
    seps = []
    for a, b in itertools.combinations(range(1, NBODY + 1), 2):
        seps.append(np.linalg.norm(position(data, a) - position(data, b), axis=1))
    sep = np.min(np.stack(seps, axis=1), axis=1)
    idx = int(np.argmin(sep))
    return float(sep[idx]), float(data["time"][idx])


def totals(data: np.ndarray, prefix: str) -> np.ndarray:
    out = vec(data, prefix, 1)
    for b in range(2, NBODY + 1):
        out = out + vec(data, prefix, b)
    return out


def transport_check(data: np.ndarray) -> float:
    """Integrate sum_b v_b x pcon_b (the buggy torque's H-drift rate) and compare
    against actual H drift. Near 0 means the transport term is doing its job."""
    t = data["time"]
    cross = np.zeros((len(t), 3))
    for b in range(1, NBODY + 1):
        cross = cross + np.cross(velocity(data, b), vec(data, "pcon", b))
    H = totals(data, "hcon")
    dt = np.diff(t)
    inc = 0.5 * (cross[1:] + cross[:-1]) * dt[:, None]
    omit_drift = np.zeros_like(H)
    omit_drift[1:] = np.cumsum(inc, axis=0)
    return float(np.max(np.linalg.norm(omit_drift, axis=1)))


def summarize_case(case: dict[str, object], data: np.ndarray) -> dict[str, object]:
    drift = drift_pct(data)
    min_sep, min_sep_time = min_separation(data)
    return {
        "case": case["name"],
        "density": case["density"],
        "ndiv": case["ndiv"],
        "dt": case["dt"],
        "tend": T_END,
        "rows": len(data),
        "ke0": data["ke_total"][0],
        "ke_end": data["ke_total"][-1],
        "drift_end_pct": drift[-1],
        "drift_max_abs_pct": float(np.max(np.abs(drift))),
        "p_total_span": span(totals(data, "pcon")),
        "h_total_span": span(totals(data, "hcon")),
        "h_omit_term_span": transport_check(data),
        "min_separation": min_sep,
        "min_separation_time": min_sep_time,
    }


def write_summary(rows: list[dict[str, object]]) -> None:
    out = STUDY / "three_body_summary.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {out.relative_to(ROOT)}")
    print(f"{'case':<22}{'maxdrift%':>11}{'enddrift%':>11}{'P span':>11}{'H span':>11}{'minsep':>9}")
    for row in rows:
        print(
            f"{row['case']:<22}{row['drift_max_abs_pct']:>11.4f}{row['drift_end_pct']:>11.4f}"
            f"{row['p_total_span']:>11.2e}{row['h_total_span']:>11.2e}{row['min_separation']:>9.3f}"
        )


def plot(
    results: dict[str, np.ndarray],
    rows: list[dict[str, object]],
    valid: list[dict[str, object]],
) -> None:
    row_by_name = {str(r["case"]): r for r in rows}
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    styles = {2: "-", 3: "--"}
    colors = {1.0: "tab:blue", 0.1: "tab:red"}

    # (0,0)+(0,1): KE drift vs time, one panel per dt.
    for col, dt in enumerate(sorted(set(DTS))):
        axis = ax[0, col]
        for case in valid:
            if float(case["dt"]) != dt:
                continue
            data = results[str(case["name"])]
            axis.plot(
                data["time"], drift_pct(data),
                color=colors[float(case["density"])], ls=styles[int(case["ndiv"])], lw=1.3,
                label=f"rho={case['density']:g}, nd{case['ndiv']}",
            )
        axis.axhline(0, color="k", lw=0.7)
        axis.set(title=f"Total KE drift (dt={dt:g})", xlabel="t", ylabel="drift (%)")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)

    # (0,2): max drift bars.
    axis = ax[0, 2]
    names = [str(c["name"]) for c in valid]
    vals = [float(row_by_name[n]["drift_max_abs_pct"]) for n in names]
    axis.barh(range(len(names)), vals, color="tab:purple")
    axis.set_yticks(range(len(names)))
    axis.set_yticklabels(names, fontsize=7)
    axis.invert_yaxis()
    axis.set(title="max |KE drift| (%)", xlabel="percent")
    axis.grid(alpha=0.3, axis="x")

    # (1,0): H span vs case (the conservation check).
    axis = ax[1, 0]
    h_span = [float(row_by_name[n]["h_total_span"]) for n in names]
    omit = [float(row_by_name[n]["h_omit_term_span"]) for n in names]
    y = range(len(names))
    axis.barh(y, omit, color="tab:gray", alpha=0.6, label="if transport term omitted")
    axis.barh(y, h_span, color="tab:green", label="actual H drift")
    axis.set_yticks(list(y))
    axis.set_yticklabels(names, fontsize=7)
    axis.invert_yaxis()
    axis.set(title="Total angular momentum drift", xlabel="|H - H0| max")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3, axis="x")

    # (1,1): pairwise separations for a representative case (prefer the lowest
    # density / finest valid case to show the strongest coupling).
    axis = ax[1, 1]
    rep = sorted(valid, key=lambda c: (float(c["density"]), -int(c["ndiv"]), float(c["dt"])))[0]
    rep = str(rep["name"])
    data = results[rep]
    for a, b in itertools.combinations(range(1, NBODY + 1), 2):
        sep = np.linalg.norm(position(data, a) - position(data, b), axis=1)
        axis.plot(data["time"], sep, lw=1.2, label=f"|{a}-{b}|")
    axis.set(title=f"Pairwise centre separation ({rep})", xlabel="t", ylabel="distance")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)

    # (1,2): energy exchange (fluid vs solid) for representative case.
    axis = ax[1, 2]
    ke0 = data["ke_total"][0]
    axis.plot(data["time"], 100 * data["ke_fluid"] / ke0, lw=1.2, label="fluid")
    axis.plot(data["time"], 100 * data["ke_solid"] / ke0, lw=1.2, ls="--", label="solid")
    axis.set(title=f"Energy exchange ({rep})", xlabel="t", ylabel="% initial total KE")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)

    fig.suptitle(
        f"Three-body impulse scheme: density x dt x ndiv factorial, t=0..{T_END:g}", fontsize=14
    )
    fig.tight_layout()
    out = STUDY / "three_body_dashboard.png"
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
        print("\n" + "=" * 72)
        print(f"Case {idx}/{len(all_cases)}: {case['name']}")
        print(f"Elapsed study time: {fmt_hms(time.monotonic() - started)}")
        print("=" * 72)
        if output_complete(case) and not args.rerun:
            print("Output already complete; skipping. Use --rerun to regenerate.", flush=True)
            continue
        case_start = time.monotonic()
        try:
            stream_run([str(exe), str(input_path), str(out_dir)], out_dir / "run.log")
            print(f"Case finished in {fmt_hms(time.monotonic() - case_start)}", flush=True)
        except subprocess.CalledProcessError as exc:
            # A case past the stability limit can crash/diverge; record and carry on
            # to the remaining cases rather than aborting the whole sweep.
            print(f"Case FAILED (exit {exc.returncode}); continuing.", flush=True)
    print(f"\nRun phase elapsed: {fmt_hms(time.monotonic() - started)}")


def is_valid_output(case: dict[str, object]) -> bool:
    """Output exists, is complete, and did not diverge (finite, drift not absurd).
    The rho=0.1 / ndiv=3 corner is past the explicit-PCDM angular stability limit
    and blows up; those cases are excluded from the summary/plot."""
    path = output_file(case)
    if not path.exists() or not output_complete(case):
        return False
    try:
        data = load(case)
    except ValueError:
        return False
    ke = data["ke_total"]
    if not np.all(np.isfinite(ke)):
        return False
    return float(np.max(np.abs(ke - ke[0])) / ke[0]) < 1.0  # <100% drift => not diverged


def summarize_and_plot() -> None:
    all_cases = cases()
    valid = [c for c in all_cases if is_valid_output(c)]
    skipped = [str(c["name"]) for c in all_cases if c not in valid]
    if skipped:
        print(f"Excluding {len(skipped)} missing/diverged case(s): {', '.join(skipped)}")
    if not valid:
        print("No valid cases to summarize.")
        return
    results = {str(case["name"]): load(case) for case in valid}
    rows = [summarize_case(case, results[str(case["name"])]) for case in valid]
    write_summary(rows)
    plot(results, rows, valid)


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-body energy-conservation study.")
    parser.add_argument("--rerun", action="store_true", help="rerun cases even if output exists")
    parser.add_argument("--plot-only", action="store_true", help="only summarize/plot existing output")
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
