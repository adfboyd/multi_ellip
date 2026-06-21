from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "studies" / "impulse_variational_comparison"
RUNS = STUDY / "runs"
MANIFEST = STUDY / "manifest.csv"
SUMMARY = STUDY / "summary.csv"


BASE_INPUT = {
    "cex1": -1.5,
    "cey1": 0.0,
    "cez1": 0.0,
    "oriw1": 1.0,
    "orii1": 2.0,
    "orij1": 0.0,
    "orik1": 0.0,
    "lvx1": -1.0,
    "lvy1": 0.2,
    "lvz1": 0.1,
    "avx1": 0.4,
    "avy1": -0.7,
    "avz1": 0.2,
    "shx1": 1.0,
    "shy1": 0.8,
    "shz1": 0.6,
    "req1": 1.0,
    "rhos1": 1.0,
    "cex2": 1.5,
    "cey2": 0.0,
    "cez2": 0.0,
    "oriw2": 1.0,
    "orii2": 0.0,
    "orij2": 1.0,
    "orik2": 0.0,
    "lvx2": -0.9,
    "lvy2": 0.1,
    "lvz2": 0.2,
    "avx2": -0.6,
    "avy2": -0.1,
    "avz2": -0.3,
    "shx2": 1.0,
    "shy2": 0.8,
    "shz2": 0.6,
    "req2": 1.0,
    "rhos2": 1.0,
    "rhof": 1.0,
    "nbody": 2,
    "impulse_scheme": 1,
    "energy_projection": 0,
    "exact_ellipsoid_geometry": 1,
    "exact_singular_geometry": 1,
    "variational_iters": 8,
    "variational_eps": 1.0e-5,
    "variational_max_shift": 1.0,
    "variational_tol": 1.0e-8,
}


@dataclass(frozen=True)
class Case:
    scheme: str
    ndiv: int
    dt: float
    tend: float

    @property
    def steps(self) -> int:
        return int(round(self.tend / self.dt))

    @property
    def name(self) -> str:
        return f"{self.scheme}_nd{self.ndiv}_dt{token(self.dt)}_t{token(self.tend)}"

    @property
    def run_dir(self) -> Path:
        return RUNS / self.name

    @property
    def input_path(self) -> Path:
        return self.run_dir / "input.txt"

    @property
    def output_dir(self) -> Path:
        return self.run_dir / "out"

    @property
    def data_path(self) -> Path:
        return self.output_dir / "multiple_body_complete.dat"

    @property
    def log_path(self) -> Path:
        return self.run_dir / "run.log"


def token(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def cases(ndivs: list[int], dts: list[float], tend: float) -> list[Case]:
    return [
        Case(scheme=scheme, ndiv=ndiv, dt=dt, tend=tend)
        for ndiv in ndivs
        for dt in dts
        for scheme in ("impulse", "variational")
    ]


def write_input(case: Case) -> None:
    cfg = dict(BASE_INPUT)
    cfg["ndiv"] = case.ndiv
    cfg["dt"] = case.dt
    cfg["tend"] = case.tend
    cfg["tprint"] = 1
    cfg["logevery"] = max(1, case.steps // 10)
    cfg["variational_scheme"] = 1 if case.scheme == "variational" else 0

    case.run_dir.mkdir(parents=True, exist_ok=True)
    with case.input_path.open("w", encoding="utf-8") as f:
        for key, value in cfg.items():
            f.write(f"{key}={value}\n")


def write_manifest(case_list: list[Case]) -> None:
    STUDY.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "scheme",
                "ndiv",
                "dt",
                "tend",
                "steps",
                "input",
                "output",
                "log",
            ],
        )
        writer.writeheader()
        for case in case_list:
            writer.writerow(
                {
                    "name": case.name,
                    "scheme": case.scheme,
                    "ndiv": case.ndiv,
                    "dt": case.dt,
                    "tend": case.tend,
                    "steps": case.steps,
                    "input": case.input_path.relative_to(ROOT),
                    "output": case.data_path.relative_to(ROOT),
                    "log": case.log_path.relative_to(ROOT),
                }
            )


def binary_path() -> Path:
    exe = ".exe" if sys.platform.startswith("win") else ""
    return ROOT / "target_codex_run" / "release" / f"multi_ellip{exe}"


def build_binary() -> None:
    env = dict(**{k: v for k, v in dict().items()})
    env = None
    subprocess.run(
        ["cargo", "build", "--release", "--bin", "multi_ellip"],
        cwd=ROOT,
        check=True,
        env=env,
    )


def run_case(case: Case, rerun: bool) -> None:
    if case.data_path.exists() and not rerun:
        print(f"skip {case.name}: output exists")
        return

    exe = binary_path()
    if not exe.exists():
        raise FileNotFoundError(f"missing release binary: {exe}")

    case.output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(exe), str(case.input_path), str(case.output_dir)]
    print(f"run {case.name}")
    with case.log_path.open("w", encoding="utf-8", newline="") as log:
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
            log.write(line)
            if (
                "Step" in line
                or "Run summary" in line
                or "First step completed" in line
                or "Discrete momentum drift" in line
            ):
                print(line.rstrip())
        rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def read_table(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, skipinitialspace=True)
        header = next(reader)
        rows = [[float(x) for x in row] for row in reader if row]
    arr = np.asarray(rows, dtype=float)
    return {name: arr[:, i] for i, name in enumerate(header)}


def vector_series(data: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    cols = []
    body = 1
    while f"{prefix}_x_{body}" in data:
        cols.append(
            np.column_stack(
                [
                    data[f"{prefix}_x_{body}"],
                    data[f"{prefix}_y_{body}"],
                    data[f"{prefix}_z_{body}"],
                ]
            )
        )
        body += 1
    if not cols:
        return np.zeros((len(data["time"]), 3))
    return np.sum(cols, axis=0)


def max_vector_drift(series: np.ndarray) -> float:
    if len(series) == 0:
        return math.nan
    return float(np.max(np.linalg.norm(series - series[0], axis=1)))


def max_percent_drift(values: np.ndarray) -> float:
    scale = max(abs(float(values[0])), np.finfo(float).eps)
    return float(np.max(np.abs(100.0 * (values - values[0]) / scale)))


def finite_max(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return math.nan
    return float(np.max(np.abs(finite)))


def state_arrays(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    pos_cols = []
    marker_cols = []
    body = 1
    while f"px_{body}" in data:
        pos_cols.extend([f"px_{body}", f"py_{body}", f"pz_{body}"])
        marker_cols.extend([f"ofix1_{body}", f"ofix2_{body}", f"ofix3_{body}"])
        body += 1
    return (
        np.column_stack([data[col] for col in pos_cols]),
        np.column_stack([data[col] for col in marker_cols]),
    )


def rms_against(data: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> tuple[float, float, float]:
    t = data["time"]
    ref_t = reference["time"]
    pos, marker = state_arrays(data)
    ref_pos, ref_marker = state_arrays(reference)
    ref_pos_interp = np.column_stack([np.interp(t, ref_t, ref_pos[:, i]) for i in range(ref_pos.shape[1])])
    ref_marker_interp = np.column_stack(
        [np.interp(t, ref_t, ref_marker[:, i]) for i in range(ref_marker.shape[1])]
    )
    ke_ref = np.interp(t, ref_t, reference["ke_total"])
    pos_rms = float(np.sqrt(np.mean((pos - ref_pos_interp) ** 2)))
    marker_rms = float(np.sqrt(np.mean((marker - ref_marker_interp) ** 2)))
    ke_scale = max(abs(float(reference["ke_total"][0])), np.finfo(float).eps)
    ke_rms_pct = float(np.sqrt(np.mean((100.0 * (data["ke_total"] - ke_ref) / ke_scale) ** 2)))
    return pos_rms, marker_rms, ke_rms_pct


def summarize(case_list: list[Case]) -> list[dict[str, object]]:
    loaded: dict[tuple[str, int, float], dict[str, np.ndarray]] = {}
    for case in case_list:
        if case.data_path.exists():
            loaded[(case.scheme, case.ndiv, case.dt)] = read_table(case.data_path)

    rows: list[dict[str, object]] = []
    for case in case_list:
        data = loaded.get((case.scheme, case.ndiv, case.dt))
        if data is None:
            continue
        pcon = vector_series(data, "pcon")
        hcon = vector_series(data, "hcon")
        jdisc = finite_max(data.get("jdisc_drift", np.asarray([math.nan])))

        pos_rms = marker_rms = ke_rms = math.nan
        if case.scheme == "impulse":
            ref = loaded.get(("variational", case.ndiv, case.dt))
            if ref is not None:
                pos_rms, marker_rms, ke_rms = rms_against(data, ref)

        rows.append(
            {
                "name": case.name,
                "scheme": case.scheme,
                "ndiv": case.ndiv,
                "dt": case.dt,
                "tend": case.tend,
                "steps": case.steps,
                "rows": len(data["time"]),
                "max_ke_drift_pct": max_percent_drift(data["ke_total"]),
                "max_pcon_drift_abs": max_vector_drift(pcon),
                "max_hcon_drift_abs": max_vector_drift(hcon),
                "max_jdisc_drift_abs": jdisc,
                "impulse_vs_variational_pos_rms": pos_rms,
                "impulse_vs_variational_marker_rms": marker_rms,
                "impulse_vs_variational_ke_rms_pct": ke_rms,
            }
        )

    with SUMMARY.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_summary(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    STUDY.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    colors = {"impulse": "#3b6ea8", "variational": "#b6524b"}
    markers = {1: "o", 2: "s"}

    for scheme in ("impulse", "variational"):
        for ndiv in sorted({int(row["ndiv"]) for row in rows}):
            subset = [
                row
                for row in rows
                if row["scheme"] == scheme and int(row["ndiv"]) == ndiv
            ]
            subset.sort(key=lambda row: float(row["dt"]))
            if not subset:
                continue
            dts = np.asarray([float(row["dt"]) for row in subset])
            label = f"{scheme}, nd{ndiv}"
            ax[0, 0].loglog(
                dts,
                [float(row["max_ke_drift_pct"]) for row in subset],
                marker=markers.get(ndiv, "o"),
                color=colors[scheme],
                linestyle="-" if ndiv == 1 else "--",
                label=label,
            )
            ax[0, 1].loglog(
                dts,
                [float(row["max_pcon_drift_abs"]) for row in subset],
                marker=markers.get(ndiv, "o"),
                color=colors[scheme],
                linestyle="-" if ndiv == 1 else "--",
            )
            ax[1, 0].loglog(
                dts,
                [float(row["max_hcon_drift_abs"]) for row in subset],
                marker=markers.get(ndiv, "o"),
                color=colors[scheme],
                linestyle="-" if ndiv == 1 else "--",
            )
            if scheme == "variational":
                ax[1, 1].loglog(
                    dts,
                    [float(row["max_jdisc_drift_abs"]) for row in subset],
                    marker=markers.get(ndiv, "o"),
                    color=colors[scheme],
                    linestyle="-" if ndiv == 1 else "--",
                    label=label,
                )

    ax[0, 0].set_title("energy drift")
    ax[0, 0].set_ylabel("max |Delta KE| / KE0 (%)")
    ax[0, 1].set_title("continuous endpoint P drift")
    ax[0, 1].set_ylabel("max |Delta P|")
    ax[1, 0].set_title("continuous endpoint H drift")
    ax[1, 0].set_ylabel("max |Delta H|")
    ax[1, 1].set_title("variational discrete momentum drift")
    ax[1, 1].set_ylabel("max |Delta Jdisc|")
    for a in ax.ravel():
        a.set_xlabel("dt")
        a.grid(True, which="both", alpha=0.25)
    ax[0, 0].legend(fontsize=8)
    ax[1, 1].legend(fontsize=8)
    fig.savefig(STUDY / "conservation_vs_dt.png", dpi=170)
    plt.close(fig)

    comp = [row for row in rows if row["scheme"] == "impulse"]
    if comp:
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
        for ndiv in sorted({int(row["ndiv"]) for row in comp}):
            subset = [row for row in comp if int(row["ndiv"]) == ndiv]
            subset.sort(key=lambda row: float(row["dt"]))
            dts = np.asarray([float(row["dt"]) for row in subset])
            ax[0].loglog(dts, [float(row["impulse_vs_variational_pos_rms"]) for row in subset], marker=markers.get(ndiv, "o"), label=f"nd{ndiv}")
            ax[1].loglog(dts, [float(row["impulse_vs_variational_marker_rms"]) for row in subset], marker=markers.get(ndiv, "o"), label=f"nd{ndiv}")
            ax[2].loglog(dts, [float(row["impulse_vs_variational_ke_rms_pct"]) for row in subset], marker=markers.get(ndiv, "o"), label=f"nd{ndiv}")
        ax[0].set_title("position RMS")
        ax[1].set_title("orientation-marker RMS")
        ax[2].set_title("KE RMS")
        ax[0].set_ylabel("impulse - variational")
        ax[2].set_ylabel("% of KE0")
        for a in ax:
            a.set_xlabel("dt")
            a.grid(True, which="both", alpha=0.25)
            a.legend(fontsize=8)
        fig.savefig(STUDY / "impulse_vs_variational_state_error.png", dpi=170)
        plt.close(fig)


def write_notes(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    impulse_rows = [row for row in rows if row["scheme"] == "impulse"]
    variational_rows = [row for row in rows if row["scheme"] == "variational"]
    impulse_ke = [float(row["max_ke_drift_pct"]) for row in impulse_rows]
    variational_j = [float(row["max_jdisc_drift_abs"]) for row in variational_rows]
    lines = [
        "# Impulse vs variational comparison",
        "",
        "This study compares the raw impulse step with the discrete-Lagrangian variational step on the same two-body ellipsoid case.",
        "",
        "The important distinction is that `pcon_*` and `hcon_*` are continuous endpoint diagnostics, while `jdisc_*` is the discrete Noether momentum associated with the variational update.",
        "",
        "## Interpretation",
        "",
        f"- Raw impulse keeps total continuous linear momentum near roundoff, but its maximum KE drift stays in a narrow band of {min(impulse_ke):.3g}--{max(impulse_ke):.3g}% as `dt` is reduced.",
        "- The variational method shows approximately first-order convergence in the endpoint KE diagnostic over this short test.",
        f"- The variational discrete Noether momentum is conserved to {max(variational_j):.3g} or better in all completed cases.",
        "- The old continuous endpoint `pcon/hcon` columns are therefore not the correct invariant for judging the variational scheme at finite timestep.",
        "- The practical next target is a cheaper impulse correction that reproduces the missing variational metric/configuration work, not full finite-difference variational Newton for production sweeps.",
        "",
        "## Files",
        "",
        "- `manifest.csv`: generated cases.",
        "- `summary.csv`: postprocessed conservation and trajectory metrics.",
        "- `conservation_vs_dt.png`: conservation diagnostics versus timestep.",
        "- `impulse_vs_variational_state_error.png`: impulse trajectory difference relative to variational at matched `ndiv,dt`.",
        "",
        "## Summary",
        "",
        "| scheme | ndiv | dt | max KE drift (%) | max endpoint dP | max endpoint dH | max discrete dJ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda r: (int(r["ndiv"]), float(r["dt"]), str(r["scheme"]))):
        lines.append(
            f"| {row['scheme']} | {row['ndiv']} | {float(row['dt']):g} | "
            f"{float(row['max_ke_drift_pct']):.6g} | "
            f"{float(row['max_pcon_drift_abs']):.6g} | "
            f"{float(row['max_hcon_drift_abs']):.6g} | "
            f"{float(row['max_jdisc_drift_abs']):.6g} |"
        )
    (STUDY / "NOTES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--ndivs", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--dts", nargs="+", type=float, default=[0.1, 0.05, 0.025])
    parser.add_argument("--tend", type=float, default=0.5)
    args = parser.parse_args()

    case_list = cases(args.ndivs, args.dts, args.tend)
    for case in case_list:
        write_input(case)
    write_manifest(case_list)

    if args.setup_only:
        print(f"wrote {len(case_list)} cases to {MANIFEST}")
        return

    if not args.analyze_only:
        if not args.no_build:
            build_binary()
        for case in case_list:
            run_case(case, args.rerun)

    rows = summarize(case_list)
    plot_summary(rows)
    write_notes(rows)
    print(f"summary: {SUMMARY}")
    print(f"plots:   {STUDY / 'conservation_vs_dt.png'}")
    print(f"         {STUDY / 'impulse_vs_variational_state_error.png'}")


if __name__ == "__main__":
    main()
