"""Diagnose global vs per-body continuous impulse drift in multibody runs.

The impulse scheme writes per-body ``pcon_*`` and ``hcon_*`` columns.  For
multiple bodies only the sums are Noether invariants; this script quantifies
how strongly a run keeps each body's continuous impulse candidate fixed, and
compares that with the total kinetic-energy drift.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = ROOT / "studies" / "two_body_parameter_sweep_ndiv2_noproj" / "separation_convergence_runs"
DEFAULT_OUTPUT_DIR = ROOT / "scratch_impulse_partition"
DEFAULT_OUT = DEFAULT_OUTPUT_DIR / "impulse_partition_summary.csv"
DEFAULT_PLOT = DEFAULT_OUTPUT_DIR / "impulse_partition_summary.png"


def rel_or_abs_drift(values: np.ndarray, initial: np.ndarray) -> np.ndarray:
    diff = np.linalg.norm(values - initial[None, :], axis=1)
    scale = np.linalg.norm(initial)
    if scale > 1.0e-14:
        return diff / scale
    return diff


def body_count(columns: list[str]) -> int:
    bodies = []
    for col in columns:
        match = re.fullmatch(r"pcon_x_(\d+)", col)
        if match:
            bodies.append(int(match.group(1)))
    return max(bodies, default=0)


def load_numeric_table(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="") as f:
        reader = csv.reader(f, skipinitialspace=True)
        columns = next(reader)
        rows = [[float(value) for value in row] for row in reader if row]
    return columns, np.asarray(rows, dtype=float)


def column_map(columns: list[str]) -> dict[str, int]:
    return {name.strip(): idx for idx, name in enumerate(columns)}


def vector_series(data: np.ndarray, idx: dict[str, int], prefix: str, body: int) -> np.ndarray:
    return np.column_stack(
        [
            data[:, idx[f"{prefix}_x_{body}"]],
            data[:, idx[f"{prefix}_y_{body}"]],
            data[:, idx[f"{prefix}_z_{body}"]],
        ]
    )


def min_body_separation(data: np.ndarray, idx: dict[str, int], nbody: int) -> float:
    if nbody < 2:
        return math.nan
    positions = [
        np.column_stack(
            [
                data[:, idx[f"px_{body}"]],
                data[:, idx[f"py_{body}"]],
                data[:, idx[f"pz_{body}"]],
            ]
        )
        for body in range(1, nbody + 1)
    ]
    min_sep = math.inf
    for i in range(nbody):
        for j in range(i + 1, nbody):
            sep = np.linalg.norm(positions[i] - positions[j], axis=1)
            min_sep = min(min_sep, float(np.nanmin(sep)))
    return min_sep


def summarize_run(dat_path: Path, root: Path) -> dict[str, object]:
    columns, data = load_numeric_table(dat_path)
    idx = column_map(columns)
    nbody = body_count(columns)
    if nbody == 0:
        raise ValueError(f"could not detect pcon/hcon body columns in {dat_path}")

    ke = data[:, idx["ke_total"]]
    ke0 = ke[0]
    ke_drift = 100.0 * (ke - ke0) / ke0 if abs(ke0) > 1.0e-14 else np.full_like(ke, math.nan)

    p_body = [vector_series(data, idx, "pcon", body) for body in range(1, nbody + 1)]
    h_body = [vector_series(data, idx, "hcon", body) for body in range(1, nbody + 1)]
    p_global = np.sum(p_body, axis=0)
    h_global = np.sum(h_body, axis=0)

    p_body_drift = np.column_stack([rel_or_abs_drift(series, series[0]) for series in p_body])
    h_body_drift = np.column_stack([rel_or_abs_drift(series, series[0]) for series in h_body])
    p_global_drift = rel_or_abs_drift(p_global, p_global[0])
    h_global_drift = rel_or_abs_drift(h_global, h_global[0])

    return {
        "run": str(dat_path.parent.relative_to(root)),
        "nbody": nbody,
        "rows": data.shape[0],
        "t_end": float(data[-1, idx["time"]]),
        "min_separation": min_body_separation(data, idx, nbody),
        "max_abs_ke_drift_pct": float(np.nanmax(np.abs(ke_drift))),
        "final_ke_drift_pct": float(ke_drift[-1]),
        "max_global_p_drift": float(np.nanmax(p_global_drift)),
        "max_global_h_drift": float(np.nanmax(h_global_drift)),
        "max_body_p_drift": float(np.nanmax(p_body_drift)),
        "max_body_h_drift": float(np.nanmax(h_body_drift)),
        "dat": str(dat_path),
    }


def write_summary(rows: list[dict[str, object]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run",
        "nbody",
        "rows",
        "t_end",
        "min_separation",
        "max_abs_ke_drift_pct",
        "final_ke_drift_pct",
        "max_global_p_drift",
        "max_global_h_drift",
        "max_body_p_drift",
        "max_body_h_drift",
        "dat",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(rows: list[dict[str, object]], plot_path: Path) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda r: float(r["min_separation"]))
    sep = np.asarray([float(r["min_separation"]) for r in rows])
    ke = np.asarray([float(r["max_abs_ke_drift_pct"]) for r in rows])
    gp = np.asarray([float(r["max_global_p_drift"]) for r in rows])
    gh = np.asarray([float(r["max_global_h_drift"]) for r in rows])
    bp = np.asarray([float(r["max_body_p_drift"]) for r in rows])
    bh = np.asarray([float(r["max_body_h_drift"]) for r in rows])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    ax[0].plot(sep, ke, "o-", color="tab:red")
    ax[0].set_xlabel("minimum centre separation")
    ax[0].set_ylabel("max |KE drift| (%)")
    ax[0].set_title("Energy drift grows in close interaction")
    ax[0].grid(True, alpha=0.3)

    ax[1].semilogy(sep, gp, "o-", label="global P", color="tab:blue")
    ax[1].semilogy(sep, gh, "o-", label="global H", color="tab:orange")
    ax[1].semilogy(sep, bp, "s--", label="max body P", color="tab:green")
    ax[1].semilogy(sep, bh, "s--", label="max body H", color="tab:purple")
    ax[1].set_xlabel("minimum centre separation")
    ax[1].set_ylabel("relative drift")
    ax[1].set_title("Continuous impulse partition drift")
    ax[1].grid(True, which="both", alpha=0.3)
    ax[1].legend(fontsize=8)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    args = parser.parse_args()

    dat_paths = sorted(args.run_root.rglob("multiple_body_complete.dat"))
    rows = [summarize_run(path, args.run_root) for path in dat_paths]
    rows.sort(key=lambda r: (float(r["min_separation"]), str(r["run"])))
    write_summary(rows, args.out)
    plot_summary(rows, args.plot)
    print(f"wrote {len(rows)} rows to {args.out}")
    print(f"wrote plot to {args.plot}")


if __name__ == "__main__":
    main()
