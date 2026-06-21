#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "studies" / "exact_geometry_validation"
OUT = ROOT / "scratch_exact_geometry_validation"
RUNS = OUT / "runs"

SHAPE_RATIO = np.array([1.0, 0.8, 0.6], dtype=float)
REQ = 1.0
RHO_F = 1.0
RHO_S = 1.0
DT = 0.1
T_END = 0.1


def actual_axes(shape_ratio: np.ndarray = SHAPE_RATIO, req: float = REQ) -> np.ndarray:
    return shape_ratio * (req / np.prod(shape_ratio) ** (1.0 / 3.0))


def ellipsoid_shape_factors(axes: np.ndarray, order: int = 1200) -> np.ndarray:
    x, w = np.polynomial.legendre.leggauss(order)
    u = 0.5 * (x + 1.0)
    weights = 0.5 * w
    s = u / (1.0 - u)
    ds_du = 1.0 / (1.0 - u) ** 2
    delta = np.sqrt((axes[0] ** 2 + s) * (axes[1] ** 2 + s) * (axes[2] ** 2 + s))
    pref = np.prod(axes)
    out = []
    for axis in axes:
        integrand = pref * ds_du / ((axis**2 + s) * delta)
        out.append(float(np.sum(weights * integrand)))
    return np.array(out)


def analytic_added_inertia(axes: np.ndarray, rho_f: float = RHO_F) -> np.ndarray:
    alpha, beta, gamma = ellipsoid_shape_factors(axes)
    volume = 4.0 * math.pi * np.prod(axes) / 3.0

    mf = volume * rho_f * np.diag(
        [
            alpha / (2.0 - alpha),
            beta / (2.0 - beta),
            gamma / (2.0 - gamma),
        ]
    )

    a, b, c = axes
    e1 = ((b**2 - c**2) ** 2 * (gamma - beta)) / (
        2.0 * (b**2 - c**2) + (beta - gamma) * (b**2 + c**2)
    )
    e2 = ((a**2 - c**2) ** 2 * (gamma - alpha)) / (
        2.0 * (a**2 - c**2) + (alpha - gamma) * (a**2 + c**2)
    )
    e3 = ((a**2 - b**2) ** 2 * (beta - alpha)) / (
        2.0 * (a**2 - b**2) + (alpha - beta) * (a**2 + b**2)
    )
    jf = 0.2 * volume * rho_f * np.diag([e1, e2, e3])

    out = np.zeros((6, 6), dtype=float)
    out[:3, :3] = mf
    out[3:, 3:] = jf
    return out


def input_text(ndiv: int, dof: int, exact: bool, exact_singular: bool = False) -> str:
    lin = [0.0, 0.0, 0.0]
    ang = [0.0, 0.0, 0.0]
    if dof < 3:
        lin[dof] = 1.0
    else:
        ang[dof - 3] = 1.0

    lines = {
        "cex1": 0.0,
        "cey1": 0.0,
        "cez1": 0.0,
        "oriw1": 1.0,
        "orii1": 0.0,
        "orij1": 0.0,
        "orik1": 0.0,
        "lvx1": lin[0],
        "lvy1": lin[1],
        "lvz1": lin[2],
        "avx1": ang[0],
        "avy1": ang[1],
        "avz1": ang[2],
        "shx1": SHAPE_RATIO[0],
        "shy1": SHAPE_RATIO[1],
        "shz1": SHAPE_RATIO[2],
        "req1": REQ,
        "rhos1": RHO_S,
        "rhof": RHO_F,
        "ndiv": ndiv,
        "tend": T_END,
        "dt": DT,
        "tprint": 1,
        "logevery": 1000000,
        "nbody": 1,
        "impulse_scheme": 1,
        "energy_projection": 0,
        "exact_ellipsoid_geometry": 1 if exact else 0,
        "exact_singular_geometry": 1 if exact_singular else 0,
    }
    return "\n".join(f"{k}={v}" for k, v in lines.items()) + "\n"


def resolve_binary(args: argparse.Namespace) -> Path:
    if args.binary:
        binary = Path(args.binary)
        if not binary.is_absolute():
            binary = ROOT / binary
        if not binary.exists():
            raise FileNotFoundError(binary)
        return binary

    target_dir = ROOT / args.target_dir
    exe = target_dir / "release" / ("multi_ellip.exe" if os.name == "nt" else "multi_ellip")
    if not exe.exists() or args.build:
        env = os.environ.copy()
        env["CARGO_TARGET_DIR"] = str(target_dir)
        subprocess.run(
            ["cargo", "build", "--release", "--bin", "multi_ellip"],
            cwd=ROOT,
            env=env,
            check=True,
        )
    return exe


def mode_name(exact: bool, exact_singular: bool = False) -> str:
    if exact_singular:
        return "exact_singular"
    return "exact" if exact else "default"


def run_case(
    binary: Path,
    ndiv: int,
    dof: int,
    exact: bool,
    exact_singular: bool,
    rerun: bool,
) -> Path:
    geom = mode_name(exact, exact_singular)
    out_dir = RUNS / f"{geom}_nd{ndiv}" / f"dof_{dof:02d}"
    input_path = out_dir / "input.txt"
    data_path = out_dir / "single_body_complete.dat"
    log_path = out_dir / "run.log"
    if data_path.exists() and not rerun:
        return data_path

    out_dir.mkdir(parents=True, exist_ok=True)
    input_path.write_text(input_text(ndiv, dof, exact, exact_singular), encoding="utf-8")
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            [str(binary), str(input_path), str(out_dir)],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return data_path


def read_impulse_column(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader)
    l = np.array(
        [
            float(row["lfluid_x_1"]),
            float(row["lfluid_y_1"]),
            float(row["lfluid_z_1"]),
            float(row["lambdafluid_x_1"]),
            float(row["lambdafluid_y_1"]),
            float(row["lambdafluid_z_1"]),
        ]
    )
    return -l


def reconstruct_matrix(
    binary: Path,
    ndiv: int,
    exact: bool,
    exact_singular: bool,
    rerun: bool,
) -> np.ndarray:
    matrix = np.zeros((6, 6), dtype=float)
    for dof in range(6):
        data_path = run_case(binary, ndiv, dof, exact, exact_singular, rerun)
        matrix[:, dof] = read_impulse_column(data_path)
    geom = mode_name(exact, exact_singular)
    out_dir = RUNS / f"{geom}_nd{ndiv}"
    np.save(out_dir / "added_inertia_matrix.npy", matrix)
    np.savetxt(out_dir / "added_inertia_matrix.csv", matrix, delimiter=",")
    return matrix


def error_row(geom: str, ndiv: int, matrix: np.ndarray, analytic: np.ndarray) -> dict[str, float | int | str]:
    diff = matrix - analytic
    diag_matrix = np.diag(np.diag(matrix))
    return {
        "geometry": geom,
        "ndiv": ndiv,
        "h": 2.0 ** (-ndiv),
        "rel_fro_error": np.linalg.norm(diff) / np.linalg.norm(analytic),
        "rel_lin_error": np.linalg.norm(diff[:3, :3]) / np.linalg.norm(analytic[:3, :3]),
        "rel_ang_error": np.linalg.norm(diff[3:, 3:]) / np.linalg.norm(analytic[3:, 3:]),
        "rel_diag_error": np.linalg.norm(np.diag(matrix) - np.diag(analytic))
        / np.linalg.norm(np.diag(analytic)),
        "offdiag_rel": np.linalg.norm(matrix - diag_matrix) / np.linalg.norm(np.diag(matrix)),
        "symmetry_rel": np.linalg.norm(matrix - matrix.T) / np.linalg.norm(matrix),
    }


def add_orders(rows: list[dict[str, float | int | str]], key: str) -> None:
    for geom in sorted({str(r["geometry"]) for r in rows}):
        grows = sorted((r for r in rows if r["geometry"] == geom), key=lambda r: int(r["ndiv"]))
        prev = None
        for row in grows:
            if prev is None:
                row[f"{key}_order"] = ""
            else:
                e0 = float(prev[key])
                e1 = float(row[key])
                row[f"{key}_order"] = math.log(e0 / e1, 2.0) if e0 > 0.0 and e1 > 0.0 else ""
            prev = row


def write_summary(rows: list[dict[str, float | int | str]], analytic: np.ndarray) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    np.savetxt(OUT / "analytic_added_inertia_matrix.csv", analytic, delimiter=",")
    with (OUT / "analytic_added_inertia_diagonal.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dof", "value"])
        for i, value in enumerate(np.diag(analytic)):
            writer.writerow([i, value])

    add_orders(rows, "rel_fro_error")
    add_orders(rows, "rel_lin_error")
    add_orders(rows, "rel_ang_error")

    keys = list(rows[0].keys())
    with (OUT / "single_body_added_inertia_convergence.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_summary(rows: list[dict[str, float | int | str]]) -> None:
    metrics = [
        ("rel_fro_error", "full matrix"),
        ("rel_lin_error", "linear block"),
        ("rel_ang_error", "angular block"),
        ("offdiag_rel", "off-diagonal leakage"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, (key, title) in zip(axes.ravel(), metrics):
        for geom, marker in [("default", "o"), ("exact", "s"), ("exact_singular", "^")]:
            grows = sorted((r for r in rows if r["geometry"] == geom), key=lambda r: int(r["ndiv"]))
            if not grows:
                continue
            x = [int(r["ndiv"]) for r in grows]
            y = [float(r[key]) for r in grows]
            ax.semilogy(x, y, marker=marker, label=geom)
        ax.set_title(title)
        ax.set_xlabel("ndiv")
        ax.set_ylabel("relative error")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
    fig.suptitle("Single-body analytic added-inertia convergence")
    fig.savefig(OUT / "single_body_added_inertia_convergence.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", help="Path to an existing multi_ellip executable")
    parser.add_argument("--target-dir", default="target_exact_geometry_validation")
    parser.add_argument("--build", action="store_true", help="Force rebuild of the release binary")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--max-ndiv", type=int, default=4)
    parser.add_argument("--min-ndiv", type=int, default=1)
    parser.add_argument(
        "--include-exact-singular",
        action="store_true",
        help="Also run exact geometry with experimental exact singular quadrature",
    )
    args = parser.parse_args()

    binary = resolve_binary(args)
    axes = actual_axes()
    analytic = analytic_added_inertia(axes)
    print("axes:", " ".join(f"{x:.12g}" for x in axes), flush=True)
    print("analytic diagonal:", " ".join(f"{x:.12g}" for x in np.diag(analytic)), flush=True)

    rows: list[dict[str, float | int | str]] = []
    modes = [("default", False, False), ("exact", True, False)]
    if args.include_exact_singular:
        modes.append(("exact_singular", True, True))
    for ndiv in range(args.min_ndiv, args.max_ndiv + 1):
        for geom, exact, exact_singular in modes:
            print(f"running {geom} ndiv={ndiv}", flush=True)
            matrix = reconstruct_matrix(binary, ndiv, exact, exact_singular, args.rerun)
            rows.append(error_row(geom, ndiv, matrix, analytic))
            print(
                f"  rel_fro={rows[-1]['rel_fro_error']:.6e} "
                f"lin={rows[-1]['rel_lin_error']:.6e} "
                f"ang={rows[-1]['rel_ang_error']:.6e} "
                f"offdiag={rows[-1]['offdiag_rel']:.6e}",
                flush=True,
            )

    write_summary(rows, analytic)
    plot_summary(rows)
    print(f"wrote {OUT / 'single_body_added_inertia_convergence.csv'}")
    print(f"wrote {OUT / 'single_body_added_inertia_convergence.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
