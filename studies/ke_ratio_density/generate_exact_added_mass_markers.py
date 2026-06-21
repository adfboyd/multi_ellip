from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "studies" / "ke_ratio_density"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import exact_added_mass as exact  # noqa: E402


def load_output(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    return data


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


def fmt(value: float) -> str:
    return "" if not np.isfinite(value) else f"{float(value):.12g}"


def case_dirs(run_root: Path) -> list[Path]:
    return sorted(path for path in run_root.iterdir() if (path / "input.txt").exists())


def group_label(case: str) -> str:
    return "ratio1_rho4" if case.startswith("ratio1_") else "ratio20_rho0p25"


def run_number(case: str) -> int:
    try:
        return int(case.rsplit("_run", 1)[1])
    except (IndexError, ValueError):
        return 0


def lin_rot_ratio_from_case(case: str) -> float:
    return 1.0 if case.startswith("ratio1_") else 20.0


def span(data: np.ndarray, *cols: str) -> float:
    values = np.column_stack([data[col] for col in cols])
    return float(np.max(np.linalg.norm(values - values[0], axis=1)))


def analyse(case: str, params: dict[str, float], data: np.ndarray, out_csv: Path) -> dict[str, str]:
    ke0 = float(data["ke_total"][0])
    drift = 100.0 * (data["ke_total"] - ke0) / ke0
    return {
        "case": case,
        "density": fmt(params.get("rhos1", float("nan"))),
        "target_linKE_rotKE": fmt(lin_rot_ratio_from_case(case)),
        "repeat": str(run_number(case)),
        "shape_x": fmt(params.get("shx1", float("nan"))),
        "shape_y": fmt(params.get("shy1", float("nan"))),
        "shape_z": fmt(params.get("shz1", float("nan"))),
        "ndiv": "exact",
        "dt": fmt(params.get("dt", float("nan"))),
        "tend": fmt(params.get("tend", float("nan"))),
        "vx": fmt(params.get("lvx1", float("nan"))),
        "vy": fmt(params.get("lvy1", float("nan"))),
        "vz": fmt(params.get("lvz1", float("nan"))),
        "wx": fmt(params.get("avx1", float("nan"))),
        "wy": fmt(params.get("avy1", float("nan"))),
        "wz": fmt(params.get("avz1", float("nan"))),
        "ke0": fmt(ke0),
        "solid_ke0": fmt(float(data["ke_solid"][0])),
        "fluid_ke0": fmt(float(data["ke_fluid"][0])),
        "end_drift_pct": fmt(float(drift[-1])),
        "max_abs_drift_pct": fmt(float(np.max(np.abs(drift)))),
        "path_span": fmt(span(data, "px", "py", "pz")),
        "ofix_span": fmt(span(data, "ofx", "ofy", "ofz")),
        "output": str(out_csv),
    }


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def generate_exact_csv(case_dir: Path, rerun: bool) -> tuple[np.ndarray, dict[str, float], Path]:
    input_path = case_dir / "input.txt"
    out_csv = case_dir / "exact_added_mass.csv"
    params = parse_input(input_path)
    if rerun or not out_csv.exists():
        data, header, _ = exact.run(input_path, t_end=params.get("tend"), n_out=int(round(params.get("tend", 100.0) / params.get("dt", 0.05))) + 1)
        np.savetxt(out_csv, data, delimiter=",", header=header, comments="")
    data = load_output(out_csv)
    return data, params, out_csv


def plot_dashboard(rows: list[dict[str, str]], data_by_case: dict[str, np.ndarray], out: Path, title_suffix: str) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    colors = {"ratio1_rho4": "#1f77b4", "ratio20_rho0p25": "#d62728"}
    labels_done: set[tuple[str, str]] = set()

    for row in sorted(rows, key=lambda r: (group_label(r["case"]), run_number(r["case"]))):
        data = data_by_case[row["case"]]
        t = data["time"]
        drift = 100.0 * (data["ke_total"] - data["ke_total"][0]) / data["ke_total"][0]
        group = group_label(row["case"])
        color = colors[group]
        label = group.replace("ratio", "ratio ") if ("drift", group) not in labels_done else None
        labels_done.add(("drift", group))
        ax[0, 0].plot(t, drift, lw=1.0, alpha=0.55, color=color, label=label)

        solid_label = f"{group.replace('ratio', 'ratio ')} solid" if ("solid", group) not in labels_done else None
        fluid_label = f"{group.replace('ratio', 'ratio ')} fluid" if ("fluid", group) not in labels_done else None
        labels_done.add(("solid", group))
        labels_done.add(("fluid", group))
        ax[0, 1].plot(t, 100.0 * data["ke_solid"] / data["ke_total"][0], lw=0.9, alpha=0.45, color=color, label=solid_label)
        ax[0, 1].plot(t, 100.0 * data["ke_fluid"] / data["ke_total"][0], lw=0.9, ls="--", alpha=0.45, color=color, label=fluid_label)
        ax[1, 0].plot(data["px"], data["py"], lw=0.9, alpha=0.45, color=color)
        ax[1, 1].plot(data["ofx"], data["ofy"], lw=0.75, alpha=0.35, color=color)

    ax[0, 0].axhline(0.0, lw=0.8, color="black")
    ax[0, 0].set(title="Total KE drift", xlabel="t", ylabel="drift (%)")
    ax[0, 1].set(title="Energy exchange", xlabel="t", ylabel="% initial total KE")
    ax[1, 0].set(title="Centre path projection", xlabel="x", ylabel="y")
    ax[1, 1].set(title="Orientation marker projection", xlabel="ofix x", ylabel="ofix y")
    for axis in ax.flat:
        axis.grid(alpha=0.25)
    ax[0, 0].legend(fontsize=8)
    ax[0, 1].legend(fontsize=8)
    fig.suptitle(f"Exact added-mass single-body reference, {title_suffix}, t=0..100", fontsize=14)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def plot_orientation_panels(rows: list[dict[str, str]], data_by_case: dict[str, np.ndarray], out: Path, title_suffix: str) -> None:
    ordered = sorted(rows, key=lambda r: (group_label(r["case"]), run_number(r["case"])))
    fig, axes = plt.subplots(2, 5, figsize=(18, 8), sharex=True, sharey=True, constrained_layout=True)
    norm = plt.Normalize(0.0, 100.0)
    cmap = plt.get_cmap("viridis")

    for axis, row in zip(axes.flat, ordered):
        data = data_by_case[row["case"]]
        axis.scatter(data["ofx"], data["ofy"], c=data["time"], cmap=cmap, norm=norm, s=3.0, linewidths=0.0)
        axis.plot(data["ofx"], data["ofy"], lw=0.35, alpha=0.28, color="#303030")
        axis.scatter(data["ofx"][0], data["ofy"][0], s=22, c="black", marker="o", zorder=3)
        axis.scatter(data["ofx"][-1], data["ofy"][-1], s=28, c="#d62728", marker="x", zorder=3)
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
    fig.suptitle(f"Exact added-mass orientation marker projections, {title_suffix}, t=0..100", fontsize=14)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate exact added-mass marker plots for KE-ratio runs.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--suffix", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    run_root = args.run_root if args.run_root.is_absolute() else ROOT / args.run_root
    rows: list[dict[str, str]] = []
    data_by_case: dict[str, np.ndarray] = {}
    for case_dir in case_dirs(run_root):
        data, params, out_csv = generate_exact_csv(case_dir, args.rerun)
        rows.append(analyse(case_dir.name, params, data, out_csv))
        data_by_case[case_dir.name] = data

    summary = STUDY / f"ke_ratio_density_exact_{args.suffix}_summary.csv"
    dashboard = STUDY / f"ke_ratio_density_exact_{args.suffix}_dashboard.png"
    panels = STUDY / f"exact_orientation_marker_panels_{args.suffix}.png"
    write_rows(summary, rows)
    plot_dashboard(rows, data_by_case, dashboard, args.title)
    plot_orientation_panels(rows, data_by_case, panels, args.title)
    print(f"cases: {len(rows)}")
    print(f"summary: {summary}")
    print(f"dashboard: {dashboard}")
    print(f"panels: {panels}")


if __name__ == "__main__":
    main()
