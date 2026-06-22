import csv
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "studies" / "multibody_convergence_sweep"
MANIFEST = STUDY / "manifest.csv"
FIGURES = STUDY / "figures"
SUMMARY = STUDY / "multibody_convergence_summary.csv"
ORDERS = STUDY / "multibody_convergence_orders.csv"


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def read_output(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    return data


def output_complete(row: dict[str, str]) -> bool:
    output = resolve(row["output"])
    if not output.exists():
        return False
    expected_lines = int(round(float(row["tend"]) / float(row["dt"]))) + 2
    with output.open("r", encoding="utf-8") as f:
        actual_lines = sum(1 for _ in f)
    return actual_lines >= expected_lines


def body_count(data: np.ndarray) -> int:
    n = 0
    while f"px_{n + 1}" in data.dtype.names:
        n += 1
    return n


def vec(data: np.ndarray, prefix: str, body: int) -> np.ndarray:
    return np.column_stack(
        [data[f"{prefix}_x_{body}"], data[f"{prefix}_y_{body}"], data[f"{prefix}_z_{body}"]]
    )


def endpoint_vectors(data: np.ndarray, prefix: str) -> np.ndarray:
    rows = []
    for body in range(1, body_count(data) + 1):
        rows.append(np.array([data[f"{prefix}1_{body}"][-1], data[f"{prefix}2_{body}"][-1], data[f"{prefix}3_{body}"][-1]]))
    return np.vstack(rows)


def endpoint_positions(data: np.ndarray) -> np.ndarray:
    rows = []
    for body in range(1, body_count(data) + 1):
        rows.append(np.array([data[f"px_{body}"][-1], data[f"py_{body}"][-1], data[f"pz_{body}"][-1]]))
    return np.vstack(rows)


def separation(data: np.ndarray) -> float:
    positions = endpoint_positions(data)
    if len(positions) != 2:
        return float("nan")
    return float(np.linalg.norm(positions[1] - positions[0]))


def marker_angle_error(a: np.ndarray, b: np.ndarray) -> float:
    angles = []
    for va, vb in zip(a, b):
        na = float(np.linalg.norm(va))
        nb = float(np.linalg.norm(vb))
        if na == 0.0 or nb == 0.0:
            angles.append(float("nan"))
        else:
            cosang = float(np.dot(va, vb) / (na * nb))
            angles.append(math.acos(max(-1.0, min(1.0, cosang))))
    return float(np.nanmean(angles))


def rms(a: np.ndarray) -> float:
    return float(math.sqrt(np.mean(np.square(a))))


def vector_drift_pct(data: np.ndarray, prefix: str) -> float:
    total = None
    for body in range(1, body_count(data) + 1):
        item = vec(data, prefix, body)
        total = item if total is None else total + item
    if total is None:
        return float("nan")
    scale = max(float(np.linalg.norm(total[0])), 1.0)
    return float(100.0 * np.max(np.linalg.norm(total - total[0], axis=1)) / scale)


def ke_drift_pct(data: np.ndarray) -> float:
    ke0 = float(data["ke_total"][0])
    if ke0 == 0.0:
        return float("nan")
    return float(100.0 * np.max(np.abs(data["ke_total"] / ke0 - 1.0)))


def line_key(row: dict[str, str], suite: str) -> tuple[Any, ...]:
    if suite == "temporal":
        return (
            row["family"],
            row["shape_name"],
            float(row["rho"]),
            float(row["energy_ratio"]),
            float(row["separation"]),
            int(row["ndiv"]),
            suite,
        )
    return (
        row["family"],
        row["shape_name"],
        float(row["rho"]),
        float(row["energy_ratio"]),
        float(row["separation"]),
        float(row["dt"]),
        suite,
    )


def independent_value(row: dict[str, str], suite: str) -> float:
    if suite == "temporal":
        return float(row["dt"])
    ndiv = int(row["ndiv"])
    return 1.0 / (2.0**ndiv)


def reference_row(rows: list[dict[str, str]], suite: str) -> dict[str, str]:
    if suite == "temporal":
        return min(rows, key=lambda r: float(r["dt"]))
    return max(rows, key=lambda r: int(r["ndiv"]))


def fit_order(xs: list[float], ys: list[float]) -> float:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
    return float(slope)


def summarize(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    completed = [row for row in rows if output_complete(row)]
    data = {row["name"]: read_output(resolve(row["output"])) for row in completed}
    summary_rows: list[dict[str, Any]] = []
    order_rows: list[dict[str, Any]] = []

    for row in completed:
        item = data[row["name"]]
        for suite in row["suites"].split(";"):
            summary_rows.append(
                {
                    **{k: row[k] for k in row.keys() if k not in {"input", "output"}},
                    "suite": suite,
                    "independent_value": independent_value(row, suite),
                    "complete": True,
                    "ke_drift_pct": ke_drift_pct(item),
                    "global_p_drift_pct": vector_drift_pct(item, "pcon"),
                    "global_h_drift_pct": vector_drift_pct(item, "hcon"),
                    "final_separation": separation(item),
                    "centroid_error": "",
                    "orientation_angle_error_rad": "",
                    "separation_error": "",
                    "reference": False,
                    "ref_case": "",
                }
            )

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for srow in summary_rows:
        key = line_key({k: str(v) for k, v in srow.items()}, str(srow["suite"]))
        groups.setdefault(key, []).append(srow)

    for key, group in groups.items():
        suite = str(group[0]["suite"])
        ref = reference_row([{k: str(v) for k, v in row.items()} for row in group], suite)
        ref_name = ref["name"]
        ref_data = data[ref_name]
        ref_positions = endpoint_positions(ref_data)
        ref_markers = endpoint_vectors(ref_data, "ofix")
        ref_sep = separation(ref_data)

        for row in group:
            item = data[str(row["name"])]
            row["reference"] = str(row["name"]) == ref_name
            row["ref_case"] = ref_name
            row["centroid_error"] = rms(endpoint_positions(item) - ref_positions)
            row["orientation_angle_error_rad"] = marker_angle_error(endpoint_vectors(item, "ofix"), ref_markers)
            row["separation_error"] = abs(separation(item) - ref_sep)

        fit_group = [row for row in group if not row["reference"]]
        if suite == "grid":
            fit_group = [row for row in fit_group if int(row["ndiv"]) >= 2]
        for metric in ["centroid_error", "orientation_angle_error_rad", "separation_error"]:
            order_rows.append(
                {
                    "family": group[0]["family"],
                    "shape_name": group[0]["shape_name"],
                    "rho": group[0]["rho"],
                    "energy_ratio": group[0]["energy_ratio"],
                    "separation": group[0]["separation"],
                    "suite": suite,
                    "metric": metric,
                    "order": fit_order(
                        [float(row["independent_value"]) for row in fit_group],
                        [float(row[metric]) for row in fit_group],
                    ),
                    "ref_case": ref_name,
                }
            )

    return summary_rows, order_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def label(row: dict[str, Any]) -> str:
    shape = "spheroid" if "spheroid" in str(row["shape_name"]) else "triaxial"
    return f"{shape}, sep={float(row['separation']):g}"


def plot_endpoint(summary_rows: list[dict[str, Any]]) -> None:
    rows = [r for r in summary_rows if r["family"] == "periodic" and not r["reference"]]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.5), constrained_layout=True)
    specs = [
        ("temporal", "centroid_error", "dt", "centroid RMS error"),
        ("temporal", "orientation_angle_error_rad", "dt", "orientation angle error"),
        ("grid", "centroid_error", "h", "centroid RMS error"),
        ("grid", "orientation_angle_error_rad", "h", "orientation angle error"),
    ]
    colors = plt.cm.tab10(np.linspace(0, 1, 6))
    groups: dict[tuple[str, float, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["shape_name"]), float(row["separation"]), str(row["suite"])), []).append(row)
    for ax, (suite, metric, xlabel, ylabel) in zip(axes.flat, specs):
        for i, ((shape, sep, group_suite), group) in enumerate(sorted(groups.items())):
            if group_suite != suite:
                continue
            group = sorted(group, key=lambda r: float(r["independent_value"]))
            x = [float(r["independent_value"]) for r in group]
            y = [float(r[metric]) for r in group]
            ax.loglog(x, y, "o-", color=colors[i % len(colors)], label=label(group[0]))
        ax.invert_xaxis()
        ax.grid(alpha=0.25, which="both")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{suite} {metric.replace('_', ' ')}")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Two-body impulse self-convergence, periodic cases")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "periodic_endpoint_self_convergence.png", dpi=220)
    plt.close(fig)


def plot_energy(summary_rows: list[dict[str, Any]]) -> None:
    rows = [r for r in summary_rows if r["family"] == "stress"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)
    for ax, suite in zip(axes, ["temporal", "grid"]):
        subset = [r for r in rows if r["suite"] == suite]
        groups: dict[tuple[float, float], list[dict[str, Any]]] = {}
        for row in subset:
            groups.setdefault((float(row["rho"]), float(row["separation"])), []).append(row)
        for (rho, sep), group in sorted(groups.items()):
            group = sorted(group, key=lambda r: float(r["independent_value"]))
            x = [float(r["independent_value"]) for r in group]
            y = [float(r["ke_drift_pct"]) for r in group]
            ax.loglog(x, y, "o-", label=f"rho={rho:g}, sep={sep:g}")
        ax.invert_xaxis()
        ax.grid(alpha=0.25, which="both")
        ax.set_xlabel("dt" if suite == "temporal" else "h")
        ax.set_ylabel("max KE drift (%)")
        ax.set_title(f"stress {suite}")
    axes[0].legend(fontsize=8)
    fig.suptitle("Close-contact stress energy drift")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "stress_energy_drift.png", dpi=220)
    plt.close(fig)


def plot_conservation(summary_rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), constrained_layout=True)
    for ax, metric, title in [
        (axes[0], "global_p_drift_pct", "linear momentum"),
        (axes[1], "global_h_drift_pct", "angular momentum"),
    ]:
        for family, marker in [("periodic", "o"), ("stress", "s")]:
            subset = [r for r in summary_rows if r["family"] == family]
            ax.loglog(
                [float(r["ke_drift_pct"]) for r in subset],
                [float(r[metric]) for r in subset],
                marker,
                linestyle="none",
                alpha=0.75,
                label=family,
            )
        ax.grid(alpha=0.25, which="both")
        ax.set_xlabel("max KE drift (%)")
        ax.set_ylabel("global drift (%)")
        ax.set_title(title)
    axes[0].legend(fontsize=8)
    fig.suptitle("Global momentum conservation versus energy drift")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "momentum_vs_energy_drift.png", dpi=220)
    plt.close(fig)


def main() -> None:
    rows = read_manifest(MANIFEST)
    summary_rows, order_rows = summarize(rows)
    write_csv(SUMMARY, summary_rows)
    write_csv(ORDERS, order_rows)
    plot_endpoint(summary_rows)
    plot_energy(summary_rows)
    plot_conservation(summary_rows)

    complete_cases = len({row["name"] for row in summary_rows})
    print(f"Completed cases: {complete_cases}/{len(rows)}")
    print(f"Wrote {SUMMARY}")
    print(f"Wrote {ORDERS}")
    print(f"Wrote figures under {FIGURES}")


if __name__ == "__main__":
    main()
