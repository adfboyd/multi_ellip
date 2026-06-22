from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = (
    ROOT
    / "docs"
    / "paper_figures"
    / "ke_ratio_recurrence_current_comparison"
    / "ke_ratio_density_recurrence_current_comparison_summary.csv"
)
DEFAULT_OUT = ROOT / "docs" / "paper_figures" / "representative_single_body_orbits"


@dataclass(frozen=True)
class Row:
    family: str
    source: str
    method: str
    case: str
    path: Path
    ratio: float
    rho: float
    run: int
    cls: str
    chaos_score: float
    broadband_chaos_score: float


def parse_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return math.nan


def parse_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open(newline="", encoding="utf-8") as f:
        for record in csv.DictReader(f):
            rows.append(
                Row(
                    family=record["family"],
                    source=record["source"],
                    method=record["method"],
                    case=record["case"],
                    path=Path(record["path"]),
                    ratio=parse_float(record["ratio"]),
                    rho=parse_float(record["rho"]),
                    run=int(float(record["run"])),
                    cls=record["class"],
                    chaos_score=parse_float(record["chaos_score"]),
                    broadband_chaos_score=parse_float(record["broadband_chaos_score"]),
                )
            )
    return rows


def read_numeric_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        cols: dict[str, list[float]] = {}
        for row in reader:
            for key, value in row.items():
                if key is None:
                    continue
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    number = math.nan
                cols.setdefault(key.strip(), []).append(number)
    return {key: np.asarray(values, dtype=float) for key, values in cols.items()}


def marker_columns(row: Row) -> tuple[str, str, str]:
    if row.method == "exact_added_mass":
        return ("ofx", "ofy", "ofz")
    return ("ofix1_1", "ofix2_1", "ofix3_1")


def load_marker(row: Row) -> tuple[np.ndarray, np.ndarray]:
    cols = read_numeric_csv(row.path)
    names = marker_columns(row)
    missing = [name for name in ("time", *names) if name not in cols]
    if missing:
        raise KeyError(f"{row.path} is missing {missing}")
    marker = np.column_stack([cols[name] for name in names])
    time = cols["time"]
    finite = np.isfinite(time) & np.all(np.isfinite(marker), axis=1)
    return time[finite], marker[finite]


def set_equal_axes(ax: plt.Axes, xyz: np.ndarray) -> None:
    lo = np.nanmin(xyz, axis=0)
    hi = np.nanmax(xyz, axis=0)
    center = 0.5 * (lo + hi)
    radius = 0.55 * max(float(np.max(hi - lo)), 1.0e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_marker_orbit(ax: plt.Axes, row: Row, title: str) -> None:
    time, marker = load_marker(row)
    if len(time) > 1200:
        # Keep the path readable while retaining the full time interval.
        step = math.ceil(len(time) / 1200)
        time = time[::step]
        marker = marker[::step]

    points = marker.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    lc = Line3DCollection(segments, cmap="viridis", linewidth=0.55, alpha=0.92)
    lc.set_array(time[:-1])
    ax.add_collection3d(lc)
    ax.scatter(*marker[0], s=22, color="black", label="start", zorder=4)
    ax.scatter(*marker[-1], s=30, color="#d62728", marker="x", label="end", zorder=4)
    set_equal_axes(ax, marker)
    ax.set_xlabel("$m_x$")
    ax.set_ylabel("$m_y$")
    ax.set_zlabel("$m_z$")
    ax.set_title(title, pad=8)
    ax.view_init(elev=24, azim=-52)
    ax.grid(alpha=0.18)


def pretty_source(source: str) -> str:
    return {
        "triaxial_nd2": "Triaxial impulse, ndiv=2",
        "triaxial_nd3": "Triaxial impulse, ndiv=3",
        "triaxial_exact": "Triaxial exact added mass",
        "spheroid_nd2": "Spheroid impulse, ndiv=2",
        "spheroid_exact": "Spheroid exact added mass",
    }.get(source, source)


def choose_examples(rows: list[Row], source: str) -> tuple[Row, Row | None]:
    source_rows = [row for row in rows if row.source == source]
    regular = [
        row
        for row in source_rows
        if row.cls in {"quasi-periodic", "periodic", "sensitive-regular", "complex-regular"}
    ]
    chaotic = [row for row in source_rows if row.cls == "chaotic-like"]
    if not regular:
        raise ValueError(f"No regular/quasi examples found for {source}")
    regular_choice = min(
        regular,
        key=lambda row: (
            0 if row.cls == "quasi-periodic" else 1,
            row.broadband_chaos_score,
            row.chaos_score,
            row.run,
        ),
    )
    chaotic_choice = None
    if chaotic:
        chaotic_choice = max(
            chaotic,
            key=lambda row: (row.broadband_chaos_score, row.chaos_score, -row.run),
        )
    return regular_choice, chaotic_choice


def plot_source(rows: list[Row], source: str, out_dir: Path) -> Path:
    regular, chaotic = choose_examples(rows, source)
    ncols = 2 if chaotic is not None else 1
    fig = plt.figure(figsize=(5.2 * ncols, 4.8), constrained_layout=True)
    ax = fig.add_subplot(1, ncols, 1, projection="3d")
    plot_marker_orbit(
        ax,
        regular,
        (
            f"Regular/quasiperiodic\n"
            f"{regular.case}, rho={regular.rho:g}, E={regular.ratio:g}, "
            f"score={regular.broadband_chaos_score:.2f}"
        ),
    )
    if chaotic is not None:
        ax = fig.add_subplot(1, ncols, 2, projection="3d")
        plot_marker_orbit(
            ax,
            chaotic,
            (
                f"Chaotic-like\n"
                f"{chaotic.case}, rho={chaotic.rho:g}, E={chaotic.ratio:g}, "
                f"score={chaotic.broadband_chaos_score:.2f}"
            ),
        )
    fig.suptitle(pretty_source(source), fontsize=13)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{source}_representative_marker_orbits.png"
    fig.savefig(out, dpi=240)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot representative single-body marker orbits.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["triaxial_nd2", "triaxial_nd3", "triaxial_exact", "spheroid_nd2", "spheroid_exact"],
    )
    args = parser.parse_args()

    rows = parse_rows(args.summary)
    for source in args.sources:
        out = plot_source(rows, source, args.out)
        print(out)


if __name__ == "__main__":
    main()
