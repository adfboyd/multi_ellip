#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean_by(rows: list[dict[str, str]], key: str) -> dict[tuple[str, str, int], float]:
    buckets: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        tri_quad = row.get("tri_quad", "trgl13")
        buckets[(row["mode"], tri_quad, int(row["ndiv"]))].append(float(row[key]))
    return {k: sum(v) / len(v) for k, v in buckets.items()}


def plot(csv_paths: list[Path], output: Path) -> None:
    rows: list[dict[str, str]] = []
    for path in csv_paths:
        rows.extend(read_rows(path))

    metrics = [
        ("solved_added_rel_error", "BEM-solved added inertia"),
        ("analytic_phi_added_rel_error", "analytic phi through impulse integral"),
        ("phi_rel_error", "solved phi vs analytic phi"),
        ("analytic_residual_rel", "analytic phi residual in discrete BEM"),
    ]
    modes = ["default", "exact", "exact_singular"]
    mode_markers = {"default": "o", "exact": "s", "exact_singular": "^"}
    quad_styles = {"trgl13": "-", "duffy8": "--"}

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, (key, title) in zip(axes.ravel(), metrics):
        data = mean_by(rows, key)
        for mode in modes:
            quads = sorted({quad for (m, quad, _ndiv) in data if m == mode})
            for quad in quads:
                points = sorted((ndiv, val) for (m, q, ndiv), val in data.items() if m == mode and q == quad)
                if not points:
                    continue
                x, y = zip(*points)
                label = mode if quad == "trgl13" else f"{mode} ({quad})"
                ax.semilogy(
                    x,
                    y,
                    marker=mode_markers.get(mode, "o"),
                    linestyle=quad_styles.get(quad, ":"),
                    label=label,
                )
        ax.set_title(title)
        ax.set_xlabel("ndiv")
        ax.set_ylabel("relative error")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("Density/collocation diagnostic for single ellipsoid")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv",
        nargs="*",
        type=Path,
        default=[ROOT / "scratch_density_collocation_probe.csv"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "scratch_density_collocation_probe.png",
    )
    args = parser.parse_args()
    plot([p if p.is_absolute() else ROOT / p for p in args.csv], args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
