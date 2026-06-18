from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = ROOT / "two_body_parameter_sweep_task_runs"
METRICS = [
    ("max_abs_ke_drift_pct", "max |KE drift| (%)", "viridis"),
    ("max_pcon_drift_pct", "max linear momentum drift (%)", "magma"),
    ("max_hcon_drift_pct", "max angular momentum drift (%)", "magma"),
    ("coupled_true_energy_error_rel", "coupled true energy err", "viridis"),
    ("coupled_residual", "coupled residual", "viridis"),
    ("mean_step_s", "mean step time (s)", "cividis"),
]
WORST_COLUMNS = [
    "name",
    "shape_name",
    "rho",
    "energy_ratio",
    "separation",
    "repeat",
    "status",
    "max_abs_ke_drift_pct",
    "max_pcon_drift_pct",
    "max_hcon_drift_pct",
    "coupled_residual",
    "coupled_true_energy_error_rel",
    "hamiltonian_adaptive_retries",
    "hamiltonian_max_substeps_used",
    "mean_step_s",
    "final_separation",
]


def parse_float(row: dict[str, str], key: str) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value not in ("", None) else float("nan")
    except ValueError:
        return float("nan")


def fmt(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.12g}"


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def group_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row["shape_name"],
        row["rho"],
        row["energy_ratio"],
        row["separation"],
    )


def aggregate(rows: list[dict[str, str]], manifest_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    expected: dict[tuple[str, str, str, str], int] = defaultdict(int)
    for row in manifest_rows:
        expected[group_key(row)] += 1

    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)

    out: list[dict[str, object]] = []
    for key in sorted(set(expected) | set(grouped)):
        group = grouped.get(key, [])
        ok = [row for row in group if row.get("status") in {"OK", "POST"}]
        item: dict[str, object] = {
            "shape_name": key[0],
            "rho": key[1],
            "energy_ratio": key[2],
            "separation": key[3],
            "n_expected": expected.get(key, 0),
            "n_completed": len(group),
            "n_ok": len(ok),
        }
        for metric, _, _ in METRICS:
            values = [parse_float(row, metric) for row in ok]
            values = [v for v in values if math.isfinite(v)]
            item[f"mean_{metric}"] = fmt(mean(values)) if values else ""
            item[f"max_{metric}"] = fmt(max(values)) if values else ""
        sep_values = [parse_float(row, "min_separation") for row in ok]
        sep_values = [v for v in sep_values if math.isfinite(v)]
        item["min_min_separation"] = fmt(min(sep_values)) if sep_values else ""
        out.append(item)
    return out


def worst_cases(rows: list[dict[str, str]], limit: int = 20) -> list[dict[str, object]]:
    scored = []
    for row in rows:
        score = max(
            parse_float(row, "max_abs_ke_drift_pct"),
            parse_float(row, "max_pcon_drift_pct"),
            parse_float(row, "max_hcon_drift_pct"),
            100.0 * parse_float(row, "coupled_true_energy_error_rel"),
        )
        if math.isfinite(score):
            scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for score, row in scored[:limit]:
        out.append({"triage_score": fmt(score), **{key: row.get(key, "") for key in WORST_COLUMNS}})
    return out


def metric_matrix(
    rows: list[dict[str, object]],
    shape: str,
    metric: str,
) -> tuple[np.ndarray, list[float], list[tuple[float, float]]]:
    sub = [row for row in rows if row["shape_name"] == shape]
    xs = sorted({float(row["energy_ratio"]) for row in sub})
    ys = sorted({(float(row["rho"]), float(row["separation"])) for row in sub}, reverse=True)
    mat = np.full((len(ys), len(xs)), np.nan)
    x_index = {x: i for i, x in enumerate(xs)}
    y_index = {y: i for i, y in enumerate(ys)}
    key = f"mean_{metric}"
    for row in sub:
        value = row.get(key, "")
        try:
            mat[y_index[(float(row["rho"]), float(row["separation"]))], x_index[float(row["energy_ratio"])]] = float(value)
        except (TypeError, ValueError):
            pass
    return mat, xs, ys


def plot_overview(aggregate_rows: list[dict[str, object]], out: Path) -> None:
    if not aggregate_rows:
        return
    shapes = sorted({str(row["shape_name"]) for row in aggregate_rows})
    fig, axes = plt.subplots(
        len(shapes),
        len(METRICS),
        figsize=(3.35 * len(METRICS), 4.2 * len(shapes)),
        constrained_layout=True,
        squeeze=False,
    )
    for r, shape in enumerate(shapes):
        for c, (metric, title, cmap) in enumerate(METRICS):
            ax = axes[r][c]
            mat, xs, ys = metric_matrix(aggregate_rows, shape, metric)
            im = ax.imshow(mat, aspect="auto", cmap=cmap)
            ax.set_title(title if r == 0 else "")
            ax.set_xticks(range(len(xs)), [f"{x:g}" for x in xs])
            ax.set_yticks(range(len(ys)), [f"rho={rho:g}\nsep={sep:g}" for rho, sep in ys])
            ax.set_xlabel("E")
            if c == 0:
                ax.set_ylabel(shape)
            for yi in range(mat.shape[0]):
                for xi in range(mat.shape[1]):
                    value = mat[yi, xi]
                    label = "" if not math.isfinite(value) else f"{value:.2g}"
                    ax.text(xi, yi, label, ha="center", va="center", fontsize=7, color="white")
            fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle("Two-body coupled endpoint sweep overview: group means over repeats", fontsize=14)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def write_notes(
    out: Path,
    run_root: Path,
    rows: list[dict[str, str]],
    aggregate_rows: list[dict[str, object]],
    worst: list[dict[str, object]],
) -> None:
    completed = sum(1 for row in rows if row.get("status") in {"OK", "POST"})
    total_groups = len(aggregate_rows)
    worst_line = "n/a"
    if worst:
        top = worst[0]
        worst_line = (
            f"{top.get('name', '')}: score={top.get('triage_score', '')}, "
            f"max_ke={top.get('max_abs_ke_drift_pct', '')}, "
            f"max_h={top.get('max_hcon_drift_pct', '')}"
        )
    out.write_text(
        "\n".join(
            [
                "# Two-body coupled endpoint sweep analysis",
                "",
                "## Questions this sweep is meant to answer",
                "",
                "- Does the coupled endpoint Hamiltonian midpoint scheme keep total kinetic energy bounded across density, spin ratio, shape, and separation?",
                "- Where do linear and angular momentum diagnostics degrade, and are those degradations correlated with density, high rotational energy, or close initial separation?",
                "- How often does the nonlinear coupled solve struggle: large residuals, adaptive substep retries, or max-substep escalation?",
                "- Do the trajectories look periodic, quasi-periodic, or chaotic under the recurrence/spectral classifier, and do repeat runs agree?",
                "- Which cases are worth rerunning at smaller `dt`, higher `ndiv`, or different solver tolerances?",
                "",
                "## Current artifacts",
                "",
                f"- Run root: `{run_root}`",
                f"- Completed rows in summary: `{completed}`",
                f"- Parameter groups represented: `{total_groups}`",
                "- `aggregate_by_params.csv`: grouped means/maxima over repeats.",
                "- `worst_cases.csv`: triage list sorted by conservation/solver-error score.",
                "- `sweep_overview.png`: heatmap dashboard over shape, density, separation, and energy ratio.",
                "- `two_body_dynamics_classification*.csv/png`: recurrence/spectral classification outputs, generated after all runs are complete.",
                "",
                "## Current worst case",
                "",
                worst_line,
                "",
                "## Suggested interpretation",
                "",
                "- Treat low KE drift but high angular momentum drift as a diagnostic mismatch or impulse-origin issue to inspect, not automatically a physical failure.",
                "- Treat adaptive retries or `hamiltonian_max_substeps_used > 1` as solver difficulty, even when KE conservation still looks good.",
                "- Compare repeat classifications before making claims about periodic/chaotic regions; disagreement means sensitivity to initial direction rather than a clean phase boundary.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize completed two-body sweep outputs.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    run_root = args.run_root if args.run_root.is_absolute() else ROOT / args.run_root
    manifest = args.manifest or run_root / "manifest.csv"
    summary = args.summary or run_root / "run_summary.csv"
    out_dir = args.out_dir or run_root / "analysis"
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    if not summary.is_absolute():
        summary = ROOT / summary
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_rows(manifest)
    rows = read_rows(summary)
    aggregate_rows = aggregate(rows, manifest_rows)
    worst = worst_cases(rows)

    aggregate_csv = out_dir / "aggregate_by_params.csv"
    worst_csv = out_dir / "worst_cases.csv"
    overview_png = out_dir / "sweep_overview.png"
    notes_md = out_dir / "ANALYSIS_NOTES.md"

    write_rows(aggregate_csv, aggregate_rows)
    write_rows(worst_csv, worst)
    plot_overview(aggregate_rows, overview_png)
    write_notes(notes_md, run_root, rows, aggregate_rows, worst)

    print(f"Wrote {aggregate_csv}")
    print(f"Wrote {worst_csv}")
    print(f"Wrote {overview_png}")
    print(f"Wrote {notes_md}")


if __name__ == "__main__":
    main()
