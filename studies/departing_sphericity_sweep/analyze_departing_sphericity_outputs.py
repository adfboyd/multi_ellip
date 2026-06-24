from __future__ import annotations

import csv
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from studies.two_body_parameter_sweep import classify_two_body_dynamics as dyn
from studies.two_body_paper_sweeps import analyze_paper_sweep_outputs as metrics


STUDY = ROOT / "studies" / "departing_sphericity_sweep"
OUT_DIR = STUDY / "analysis_overview"
DOCS_DIR = ROOT / "docs" / "paper_figures" / "section4_departing_sphericity"
MANIFESTS = [
    STUDY / "manifest_spheroid_oblate_all.csv",
    STUDY / "manifest_spheroid_prolate_all.csv",
    STUDY / "manifest_triaxial_all.csv",
]
FAMILIES = ["spheroid_oblate", "spheroid_prolate", "triaxial"]
SUITES = ["homogeneous", "heterogeneous_sphere", "heterogeneous_aspherical"]
CLASS_ORDER = {
    "periodic": 0,
    "quasi-periodic": 1,
    "sensitive-regular": 2,
    "complex-regular": 3,
    "mixed": 4,
    "chaotic-like": 5,
    "chaotic": 5,
    "chaotic-candidate": 5,
    "incomplete": 6,
    "": np.nan,
}
CLASS_CODE = {
    "periodic": "P",
    "quasi-periodic": "Q",
    "sensitive-regular": "S",
    "complex-regular": "R",
    "mixed": "M",
    "chaotic-like": "C",
    "chaotic": "C",
    "chaotic-candidate": "C",
    "incomplete": "I",
    "": "",
}
CLASS_COLORS = ["#2ca02c", "#1f77b4", "#9467bd", "#17becf", "#7f7f7f", "#d62728", "#9e9e9e"]
SUITE_LABEL = {
    "homogeneous": "same eps",
    "heterogeneous_sphere": "eps + sphere",
    "heterogeneous_aspherical": "eps + eps=1",
}
FAMILY_LABEL = {
    "spheroid_oblate": "oblate spheroids",
    "spheroid_prolate": "prolate spheroids",
    "triaxial": "triaxial ellipsoids",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def fmt(value: float) -> str:
    return "" if not math.isfinite(value) else f"{value:.12g}"


def fmean(values: list[object]) -> float:
    out = []
    for value in values:
        try:
            fvalue = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fvalue):
            out.append(fvalue)
    return float(np.mean(out)) if out else float("nan")


def state_mode(row: dict[str, str]) -> str:
    return "quaternion" if row["family"] == "triaxial" else "axis"


def analysis_row(row: dict[str, str], transient_fraction: float) -> dict[str, object]:
    classifier_row = dict(row)
    classifier_row["shape_name"] = f"{row['family']}:{row['suite']}:eps={float(row['epsilon']):.12g}"
    result = dyn.analyze_run(classifier_row, transient_fraction, state_mode(row))
    run_metrics = metrics.output_metrics(row)
    run_status = run_metrics.pop("status", "")
    class_status = result.pop("status", "")
    return {
        **row,
        **run_metrics,
        **result,
        "run_status": run_status,
        "classification_status": class_status,
        "family": row["family"],
        "suite": row["suite"],
        "epsilon": row["epsilon"],
        "state_mode": state_mode(row),
    }


def run_complete(row: dict[str, object]) -> bool:
    if row.get("run_status") == "complete":
        return True
    try:
        return int(float(row.get("n_rows", 0))) >= int(float(row.get("expected_rows", 1)))
    except (TypeError, ValueError):
        return False


def classification_ok(row: dict[str, object]) -> bool:
    return row.get("classification_status", row.get("status")) == "ok"


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, float], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["family"]), str(row["suite"]), float(row["epsilon"]))].append(row)

    out: list[dict[str, object]] = []
    for (family, suite, epsilon), group in sorted(grouped.items()):
        ok = [row for row in group if classification_ok(row)]
        complete = [row for row in group if run_complete(row)]
        classes: list[str] = []
        for row in ok:
            classes.extend([str(row.get("class_body1", "incomplete")), str(row.get("class_body2", "incomplete"))])
        if not classes:
            classes = ["incomplete"]
        counts = Counter(classes)
        broadband = []
        chaos = []
        entropy = []
        for row in ok:
            for body in ("body1", "body2"):
                broadband.append(row.get(f"{body}_broadband_chaos_score"))
                chaos.append(row.get(f"{body}_chaos_score"))
                entropy.append(row.get(f"{body}_spectral_entropy"))
        group_class = dyn.consensus(classes)
        out.append(
            {
                "family": family,
                "suite": suite,
                "epsilon": epsilon,
                "n_runs": len(group),
                "n_complete": len(complete),
                "n_ok": len(ok),
                "n_body_observations": len(classes),
                "group_class": group_class,
                "repeat_classes": ";".join(classes),
                "body_broadband_chaos_score_mean": fmt(fmean(broadband)),
                "body_chaos_score_mean": fmt(fmean(chaos)),
                "body_spectral_entropy_mean": fmt(fmean(entropy)),
                "mean_max_abs_ke_drift_pct": fmt(fmean([row.get("max_abs_ke_drift_pct") for row in complete])),
                "mean_min_separation": fmt(fmean([row.get("min_separation") for row in complete])),
                "mean_max_impulse_global_p_drift": fmt(
                    fmean([row.get("max_impulse_global_p_drift") for row in complete])
                ),
                "mean_max_impulse_global_h_drift": fmt(
                    fmean([row.get("max_impulse_global_h_drift") for row in complete])
                ),
                **{f"n_{name}": counts.get(name, 0) for name in dyn.CLASS_ORDER},
            }
        )
    return out


def eps_label(value: float) -> str:
    return "0" if value == 0 else f"{value:g}"


def plot_class_map(summary: list[dict[str, object]], out: Path) -> None:
    eps_values = sorted({float(row["epsilon"]) for row in summary})
    fig, axes = plt.subplots(1, len(FAMILIES), figsize=(15, 5.0), constrained_layout=True)
    cmap = matplotlib.colors.ListedColormap(CLASS_COLORS)
    norm = matplotlib.colors.BoundaryNorm(np.arange(-0.5, len(CLASS_COLORS) + 0.5), cmap.N)
    for ax, family in zip(axes, FAMILIES):
        matrix = np.full((len(SUITES), len(eps_values)), np.nan)
        code = [["" for _ in eps_values] for _ in SUITES]
        for row in summary:
            if row["family"] != family:
                continue
            y = SUITES.index(str(row["suite"]))
            x = eps_values.index(float(row["epsilon"]))
            cls = str(row["group_class"])
            matrix[y, x] = CLASS_ORDER.get(cls, np.nan)
            code[y][x] = CLASS_CODE.get(cls, "?")
        ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
        ax.set_title(FAMILY_LABEL[family])
        ax.set_xticks(range(len(eps_values)), [eps_label(v) for v in eps_values], rotation=45, ha="right")
        ax.set_yticks(range(len(SUITES)), [SUITE_LABEL[s] for s in SUITES])
        ax.set_xlabel("epsilon")
        for y in range(len(SUITES)):
            for x in range(len(eps_values)):
                ax.text(x, y, code[y][x], ha="center", va="center", color="white", fontsize=10)
    fig.suptitle("Departing-sphericity behaviour class")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_metric_lines(summary: list[dict[str, object]], metric: str, ylabel: str, out: Path, log_y: bool = False) -> None:
    fig, axes = plt.subplots(1, len(FAMILIES), figsize=(15, 4.8), sharey=False, constrained_layout=True)
    markers = {"homogeneous": "o", "heterogeneous_sphere": "s", "heterogeneous_aspherical": "^"}
    for ax, family in zip(axes, FAMILIES):
        rows = [row for row in summary if row["family"] == family]
        for suite in SUITES:
            sub = sorted([row for row in rows if row["suite"] == suite], key=lambda r: float(r["epsilon"]))
            x = [max(float(row["epsilon"]), 1.0e-5) for row in sub]
            y = [float(row[metric]) if row.get(metric) not in ("", None) else np.nan for row in sub]
            ax.plot(x, y, marker=markers[suite], linewidth=1.5, label=SUITE_LABEL[suite])
        ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.set_title(FAMILY_LABEL[family])
        ax.set_xlabel("epsilon (0 plotted at 1e-5)")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(loc="best", fontsize=8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def mirror_docs() -> None:
    try:
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        for png in OUT_DIR.glob("*.png"):
            shutil.copy2(png, DOCS_DIR / png.name)
    except PermissionError as exc:
        print(f"Skipping docs mirror: {exc}")


def main() -> None:
    cached_run_metrics = OUT_DIR / "departing_sphericity_run_metrics.csv"
    if cached_run_metrics.exists():
        print(f"Using cached run metrics: {cached_run_metrics}")
        run_rows = read_csv(cached_run_metrics)
    else:
        manifest_rows: list[dict[str, str]] = []
        for manifest in MANIFESTS:
            manifest_rows.extend(read_csv(manifest))

        run_rows = []
        for i, row in enumerate(manifest_rows, start=1):
            if i == 1 or i == len(manifest_rows) or i % 25 == 0:
                print(f"[{i:03d}/{len(manifest_rows):03d}] {row['name']}", flush=True)
            run_rows.append(analysis_row(row, transient_fraction=0.5))
        write_csv(cached_run_metrics, run_rows)

    summary = summarize(run_rows)
    write_csv(OUT_DIR / "departing_sphericity_summary.csv", summary)
    plot_class_map(summary, OUT_DIR / "departing_sphericity_class_map.png")
    plot_metric_lines(
        summary,
        "body_broadband_chaos_score_mean",
        "mean broadband recurrence score",
        OUT_DIR / "departing_sphericity_broadband_score.png",
    )
    plot_metric_lines(
        summary,
        "mean_max_abs_ke_drift_pct",
        "mean max |KE drift| (%)",
        OUT_DIR / "departing_sphericity_energy_drift.png",
        log_y=True,
    )
    plot_metric_lines(
        summary,
        "mean_min_separation",
        "mean minimum centre separation",
        OUT_DIR / "departing_sphericity_min_separation.png",
    )
    mirror_docs()

    print(f"Wrote {OUT_DIR}")
    for family in FAMILIES:
        sub = [row for row in summary if row["family"] == family]
        print(f"{family}: {dict(Counter(row['group_class'] for row in sub))}")


if __name__ == "__main__":
    main()
