import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "studies" / "two_body_paper_sweeps"
OUT = STUDY / "analysis_overview"

ENERGIES = [0.2, 0.55, 1.5, 4.0, 11.0, 30.0]
SHAPE_LABELS = {
    "spheroid_1_0p7_0p7": "spheroid 1:0.7:0.7",
    "ellipsoid_1_0p8_0p6": "triaxial 1:0.8:0.6",
}
MODE_LABELS = {
    "marker": "marker",
    "quaternion": "full quaternion",
    "axis": "axisymmetric axis",
}
RUNS = [
    ("spheroid_1_0p7_0p7", "marker", STUDY / "classification_spheroid_1_0p7_0p7"),
    ("spheroid_1_0p7_0p7", "quaternion", STUDY / "classification_spheroid_1_0p7_0p7_quaternion"),
    ("spheroid_1_0p7_0p7", "axis", STUDY / "classification_spheroid_1_0p7_0p7_axis"),
    ("ellipsoid_1_0p8_0p6", "marker", STUDY / "classification_ellipsoid_1_0p8_0p6"),
    ("ellipsoid_1_0p8_0p6", "quaternion", STUDY / "classification_ellipsoid_1_0p8_0p6_quaternion"),
]

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
CLASS_LABELS = [
    "P = periodic",
    "Q = quasi-periodic",
    "S = sensitive-regular",
    "R = complex-regular",
    "M = mixed",
    "C = chaotic-like",
    "I = incomplete",
]


def read_summary(path):
    with (path / "two_body_dynamics_classification_summary.csv").open(
        "r", encoding="utf-8", newline=""
    ) as f:
        return list(csv.DictReader(f))


def group_axes(rows):
    xs = sorted({float(row["energy_ratio"]) for row in rows})
    ys = sorted({(float(row["rho"]), float(row["separation"])) for row in rows}, reverse=True)
    return xs, ys


def class_matrix(rows, xs, ys):
    mat = np.full((len(ys), len(xs)), np.nan)
    labels = [["" for _ in xs] for _ in ys]
    x_index = {x: i for i, x in enumerate(xs)}
    y_index = {y: i for i, y in enumerate(ys)}
    for row in rows:
        x = float(row["energy_ratio"])
        y = (float(row["rho"]), float(row["separation"]))
        cls = row["group_class"]
        mat[y_index[y], x_index[x]] = CLASS_ORDER.get(cls, np.nan)
        labels[y_index[y]][x_index[x]] = CLASS_CODE.get(cls, "?")
    return mat, labels


def score_matrix(rows, xs, ys):
    mat = np.full((len(ys), len(xs)), np.nan)
    x_index = {x: i for i, x in enumerate(xs)}
    y_index = {y: i for i, y in enumerate(ys)}
    for row in rows:
        x = float(row["energy_ratio"])
        y = (float(row["rho"]), float(row["separation"]))
        value = row.get("body_broadband_chaos_score_mean", "")
        if value:
            mat[y_index[y], x_index[x]] = float(value)
    return mat


def annotate_classes(ax, labels):
    for i, row in enumerate(labels):
        for j, text in enumerate(row):
            if text:
                ax.text(j, i, text, ha="center", va="center", color="white", fontsize=7, fontweight="bold")


def setup_axis(ax, xs, ys, title, show_y):
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([f"{x:g}" for x in xs], rotation=45, ha="right")
    ax.set_xlabel("energy ratio")
    ax.set_yticks(range(len(ys)))
    if show_y:
        ax.set_yticklabels([f"rho={rho:g}, sep={sep:g}" for rho, sep in ys], fontsize=7)
    else:
        ax.set_yticklabels([])


def plot_class_comparison(data):
    fig, axes = plt.subplots(2, 3, figsize=(15, 11))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.13, wspace=0.03, hspace=0.3)
    cmap = matplotlib.colors.ListedColormap(CLASS_COLORS)
    norm = matplotlib.colors.BoundaryNorm(np.arange(-0.5, len(CLASS_COLORS) + 0.5), cmap.N)
    by_shape = {
        "spheroid_1_0p7_0p7": ["marker", "quaternion", "axis"],
        "ellipsoid_1_0p8_0p6": ["marker", "quaternion"],
    }
    for row_idx, (shape, modes) in enumerate(by_shape.items()):
        ref_rows = data[(shape, modes[0])]
        xs, ys = group_axes(ref_rows)
        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            if col_idx >= len(modes):
                ax.axis("off")
                continue
            mode = modes[col_idx]
            mat, labels = class_matrix(data[(shape, mode)], xs, ys)
            ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
            annotate_classes(ax, labels)
            setup_axis(ax, xs, ys, f"{SHAPE_LABELS[shape]}\n{MODE_LABELS[mode]}", col_idx == 0)
    handles = [Patch(color=color, label=label) for color, label in zip(CLASS_COLORS, CLASS_LABELS)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=4, frameon=True)
    fig.suptitle("Two-body recurrence classification by orientation state", fontsize=14)
    fig.savefig(OUT / "orientation_recurrence_mode_class_comparison.png", dpi=220)
    plt.close(fig)


def plot_score_comparison(data):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    by_shape = {
        "spheroid_1_0p7_0p7": ["marker", "quaternion", "axis"],
        "ellipsoid_1_0p8_0p6": ["marker", "quaternion"],
    }
    image = None
    for row_idx, (shape, modes) in enumerate(by_shape.items()):
        ref_rows = data[(shape, modes[0])]
        xs, ys = group_axes(ref_rows)
        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            if col_idx >= len(modes):
                ax.axis("off")
                continue
            mode = modes[col_idx]
            mat = score_matrix(data[(shape, mode)], xs, ys)
            image = ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
            setup_axis(ax, xs, ys, f"{SHAPE_LABELS[shape]}\n{MODE_LABELS[mode]}", col_idx == 0)
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.82, label="broadband recurrence score")
    fig.suptitle("Two-body recurrence broadband score by orientation state", fontsize=14)
    fig.savefig(OUT / "orientation_recurrence_mode_broadband_comparison.png", dpi=220)
    plt.close(fig)


def write_change_table(data):
    rows = []
    for shape in SHAPE_LABELS:
        marker = {(float(r["rho"]), float(r["energy_ratio"]), float(r["separation"])): r for r in data[(shape, "marker")]}
        modes = ["quaternion"]
        if shape == "spheroid_1_0p7_0p7":
            modes.append("axis")
        for mode in modes:
            lookup = {(float(r["rho"]), float(r["energy_ratio"]), float(r["separation"])): r for r in data[(shape, mode)]}
            for key in sorted(marker):
                base = marker[key]
                other = lookup[key]
                rows.append(
                    {
                        "shape_name": shape,
                        "comparison": f"marker_vs_{mode}",
                        "rho": key[0],
                        "energy_ratio": key[1],
                        "separation": key[2],
                        "marker_class": base["group_class"],
                        f"{mode}_class": other["group_class"],
                        "class_changed": base["group_class"] != other["group_class"],
                        "marker_broadband_score": base["body_broadband_chaos_score_mean"],
                        f"{mode}_broadband_score": other["body_broadband_chaos_score_mean"],
                    }
                )
    path = OUT / "orientation_recurrence_mode_class_changes.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {(shape, mode): read_summary(path) for shape, mode, path in RUNS}
    plot_class_comparison(data)
    plot_score_comparison(data)
    write_change_table(data)


if __name__ == "__main__":
    main()
