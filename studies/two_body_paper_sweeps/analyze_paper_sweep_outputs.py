import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "studies" / "two_body_paper_sweeps"
SHAPES = ["spheroid_1_0p7_0p7", "ellipsoid_1_0p8_0p6"]
ENERGIES = [0.2, 0.55, 1.5, 4.0, 11.0, 30.0]
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


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.12g}"


def load_dat(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    return data


def vec(data, prefix, body):
    if prefix == "p":
        return np.column_stack([data[f"px_{body}"], data[f"py_{body}"], data[f"pz_{body}"]])
    return np.column_stack(
        [data[f"{prefix}_x_{body}"], data[f"{prefix}_y_{body}"], data[f"{prefix}_z_{body}"]]
    )


def max_norm_drift(data, prefix, body_count):
    cols = []
    for body in range(1, body_count + 1):
        for axis in "xyz":
            name = f"{prefix}_{axis}_{body}"
            if name not in data.dtype.names:
                return float("nan")
        cols.append(vec(data, prefix, body))
    total = np.sum(cols, axis=0)
    base = total[0]
    scale = max(np.linalg.norm(base), 1.0)
    return float(np.max(np.linalg.norm(total - base, axis=1)) / scale)


def output_metrics(row):
    path = ROOT / row["output"]
    expected_rows = int(round(float(row["tend"]) / float(row["dt"]))) + 1
    if not path.exists():
        return {
            "status": "missing",
            "n_rows": 0,
            "expected_rows": expected_rows,
        }
    data = load_dat(path)
    n_rows = int(len(data))
    finite_time = np.isfinite(data["time"])
    complete = n_rows >= expected_rows and bool(np.all(finite_time))
    status = "complete" if complete else "short"

    metrics = {
        "status": status,
        "n_rows": n_rows,
        "expected_rows": expected_rows,
        "final_time": fmt(float(data["time"][-1])) if n_rows else "",
    }
    if n_rows < 2:
        return metrics

    ke0 = float(data["ke_total"][0])
    ke = data["ke_total"]
    if math.isfinite(ke0) and abs(ke0) > 0:
        metrics["max_abs_ke_drift_pct"] = fmt(float(np.nanmax(np.abs((ke - ke0) / ke0))) * 100.0)
        metrics["final_ke_drift_pct"] = fmt(float((ke[-1] - ke0) / ke0) * 100.0)

    p1 = vec(data, "p", 1)
    p2 = vec(data, "p", 2)
    sep = np.linalg.norm(p2 - p1, axis=1)
    metrics["min_separation"] = fmt(float(np.nanmin(sep)))
    metrics["final_separation"] = fmt(float(sep[-1]))

    if "impulse_global_p_drift" in data.dtype.names:
        metrics["max_impulse_global_p_drift"] = fmt(float(np.nanmax(data["impulse_global_p_drift"])))
    if "impulse_global_h_drift" in data.dtype.names:
        metrics["max_impulse_global_h_drift"] = fmt(float(np.nanmax(data["impulse_global_h_drift"])))
    metrics["max_pcon_total_drift_rel"] = fmt(max_norm_drift(data, "pcon", 2))
    metrics["max_hcon_total_drift_rel"] = fmt(max_norm_drift(data, "hcon", 2))
    return metrics


def summarize_runs(manifest_rows):
    rows = []
    for row in manifest_rows:
        out = dict(row)
        out.update(output_metrics(row))
        rows.append(out)
    return rows


def group_key(row):
    return (row["shape_name"], float(row["rho"]), float(row["energy_ratio"]), float(row["separation"]))


def mean(values):
    vals = [float(v) for v in values if v not in ("", None) and math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else float("nan")


def aggregate(run_rows, class_rows):
    classes = {(r["shape_name"], float(r["rho"]), float(r["energy_ratio"]), float(r["separation"])): r for r in class_rows}
    grouped = defaultdict(list)
    for row in run_rows:
        grouped[group_key(row)].append(row)

    out = []
    for key in sorted(grouped):
        group = grouped[key]
        cls = classes.get(key, {})
        complete = [r for r in group if r["status"] == "complete"]
        item = {
            "shape_name": key[0],
            "rho": key[1],
            "energy_ratio": key[2],
            "separation": key[3],
            "n_runs": len(group),
            "n_complete": len(complete),
            "n_short_or_missing": len(group) - len(complete),
            "class": cls.get("group_class", ""),
            "regime": cls.get("regime", ""),
            "broadband_score": cls.get("body_broadband_chaos_score_mean", ""),
        }
        for metric in [
            "max_abs_ke_drift_pct",
            "max_impulse_global_p_drift",
            "max_impulse_global_h_drift",
            "max_pcon_total_drift_rel",
            "max_hcon_total_drift_rel",
            "min_separation",
        ]:
            item[f"mean_{metric}"] = fmt(mean([r.get(metric, "") for r in complete]))
        out.append(item)
    return out


def matrix(rows, shape, metric):
    sub = [r for r in rows if r["shape_name"] == shape]
    xs = sorted({float(r["energy_ratio"]) for r in sub})
    ys = sorted({(float(r["rho"]), float(r["separation"])) for r in sub}, reverse=True)
    x_index = {x: i for i, x in enumerate(xs)}
    y_index = {y: i for i, y in enumerate(ys)}
    mat = np.full((len(ys), len(xs)), np.nan)
    for row in sub:
        value = row.get(metric, "")
        try:
            mat[y_index[(float(row["rho"]), float(row["separation"]))], x_index[float(row["energy_ratio"])]] = float(value)
        except (TypeError, ValueError):
            pass
    return mat, xs, ys


def class_matrix(rows, shape):
    sub = [r for r in rows if r["shape_name"] == shape]
    xs = sorted({float(r["energy_ratio"]) for r in sub})
    ys = sorted({(float(r["rho"]), float(r["separation"])) for r in sub}, reverse=True)
    mat = np.full((len(ys), len(xs)), np.nan)
    labels = [["" for _ in xs] for _ in ys]
    x_index = {x: i for i, x in enumerate(xs)}
    y_index = {y: i for i, y in enumerate(ys)}
    for row in sub:
        yi = y_index[(float(row["rho"]), float(row["separation"]))]
        xi = x_index[float(row["energy_ratio"])]
        cls = row.get("class", "")
        mat[yi, xi] = CLASS_ORDER.get(cls, np.nan)
        labels[yi][xi] = CLASS_CODE.get(cls, "?")
    return mat, labels, xs, ys


def plot_metric(rows, metric, title, out, cmap="viridis", log=False):
    fig, axes = plt.subplots(1, len(SHAPES), figsize=(12.5, 7.0), constrained_layout=True)
    for ax, shape in zip(axes, SHAPES):
        mat, xs, ys = matrix(rows, shape, metric)
        plot_mat = np.log10(mat) if log else mat
        im = ax.imshow(plot_mat, aspect="auto", cmap=cmap)
        ax.set_title(shape)
        ax.set_xticks(range(len(xs)), [f"{x:g}" for x in xs])
        ax.set_yticks(range(len(ys)), [f"rho={rho:g}, sep={sep:g}" for rho, sep in ys], fontsize=7)
        ax.set_xlabel("E")
        ax.set_ylabel("case")
        for yi in range(mat.shape[0]):
            for xi in range(mat.shape[1]):
                value = mat[yi, xi]
                label = "" if not math.isfinite(value) else f"{value:.2g}"
                ax.text(xi, yi, label, ha="center", va="center", fontsize=6, color="white")
        fig.colorbar(im, ax=ax, shrink=0.85, label=("log10 " if log else "") + title)
    fig.suptitle(title)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_classes(rows, out):
    from matplotlib.colors import ListedColormap, BoundaryNorm

    cmap = ListedColormap(CLASS_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, 7.5, 1.0), cmap.N)
    fig, axes = plt.subplots(1, len(SHAPES), figsize=(12.5, 7.0), constrained_layout=True)
    for ax, shape in zip(axes, SHAPES):
        mat, labels, xs, ys = class_matrix(rows, shape)
        ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm)
        ax.set_title(shape)
        ax.set_xticks(range(len(xs)), [f"{x:g}" for x in xs])
        ax.set_yticks(range(len(ys)), [f"rho={rho:g}, sep={sep:g}" for rho, sep in ys], fontsize=7)
        ax.set_xlabel("E")
        ax.set_ylabel("case")
        for yi in range(mat.shape[0]):
            for xi in range(mat.shape[1]):
                ax.text(xi, yi, labels[yi][xi], ha="center", va="center", fontsize=7, color="white")
    fig.suptitle("Recurrence/spectral behaviour class")
    fig.savefig(out, dpi=180)
    plt.close(fig)


def class_slice_matrix(rows, shape, fixed_key, fixed_value, x_key, y_key, x_reverse=False, y_reverse=True):
    sub = [
        r
        for r in rows
        if r["shape_name"] == shape and math.isclose(float(r[fixed_key]), float(fixed_value), rel_tol=0.0, abs_tol=1.0e-12)
    ]
    xs = sorted({float(r[x_key]) for r in sub}, reverse=x_reverse)
    ys = sorted({float(r[y_key]) for r in sub}, reverse=y_reverse)
    mat = np.full((len(ys), len(xs)), np.nan)
    labels = [["" for _ in xs] for _ in ys]
    x_index = {x: i for i, x in enumerate(xs)}
    y_index = {y: i for i, y in enumerate(ys)}
    for row in sub:
        yi = y_index[float(row[y_key])]
        xi = x_index[float(row[x_key])]
        cls = row.get("class", "")
        mat[yi, xi] = CLASS_ORDER.get(cls, np.nan)
        labels[yi][xi] = CLASS_CODE.get(cls, "?")
    return mat, labels, xs, ys


def plot_class_slice_family(rows, fixed_key, x_key, y_key, title, out, x_label, y_label):
    from matplotlib.colors import BoundaryNorm, ListedColormap

    fixed_values = sorted({float(r[fixed_key]) for r in rows})
    if fixed_key == "rho":
        fixed_values = sorted(fixed_values, reverse=True)
    cmap = ListedColormap(CLASS_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, 7.5, 1.0), cmap.N)
    fig, axes = plt.subplots(
        len(SHAPES),
        len(fixed_values),
        figsize=(2.55 * len(fixed_values), 4.2 * len(SHAPES)),
        constrained_layout=True,
        squeeze=False,
    )
    for row_i, shape in enumerate(SHAPES):
        for col_i, fixed_value in enumerate(fixed_values):
            ax = axes[row_i][col_i]
            mat, labels, xs, ys = class_slice_matrix(rows, shape, fixed_key, fixed_value, x_key, y_key)
            ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm)
            ax.set_title(f"{fixed_key}={fixed_value:g}", fontsize=10)
            ax.set_xticks(range(len(xs)), [f"{x:g}" for x in xs], rotation=35, ha="right")
            ax.set_yticks(range(len(ys)), [f"{y:g}" for y in ys])
            ax.set_xlabel(x_label)
            if col_i == 0:
                ax.set_ylabel(f"{shape}\n{y_label}")
            else:
                ax.set_ylabel(y_label)
            for yi in range(mat.shape[0]):
                for xi in range(mat.shape[1]):
                    ax.text(xi, yi, labels[yi][xi], ha="center", va="center", fontsize=8, color="white")
    fig.suptitle(title)
    fig.savefig(out, dpi=190)
    plt.close(fig)


def print_summary(aggregate_rows):
    for shape in SHAPES:
        sub = [r for r in aggregate_rows if r["shape_name"] == shape]
        classes = Counter(r["class"] for r in sub)
        complete_groups = sum(1 for r in sub if int(r["n_short_or_missing"]) == 0)
        print(f"{shape}: {complete_groups}/{len(sub)} groups have both repeats complete")
        print("  classes:", dict(classes))
        low = [r for r in sub if float(r["rho"]) <= 0.1]
        print("  low-density classes:", dict(Counter(r["class"] for r in low)))
        worst_ke = sorted(
            [r for r in sub if r["mean_max_abs_ke_drift_pct"]],
            key=lambda r: float(r["mean_max_abs_ke_drift_pct"]),
            reverse=True,
        )[:5]
        print("  worst mean KE drift:")
        for r in worst_ke:
            print(
                "   "
                f"rho={float(r['rho']):g} E={float(r['energy_ratio']):g} sep={float(r['separation']):g} "
                f"class={r['class']} KE={float(r['mean_max_abs_ke_drift_pct']):.3g}% "
                f"min_sep={r['mean_min_separation']}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, default=STUDY)
    args = parser.parse_args()
    study = args.study if args.study.is_absolute() else ROOT / args.study

    all_run_rows = []
    class_rows = []
    for shape in SHAPES:
        manifest = read_csv(study / f"manifest_{shape}.csv")
        run_rows = summarize_runs(manifest)
        write_csv(study / f"analysis_{shape}" / "run_metrics.csv", run_rows)
        all_run_rows.extend(run_rows)
        summary = study / f"classification_{shape}" / "two_body_dynamics_classification_summary.csv"
        class_rows.extend(read_csv(summary))

    aggregate_rows = aggregate(all_run_rows, class_rows)
    out_dir = study / "analysis_overview"
    write_csv(out_dir / "group_metrics.csv", aggregate_rows)
    plot_classes(aggregate_rows, out_dir / "behaviour_class_map.png")
    plot_metric(aggregate_rows, "broadband_score", "Broadband recurrence score", out_dir / "broadband_score_map.png", cmap="magma")
    plot_metric(aggregate_rows, "mean_max_abs_ke_drift_pct", "Mean max KE drift (%)", out_dir / "energy_drift_map.png", cmap="viridis", log=True)
    plot_metric(aggregate_rows, "mean_min_separation", "Mean minimum centroid separation", out_dir / "minimum_separation_map.png", cmap="cividis")
    plot_metric(aggregate_rows, "n_short_or_missing", "Short or missing repeats", out_dir / "incomplete_map.png", cmap="Reds")
    plot_class_slice_family(
        aggregate_rows,
        fixed_key="separation",
        x_key="energy_ratio",
        y_key="rho",
        title="Behaviour class slices: density versus energy at fixed separation",
        out=out_dir / "behaviour_slices_const_separation.png",
        x_label="E",
        y_label="rho",
    )
    plot_class_slice_family(
        aggregate_rows,
        fixed_key="energy_ratio",
        x_key="separation",
        y_key="rho",
        title="Behaviour class slices: density versus separation at fixed energy",
        out=out_dir / "behaviour_slices_const_energy.png",
        x_label="separation",
        y_label="rho",
    )
    plot_class_slice_family(
        aggregate_rows,
        fixed_key="rho",
        x_key="energy_ratio",
        y_key="separation",
        title="Behaviour class slices: separation versus energy at fixed density",
        out=out_dir / "behaviour_slices_const_density.png",
        x_label="E",
        y_label="separation",
    )
    print_summary(aggregate_rows)
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
