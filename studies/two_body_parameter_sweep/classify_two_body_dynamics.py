from __future__ import annotations

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
STUDY = ROOT / "studies" / "two_body_parameter_sweep"
MANIFEST = STUDY / "two_body_parameter_sweep_manifest.csv"
OUT_CSV = STUDY / "two_body_dynamics_classification.csv"
OUT_SUMMARY = STUDY / "two_body_dynamics_classification_summary.csv"
OUT_PNG = STUDY / "two_body_dynamics_classification.png"

DEFAULT_TRANSIENT_FRACTION = 0.5
RECURRENCE_DIM = 3
RECURRENCE_TAU = 10
RECURRENCE_RATE = 0.03
RECURRENCE_THEILER = 50
MIN_DIAG_LEN = 4
TOP_SPECTRAL_PEAKS = 12
DIVERGENCE_HORIZON = 35

CLASS_ORDER = [
    "periodic",
    "quasi-periodic",
    "chaotic",
    "chaotic-candidate",
    "ambiguous",
    "mixed",
    "incomplete",
]
CLASS_COLOR = {
    "periodic": "#2ca02c",
    "quasi-periodic": "#1f77b4",
    "chaotic": "#d62728",
    "chaotic-candidate": "#ff7f0e",
    "ambiguous": "#ffbf00",
    "mixed": "#7f7f7f",
    "incomplete": "#9e9e9e",
}
CLASS_CODE = {
    "periodic": "P",
    "quasi-periodic": "Q",
    "chaotic": "C",
    "chaotic-candidate": "c",
    "ambiguous": "A",
    "incomplete": "I",
    "mixed": "M",
}


def base_class(value: object) -> str:
    text = str(value)
    while text.startswith("mostly_"):
        text = text.removeprefix("mostly_")
    return text


def load_manifest() -> list[dict[str, str]]:
    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def output_complete(row: dict[str, str]) -> bool:
    output = Path(row["output"])
    if not output.exists():
        return False
    expected_rows = int(round(float(row["tend"]) / float(row["dt"]))) + 1
    with output.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f) >= expected_rows + 1


def load_output(row: dict[str, str]) -> np.ndarray:
    data = np.genfromtxt(Path(row["output"]), delimiter=",", names=True)
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    return data


def colvec(data: np.ndarray, prefix: str, body: int) -> np.ndarray:
    return np.column_stack(
        [data[f"{prefix}_x_{body}"], data[f"{prefix}_y_{body}"], data[f"{prefix}_z_{body}"]]
    )


def marker(data: np.ndarray, body: int) -> np.ndarray:
    return np.column_stack([data[f"ofix1_{body}"], data[f"ofix2_{body}"], data[f"ofix3_{body}"]])


def angular_velocity(data: np.ndarray, body: int) -> np.ndarray:
    return np.column_stack([data[f"w1_{body}"], data[f"w2_{body}"], data[f"w3_{body}"]])


def position(data: np.ndarray, body: int) -> np.ndarray:
    return np.column_stack([data[f"px_{body}"], data[f"py_{body}"], data[f"pz_{body}"]])


def body_state(data: np.ndarray, body: int) -> np.ndarray:
    return np.column_stack([marker(data, body), angular_velocity(data, body)])


def combined_state(data: np.ndarray) -> np.ndarray:
    p1 = position(data, 1)
    p2 = position(data, 2)
    rel = p2 - p1
    sep = np.linalg.norm(rel, axis=1)[:, None]
    return np.column_stack([marker(data, 1), marker(data, 2), rel, sep])


def standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    std[std < 1.0e-12] = 1.0
    return (values - mean) / std


def delay_embed(series: np.ndarray, dim: int = RECURRENCE_DIM, tau: int = RECURRENCE_TAU) -> np.ndarray:
    series = standardize(series)
    n = len(series) - (dim - 1) * tau
    if n <= 2:
        raise ValueError("not enough samples for delay embedding")
    return np.hstack([series[i * tau : i * tau + n] for i in range(dim)])


def pairwise_distances(embedded: np.ndarray) -> np.ndarray:
    diff = embedded[:, None, :] - embedded[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def recurrence_from_distances(
    dist: np.ndarray,
    target_rate: float = RECURRENCE_RATE,
    theiler: int = RECURRENCE_THEILER,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    n = dist.shape[0]
    idx = np.arange(n)
    mask = np.abs(idx[:, None] - idx[None, :]) > theiler
    finite = dist[mask & np.isfinite(dist)]
    if finite.size == 0:
        raise ValueError("no finite recurrence distances")
    eps = float(np.quantile(finite, target_rate))
    rec = (dist <= eps) & mask
    achieved_rr = float(rec[mask].mean()) if np.any(mask) else float("nan")
    return rec, mask, eps, achieved_rr


def diagonal_lengths(rec: np.ndarray, min_len: int = MIN_DIAG_LEN) -> list[int]:
    lengths: list[int] = []
    n = rec.shape[0]
    for offset in range(-n + 1, n):
        diag = np.diagonal(rec, offset=offset)
        run = 0
        for value in diag:
            if value:
                run += 1
            elif run:
                if run >= min_len:
                    lengths.append(run)
                run = 0
        if run >= min_len:
            lengths.append(run)
    return lengths


def entropy_from_counts(lengths: list[int]) -> float:
    if not lengths:
        return 0.0
    counts = np.array(list(Counter(lengths).values()), dtype=float)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs)))


def rqa_metrics(series: np.ndarray) -> dict[str, float]:
    embedded = delay_embed(series)
    dist = pairwise_distances(embedded)
    rec, mask, eps, rr = recurrence_from_distances(dist)
    lengths = diagonal_lengths(rec)
    recurrent_points = int(rec.sum())
    diag_points = int(sum(lengths))
    determinism = diag_points / recurrent_points if recurrent_points else 0.0
    lmax = max(lengths) if lengths else 0
    mean_diag = float(np.mean(lengths)) if lengths else 0.0
    diag_entropy = entropy_from_counts(lengths)

    masked = dist.copy()
    masked[~mask] = np.inf
    nearest = np.min(masked, axis=1)
    nearest_idx = np.argmin(masked, axis=1)
    return_lags = np.abs(nearest_idx - np.arange(len(nearest_idx)))
    finite = np.isfinite(nearest) & (return_lags > RECURRENCE_THEILER)
    nearest_distance = float(np.median(nearest[finite])) if np.any(finite) else float("nan")
    lag_values = return_lags[finite].astype(float)
    return_cv = (
        float(np.std(lag_values) / np.mean(lag_values))
        if lag_values.size > 1 and np.mean(lag_values) > 0.0
        else float("nan")
    )
    divergence = divergence_metrics(dist, mask)

    return {
        "recurrence_rate": rr,
        "recurrence_epsilon": eps,
        "determinism": float(determinism),
        "lmax_fraction": float(lmax / rec.shape[0]),
        "mean_diag": mean_diag,
        "diag_entropy": diag_entropy,
        "nearest_distance": nearest_distance,
        "return_cv": return_cv,
        **divergence,
    }


def divergence_metrics(dist: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    masked = dist.copy()
    masked[~mask] = np.inf
    pairs_i = np.arange(masked.shape[0])
    pairs_j = np.argmin(masked, axis=1)
    d0 = masked[pairs_i, pairs_j]
    valid = np.isfinite(d0) & (d0 > 1.0e-12)
    pairs_i = pairs_i[valid]
    pairs_j = pairs_j[valid]
    d0 = d0[valid]
    if len(d0) < 10:
        return {"divergence_slope": float("nan"), "divergence_gain": float("nan")}

    max_h = min(DIVERGENCE_HORIZON, dist.shape[0] - 1)
    horizons: list[int] = []
    log_growth: list[float] = []
    for h in range(1, max_h + 1):
        ok = (pairs_i + h < dist.shape[0]) & (pairs_j + h < dist.shape[0])
        if np.count_nonzero(ok) < 10:
            continue
        dh = dist[pairs_i[ok] + h, pairs_j[ok] + h]
        finite = np.isfinite(dh) & (dh > 1.0e-12)
        if np.count_nonzero(finite) < 10:
            continue
        horizons.append(h)
        log_growth.append(float(np.median(np.log(dh[finite] / d0[ok][finite]))))

    if len(horizons) < 5:
        return {"divergence_slope": float("nan"), "divergence_gain": float("nan")}
    fit_n = min(18, len(horizons))
    slope = float(np.polyfit(np.asarray(horizons[:fit_n], dtype=float), np.asarray(log_growth[:fit_n]), 1)[0])
    gain = float(max(log_growth[:fit_n]) - log_growth[0])
    return {"divergence_slope": slope, "divergence_gain": gain}


def spectral_metrics(series: np.ndarray) -> dict[str, float]:
    values = standardize(series)
    values = values - np.linspace(values[0], values[-1], len(values))
    window = np.hanning(len(values))[:, None]
    spectra = np.fft.rfft(values * window, axis=0)
    power = np.sum(np.abs(spectra) ** 2, axis=1)
    if power.size <= 2:
        return {
            "spectral_entropy": float("nan"),
            "peak_fraction": float("nan"),
            "dominant_fraction": float("nan"),
            "harmonicity": float("nan"),
        }
    power = power[1:]
    total = float(power.sum())
    if total <= 0.0 or not np.isfinite(total):
        return {
            "spectral_entropy": 0.0,
            "peak_fraction": 1.0,
            "dominant_fraction": 1.0,
            "harmonicity": 1.0,
        }
    probs = power / total
    entropy = float(-np.sum(probs * np.log(probs + 1.0e-300)) / math.log(len(probs)))
    top = np.partition(power, -min(TOP_SPECTRAL_PEAKS, len(power)))[-TOP_SPECTRAL_PEAKS:]
    peak_fraction = float(top.sum() / total)
    dominant_fraction = float(power.max() / total)
    harmonicity = harmonicity_score(power)
    return {
        "spectral_entropy": entropy,
        "peak_fraction": peak_fraction,
        "dominant_fraction": dominant_fraction,
        "harmonicity": harmonicity,
    }


def harmonicity_score(power: np.ndarray) -> float:
    n_peaks = min(TOP_SPECTRAL_PEAKS, len(power))
    if n_peaks == 0:
        return 1.0
    peak_bins = np.argpartition(power, -n_peaks)[-n_peaks:] + 1
    peak_power = power[peak_bins - 1]
    order = np.argsort(peak_power)[::-1]
    peak_bins = peak_bins[order]
    peak_power = peak_power[order]
    top_power = float(peak_power.sum())
    if top_power <= 0.0:
        return 1.0

    best = 0.0
    candidate_fundamentals = sorted(set(int(b) for b in peak_bins[: min(6, len(peak_bins))]))
    for fundamental in candidate_fundamentals:
        harmonic_power = 0.0
        for bin_idx, pwr in zip(peak_bins, peak_power):
            ratio = bin_idx / fundamental
            nearest = max(1, round(ratio))
            bin_error = abs(bin_idx - nearest * fundamental)
            tolerance = max(1.5, 0.035 * bin_idx)
            if bin_error <= tolerance:
                harmonic_power += float(pwr)
        best = max(best, harmonic_power / top_power)
    return float(best)


def classify(metrics: dict[str, float]) -> str:
    det = metrics["determinism"]
    lmax = metrics["lmax_fraction"]
    sent = metrics["spectral_entropy"]
    peak = metrics["peak_fraction"]
    rcv = metrics["return_cv"]
    div = metrics["divergence_slope"]
    gain = metrics["divergence_gain"]
    harmonicity = metrics["harmonicity"]
    dominant = metrics["dominant_fraction"]

    strong_divergence = div > 0.018 and gain > 0.25
    weak_divergence = div > 0.010 and gain > 0.18

    if strong_divergence and (sent > 0.34 or peak < 0.86 or det < 0.90):
        return "chaotic"
    if weak_divergence and (sent > 0.42 or peak < 0.78 or det < 0.82):
        return "chaotic-candidate"

    if det > 0.90 and lmax > 0.35 and sent < 0.30 and peak > 0.82 and harmonicity > 0.78:
        return "periodic"
    if det > 0.80 and lmax > 0.20 and sent < 0.50 and peak > 0.58:
        if harmonicity > 0.82 and dominant > 0.28 and sent < 0.38:
            return "periodic"
        return "quasi-periodic"
    if weak_divergence:
        return "chaotic-candidate"
    if det < 0.78 and lmax < 0.22 and sent > 0.42 and peak < 0.68:
        return "chaotic"
    if sent > 0.62 and peak < 0.50:
        return "chaotic"
    return "ambiguous"


def analyze_series(series: np.ndarray) -> dict[str, object]:
    metrics = {**rqa_metrics(series), **spectral_metrics(series)}
    metrics["class"] = classify(metrics)
    return metrics


def consensus(classes: list[str]) -> str:
    useful = [base_class(c) for c in classes if base_class(c) != "incomplete"]
    if not useful:
        return "incomplete"
    counts = Counter(useful)
    top, n_top = counts.most_common(1)[0]
    if n_top == len(useful):
        return top
    if counts["chaotic"] > 0 and counts["periodic"] > 0:
        return "mixed"
    if n_top >= math.ceil(len(useful) / 2):
        return top
    return "mixed"


def analyze_run(row: dict[str, str], transient_fraction: float) -> dict[str, object]:
    base: dict[str, object] = {
        "name": row["name"],
        "shape_name": row["shape_name"],
        "rho": row["rho"],
        "energy_ratio": row["energy_ratio"],
        "separation": row["separation"],
        "repeat": row["repeat"],
    }
    if not output_complete(row):
        return {
            **base,
            "status": "incomplete",
            "class_body1": "incomplete",
            "class_body2": "incomplete",
            "class_combined": "incomplete",
            "class_consensus": "incomplete",
        }

    data = load_output(row)
    names = data.dtype.names or ()
    required = {"time", "ofix1_1", "ofix1_2", "w1_1", "w1_2", "px_1", "px_2"}
    if not required.issubset(names):
        raise KeyError(f"{row['name']} is missing required columns: {sorted(required - set(names))}")
    if not np.all(np.isfinite(data["time"])):
        return {**base, "status": "nonfinite", "class_consensus": "incomplete"}

    t = data["time"]
    keep = t >= (float(row["tend"]) * transient_fraction)
    data = data[keep]
    analyses = {
        "body1": analyze_series(body_state(data, 1)),
        "body2": analyze_series(body_state(data, 2)),
        "combined": analyze_series(combined_state(data)),
    }
    classes = [str(analyses[key]["class"]) for key in ("body1", "body2", "combined")]
    out: dict[str, object] = {
        **base,
        "status": "ok",
        "class_body1": classes[0],
        "class_body2": classes[1],
        "class_combined": classes[2],
        "class_consensus": consensus(classes),
    }
    for label, metrics in analyses.items():
        for key, value in metrics.items():
            out[f"{label}_{key}"] = value
    return out


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["shape_name"]),
            str(row["rho"]),
            str(row["energy_ratio"]),
            str(row["separation"]),
        )
        grouped[key].append(row)

    out = []
    for (shape, rho, energy, sep), group in sorted(grouped.items()):
        classes = [str(row.get("class_consensus", "incomplete")) for row in group]
        counts = Counter(classes)
        out.append(
            {
                "shape_name": shape,
                "rho": rho,
                "energy_ratio": energy,
                "separation": sep,
                "n_repeats": len(group),
                "n_ok": sum(1 for row in group if row.get("status") == "ok"),
                "repeat_classes": ";".join(classes),
                "group_class": consensus(classes),
                **{f"n_{name}": counts.get(name, 0) for name in CLASS_ORDER},
                "n_mixed": counts.get("mixed", 0),
            }
        )
    return out


def plot_summary(summary_rows: list[dict[str, object]], out_png: Path, title_suffix: str) -> None:
    shapes = sorted({str(row["shape_name"]) for row in summary_rows})
    fig, axes = plt.subplots(len(shapes), 1, figsize=(13, 4.8 * len(shapes)), constrained_layout=True)
    if len(shapes) == 1:
        axes = [axes]

    for ax, shape in zip(axes, shapes):
        rows = [row for row in summary_rows if row["shape_name"] == shape]
        ys = sorted({(float(row["rho"]), float(row["separation"])) for row in rows}, reverse=True)
        xs = sorted({float(row["energy_ratio"]) for row in rows})
        y_index = {key: i for i, key in enumerate(ys)}
        x_index = {key: i for i, key in enumerate(xs)}

        ax.set_title(shape)
        ax.set_xticks(range(len(xs)), [f"E={x:g}" for x in xs])
        ax.set_yticks(range(len(ys)), [f"rho={rho:g}, sep={sep:g}" for rho, sep in ys])
        ax.set_xlim(-0.5, len(xs) - 0.5)
        ax.set_ylim(len(ys) - 0.5, -0.5)
        ax.grid(color="#dddddd", lw=0.8)

        for row in rows:
            x = x_index[float(row["energy_ratio"])]
            y = y_index[(float(row["rho"]), float(row["separation"]))]
            cls = str(row["group_class"])
            color = CLASS_COLOR.get(cls.replace("mostly_", ""), "#7f7f7f")
            ax.scatter(x, y, s=900, marker="s", color=color, edgecolor="black", linewidth=0.8)
            label = "/".join(CLASS_CODE.get(base_class(c), "?") for c in str(row["repeat_classes"]).split(";"))
            text_color = "black" if base_class(cls) in {"ambiguous", "incomplete"} else "white"
            ax.text(x, y, label, ha="center", va="center", fontsize=10, color=text_color, weight="bold")

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label=f"{CLASS_CODE.get(cls, '?')} = {cls}",
            markerfacecolor=color,
            markersize=12,
        )
        for cls, color in CLASS_COLOR.items()
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    fig.suptitle(f"Two-body dynamics classification from {title_suffix} trajectories", fontsize=14)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify two-body runs as periodic/quasi/chaotic.")
    parser.add_argument("--limit", type=int, default=None, help="analyze only the first N manifest rows")
    parser.add_argument(
        "--transient-fraction",
        type=float,
        default=DEFAULT_TRANSIENT_FRACTION,
        help="discard this fraction of tend before analysis; use 0 for full-run classification",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="append a suffix before file extensions, e.g. _full_run",
    )
    args = parser.parse_args()

    rows = load_manifest()
    if args.limit is not None:
        rows = rows[: args.limit]

    if not 0.0 <= args.transient_fraction < 1.0:
        raise SystemExit("--transient-fraction must be in [0, 1)")

    out_csv = OUT_CSV.with_name(f"{OUT_CSV.stem}{args.suffix}{OUT_CSV.suffix}")
    out_summary = OUT_SUMMARY.with_name(f"{OUT_SUMMARY.stem}{args.suffix}{OUT_SUMMARY.suffix}")
    out_png = OUT_PNG.with_name(f"{OUT_PNG.stem}{args.suffix}{OUT_PNG.suffix}")
    title_suffix = "full-run" if args.transient_fraction == 0.0 else f"post-transient ({args.transient_fraction:g} tend)"

    results = []
    for i, row in enumerate(rows, start=1):
        print(f"[{i:03d}/{len(rows):03d}] {row['name']}", flush=True)
        results.append(analyze_run(row, args.transient_fraction))

    summary = summarize(results)
    write_rows(out_csv, results)
    write_rows(out_summary, summary)
    plot_summary(summary, out_png, title_suffix)
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_summary}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
