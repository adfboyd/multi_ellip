"""Compare recurrence behaviour for current KE-ratio density orientation runs.

This post-processes the single-body orientation marker trajectories from the
impulse BEM runs and the analytic exact-added-mass references.  The recurrence
settings intentionally match the more selective settings used in the previous
study: 3-delay embedding, tau=10, target recurrence rate 3%, and Theiler
window 50.  Triaxial cases use the tracked marker point as the orientation
state.  Spheroidal cases use the antipodal symmetry-axis metric, so orientations
with opposite axis sign are treated as the same physical state.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = SCRIPT_DIR / "recurrence" / "current_comparison"

EMBED_DIM = 3
EMBED_TAU = 10
TARGET_RR = 0.03
THEILER = 50
MIN_LINE = 2


@dataclass(frozen=True)
class SourceSpec:
    label: str
    root: Path
    data_file: str
    marker_cols: tuple[str, str, str]
    family: str
    method: str
    metric: str


def case_sort_key(path: Path) -> tuple[float, float, int, str]:
    meta = parse_case_name(path.name)
    return (
        float(meta.get("ratio", math.inf)),
        float(meta.get("rho", math.inf)),
        int(meta.get("run", 9999)),
        path.name,
    )


def parse_case_name(name: str) -> dict[str, float | int | str]:
    meta: dict[str, float | int | str] = {"case": name}
    match = re.search(r"ratio([0-9p]+)_rho([0-9p]+)_run([0-9]+)", name)
    if match:
        meta["ratio"] = float(match.group(1).replace("p", "."))
        meta["rho"] = float(match.group(2).replace("p", "."))
        meta["run"] = int(match.group(3))
    return meta


def read_csv_numeric(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as f:
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


def load_marker(path: Path, marker_cols: tuple[str, str, str]) -> tuple[np.ndarray, np.ndarray]:
    cols = read_csv_numeric(path)
    missing = [col for col in ("time", *marker_cols) if col not in cols]
    if missing:
        raise KeyError(f"{path} is missing columns {missing}")
    marker = np.column_stack([cols[col] for col in marker_cols])
    time = cols["time"]
    finite = np.isfinite(time) & np.all(np.isfinite(marker), axis=1)
    time = time[finite]
    marker = marker[finite]
    if len(time) < (EMBED_DIM - 1) * EMBED_TAU + 10:
        raise ValueError(f"{path} has too few finite samples: {len(time)}")
    return time, marker


def orient_axis(axis: np.ndarray) -> np.ndarray:
    out = np.asarray(axis, dtype=float).copy()
    norm = np.linalg.norm(out, axis=1)
    norm[norm < 1e-14] = 1.0
    out /= norm[:, None]
    for i in range(1, len(out)):
        if float(np.dot(out[i - 1], out[i])) < 0.0:
            out[i] *= -1.0
    return out


def standardize(series: np.ndarray) -> np.ndarray:
    y = np.asarray(series, dtype=float)
    mu = np.nanmean(y, axis=0, keepdims=True)
    sd = np.nanstd(y, axis=0, keepdims=True)
    sd[sd < 1e-14] = 1.0
    return (y - mu) / sd


def delay_embed(series: np.ndarray, dim: int = EMBED_DIM, tau: int = EMBED_TAU) -> np.ndarray:
    y = standardize(series)
    n = len(y) - (dim - 1) * tau
    if n <= 0:
        raise ValueError("series is too short for requested embedding")
    return np.hstack([y[i * tau : i * tau + n] for i in range(dim)])


def delay_embed_axis_for_divergence(series: np.ndarray, dim: int = EMBED_DIM, tau: int = EMBED_TAU) -> np.ndarray:
    axis = orient_axis(series)
    n = len(axis) - (dim - 1) * tau
    if n <= 0:
        raise ValueError("series is too short for requested embedding")
    return np.hstack([axis[i * tau : i * tau + n] for i in range(dim)])


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    gram = points @ points.T
    sq = np.sum(points * points, axis=1)
    dist2 = sq[:, None] + sq[None, :] - 2.0 * gram
    np.maximum(dist2, 0.0, out=dist2)
    return np.sqrt(dist2)


def pairwise_axis_distances(series: np.ndarray, dim: int = EMBED_DIM, tau: int = EMBED_TAU) -> np.ndarray:
    axis = orient_axis(series)
    n = len(axis) - (dim - 1) * tau
    if n <= 0:
        raise ValueError("series is too short for requested embedding")
    dist2 = np.zeros((n, n), dtype=float)
    for k in range(dim):
        block = axis[k * tau : k * tau + n]
        dots = np.clip(block @ block.T, -1.0, 1.0)
        angles = np.arccos(np.abs(dots))
        finite = angles[np.triu(np.ones_like(angles, dtype=bool), k=1)]
        finite = finite[np.isfinite(finite) & (finite > 1e-12)]
        scale = float(np.median(finite)) if finite.size else 1.0
        if not math.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        dist2 += (angles / scale) ** 2
    return np.sqrt(dist2)


def recurrence_from_distances(
    dist: np.ndarray, target_rr: float = TARGET_RR, theiler: int = THEILER
) -> tuple[np.ndarray, float, float]:
    n = dist.shape[0]
    valid = np.ones((n, n), dtype=bool)
    idx = np.arange(n)
    valid[np.abs(idx[:, None] - idx[None, :]) <= theiler] = False
    values = dist[valid]
    epsilon = float(np.quantile(values, target_rr))
    rec = (dist <= epsilon) & valid
    rr = float(rec.sum() / valid.sum())
    return rec, epsilon, rr


def run_lengths(mask: np.ndarray, min_len: int = MIN_LINE) -> list[int]:
    lengths: list[int] = []
    count = 0
    for flag in mask:
        if flag:
            count += 1
        elif count:
            if count >= min_len:
                lengths.append(count)
            count = 0
    if count >= min_len:
        lengths.append(count)
    return lengths


def diagonal_lengths(rec: np.ndarray, min_len: int = MIN_LINE) -> list[int]:
    n = rec.shape[0]
    lengths: list[int] = []
    for offset in range(-(n - 1), n):
        lengths.extend(run_lengths(np.diagonal(rec, offset=offset), min_len))
    return lengths


def vertical_lengths(rec: np.ndarray, min_len: int = MIN_LINE) -> list[int]:
    lengths: list[int] = []
    for col in range(rec.shape[1]):
        lengths.extend(run_lengths(rec[:, col], min_len))
    return lengths


def line_entropy(lengths: list[int]) -> float:
    if not lengths:
        return 0.0
    values, counts = np.unique(np.asarray(lengths), return_counts=True)
    p = counts.astype(float) / counts.sum()
    return float(-np.sum(p * np.log(p + 1e-300)))


def spectral_metrics(marker: np.ndarray) -> dict[str, float]:
    y = standardize(marker)
    y = y - y.mean(axis=0, keepdims=True)
    window = np.hanning(len(y))[:, None]
    power = np.sum(np.abs(np.fft.rfft(y * window, axis=0)) ** 2, axis=1)
    power = power[1:]
    total = float(power.sum())
    if total <= 0.0 or len(power) == 0:
        return {
            "spectral_entropy": 0.0,
            "dominant_power_fraction": 0.0,
            "top5_power_fraction": 0.0,
        }
    p = power / total
    entropy = float(-np.sum(p * np.log(p + 1e-300)) / math.log(len(p)))
    sorted_p = np.sort(p)[::-1]
    return {
        "spectral_entropy": entropy,
        "dominant_power_fraction": float(sorted_p[0]),
        "top5_power_fraction": float(sorted_p[:5].sum()),
    }


def rosenstein_metrics(
    embedded: np.ndarray,
    dist: np.ndarray,
    dt: float,
    theiler: int = THEILER,
    max_horizon: int = 200,
) -> dict[str, float]:
    n = len(embedded)
    if n < 4 * theiler:
        return {
            "neighbor_divergence_slope": 0.0,
            "neighbor_divergence_r2": 0.0,
            "neighbor_divergence_gain": 0.0,
            "neighbor_initial_distance": 0.0,
        }

    valid_dist = dist.copy()
    idx = np.arange(n)
    valid_dist[np.abs(idx[:, None] - idx[None, :]) <= theiler] = np.inf
    nn = np.argmin(valid_dist, axis=1)
    d0 = valid_dist[idx, nn]
    finite0 = np.isfinite(d0) & (d0 > 1e-14)
    if not np.any(finite0):
        return {
            "neighbor_divergence_slope": 0.0,
            "neighbor_divergence_r2": 0.0,
            "neighbor_divergence_gain": 0.0,
            "neighbor_initial_distance": 0.0,
        }

    max_k = min(max_horizon, n // 3)
    horizons: list[int] = []
    log_mean: list[float] = []
    for k in range(1, max_k + 1):
        base = idx + k < n
        neigh = nn + k < n
        mask = finite0 & base & neigh
        if mask.sum() < 20:
            continue
        diff = embedded[idx[mask] + k] - embedded[nn[mask] + k]
        d = np.linalg.norm(diff, axis=1)
        d = d[np.isfinite(d) & (d > 1e-14)]
        if len(d) < 20:
            continue
        horizons.append(k)
        log_mean.append(float(np.mean(np.log(d))))

    if len(horizons) < 12:
        return {
            "neighbor_divergence_slope": 0.0,
            "neighbor_divergence_r2": 0.0,
            "neighbor_divergence_gain": 0.0,
            "neighbor_initial_distance": float(np.nanmedian(d0[finite0])),
        }

    h = np.asarray(horizons, dtype=float)
    y = np.asarray(log_mean, dtype=float)
    fit_mask = (h >= 5) & (h <= min(80, max_k))
    if fit_mask.sum() < 8:
        fit_mask = np.ones_like(h, dtype=bool)
    t = h[fit_mask] * dt
    yf = y[fit_mask]
    slope, intercept = np.polyfit(t, yf, 1)
    pred = slope * t + intercept
    ss_res = float(np.sum((yf - pred) ** 2))
    ss_tot = float(np.sum((yf - np.mean(yf)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    gain = float(yf[-1] - yf[0])
    return {
        "neighbor_divergence_slope": float(slope),
        "neighbor_divergence_r2": float(max(0.0, min(1.0, r2))),
        "neighbor_divergence_gain": gain,
        "neighbor_initial_distance": float(np.nanmedian(d0[finite0])),
    }


def chaos_score(metrics: dict[str, float | str]) -> float:
    lmax = float(metrics["max_diag_fraction"])
    entropy = float(metrics["spectral_entropy"])
    top5 = float(metrics["top5_power_fraction"])
    slope = max(0.0, float(metrics.get("neighbor_divergence_slope", 0.0)))
    gain = max(0.0, float(metrics.get("neighbor_divergence_gain", 0.0)))

    short_diagonal = max(0.0, min(1.0, (0.30 - lmax) / 0.30))
    broad_spectrum = max(0.0, min(1.0, (0.985 - top5) / 0.40))
    entropy_score = max(0.0, min(1.0, entropy / 0.55))
    divergence_score = max(0.0, min(1.0, slope / 0.30))
    gain_score = max(0.0, min(1.0, gain / 1.0))
    return float(
        0.30 * short_diagonal
        + 0.25 * broad_spectrum
        + 0.20 * entropy_score
        + 0.15 * divergence_score
        + 0.10 * gain_score
    )


def broadband_chaos_score(metrics: dict[str, float | str]) -> float:
    lmax = float(metrics["max_diag_fraction"])
    entropy = float(metrics["spectral_entropy"])
    top5 = float(metrics["top5_power_fraction"])
    short_diagonal = max(0.0, min(1.0, (0.30 - lmax) / 0.30))
    broad_spectrum = max(0.0, min(1.0, (0.985 - top5) / 0.40))
    entropy_score = max(0.0, min(1.0, entropy / 0.55))
    return float(0.40 * broad_spectrum + 0.35 * entropy_score + 0.25 * short_diagonal)


def rqa_metrics(marker: np.ndarray, dt: float, metric: str) -> tuple[dict[str, float | str], np.ndarray]:
    if metric == "axis":
        series = orient_axis(marker)
        embedded = delay_embed_axis_for_divergence(series)
        dist = pairwise_axis_distances(series)
    else:
        series = marker
        embedded = delay_embed(series)
        dist = pairwise_distances(embedded)
    rec, epsilon, rr = recurrence_from_distances(dist)
    diag = diagonal_lengths(rec)
    vert = vertical_lengths(rec)
    rec_points = int(rec.sum())
    diag_points = int(sum(diag))
    vert_points = int(sum(vert))
    max_diag = max(diag) if diag else 0
    max_vert = max(vert) if vert else 0
    n = rec.shape[0]
    metrics: dict[str, float | str] = {
        "samples": float(len(series)),
        "orientation_metric": metric,
        "embedded_samples": float(n),
        "dimension": float(EMBED_DIM),
        "delay": float(EMBED_TAU),
        "target_recurrence_rate": float(TARGET_RR),
        "theiler": float(THEILER),
        "epsilon": epsilon,
        "rr": rr,
        "det": float(diag_points / rec_points) if rec_points else 0.0,
        "lam": float(vert_points / rec_points) if rec_points else 0.0,
        "mean_diag": float(np.mean(diag)) if diag else 0.0,
        "max_diag": float(max_diag),
        "max_diag_fraction": float(max_diag / n) if n else 0.0,
        "divergence": float(1.0 / max_diag) if max_diag else math.inf,
        "mean_vertical": float(np.mean(vert)) if vert else 0.0,
        "max_vertical": float(max_vert),
        "diag_entropy": line_entropy(diag),
    }
    metrics.update(spectral_metrics(series))
    metrics.update(rosenstein_metrics(embedded, dist, dt))
    metrics["chaos_score"] = chaos_score(metrics)
    metrics["broadband_chaos_score"] = broadband_chaos_score(metrics)
    metrics["class"] = classify(metrics)
    metrics["behaviour_family"] = behaviour_family(str(metrics["class"]))
    return metrics, rec


def classify(metrics: dict[str, float | str]) -> str:
    det = float(metrics["det"])
    lmax = float(metrics["max_diag_fraction"])
    entropy = float(metrics["spectral_entropy"])
    top5 = float(metrics["top5_power_fraction"])
    dominant = float(metrics["dominant_power_fraction"])
    score = float(metrics.get("chaos_score", 0.0))
    broad_score = float(metrics.get("broadband_chaos_score", 0.0))
    gain = float(metrics.get("neighbor_divergence_gain", 0.0))

    narrow_regular_spectrum = top5 > 0.965 and entropy < 0.21
    if lmax < 0.25 and (broad_score >= 0.42 or top5 < 0.94 or entropy > 0.30):
        return "chaotic-like"
    if det < 0.88 and lmax < 0.18 and entropy > 0.70 and top5 < 0.45:
        return "chaotic-candidate"
    if score >= 0.50 and lmax < 0.30:
        return "sensitive-regular"
    if det > 0.995 and lmax > 0.45 and entropy < 0.38 and top5 > 0.80:
        return "periodic"
    if det > 0.985 and lmax > 0.22 and entropy < 0.60 and top5 > 0.62:
        return "quasi-periodic"
    if det > 0.94 and lmax > 0.10:
        return "complex-regular"
    if dominant > 0.25 and det > 0.90:
        return "weakly-regular"
    return "ambiguous/irregular"


def behaviour_family(label: str) -> str:
    if label == "periodic":
        return "periodic"
    if label in {"quasi-periodic", "complex-regular", "weakly-regular", "sensitive-regular"}:
        return "regular"
    if label in {"chaotic-like", "chaotic-candidate"}:
        return "chaotic-candidate"
    return "irregular"


def plot_recurrence(rec: np.ndarray, out_path: Path, title: str, metrics: dict[str, float | str]) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 5.8), constrained_layout=True)
    ax.imshow(rec, origin="lower", cmap="binary", interpolation="nearest", aspect="equal")
    ax.set_xlabel("embedded time index")
    ax.set_ylabel("embedded time index")
    ax.set_title(title, fontsize=10)
    label = (
        f"{metrics['class']}\n"
        f"DET={float(metrics['det']):.3f}, Lmax/N={float(metrics['max_diag_fraction']):.3f}\n"
        f"Hspec={float(metrics['spectral_entropy']):.3f}, top5={float(metrics['top5_power_fraction']):.3f}"
    )
    ax.text(
        0.01,
        0.99,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def collect_cases(spec: SourceSpec) -> list[Path]:
    if not spec.root.exists():
        return []
    return sorted(
        [path for path in spec.root.iterdir() if path.is_dir() and (path / spec.data_file).exists()],
        key=case_sort_key,
    )


def analyse_source(spec: SourceSpec, out_root: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for case_dir in collect_cases(spec):
        data_path = case_dir / spec.data_file
        time, marker = load_marker(data_path, spec.marker_cols)
        dt = float(np.median(np.diff(time))) if len(time) > 1 else 1.0
        metrics, rec = rqa_metrics(marker, dt, spec.metric)
        meta = parse_case_name(case_dir.name)
        row: dict[str, float | str] = {
            "case": case_dir.name,
            "family": spec.family,
            "source": spec.label,
            "method": spec.method,
            "orientation_metric": spec.metric,
            "path": str(data_path),
            "duration": float(time[-1] - time[0]),
        }
        row.update(meta)
        row.update(metrics)
        rows.append(row)
        out_path = out_root / "plots" / spec.label / f"{case_dir.name}_recurrence.png"
        title = f"{spec.label}: {case_dir.name}"
        plot_recurrence(rec, out_path, title, metrics)
    return rows


def rms_to_exact(
    case: str,
    source_root: Path,
    source_file: str,
    source_cols: tuple[str, str, str],
    exact_root: Path,
) -> float:
    source_path = source_root / case / source_file
    exact_path = exact_root / case / "exact_added_mass.csv"
    if not source_path.exists() or not exact_path.exists():
        return math.nan
    time_a, marker_a = load_marker(source_path, source_cols)
    time_b, marker_b = load_marker(exact_path, ("ofx", "ofy", "ofz"))
    n = min(len(time_a), len(time_b))
    if n == 0:
        return math.nan
    if np.max(np.abs(time_a[:n] - time_b[:n])) > 1e-9:
        interp = np.column_stack(
            [np.interp(time_a, time_b, marker_b[:, i]) for i in range(marker_b.shape[1])]
        )
        diff = marker_a - interp
    else:
        diff = marker_a[:n] - marker_b[:n]
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def build_agreement(rows: list[dict[str, float | str]], specs: list[SourceSpec]) -> list[dict[str, float | str]]:
    by_key: dict[tuple[str, str], dict[str, float | str]] = {
        (str(row["case"]), str(row["source"])): row for row in rows
    }
    cases = sorted({str(row["case"]) for row in rows}, key=lambda name: case_sort_key(Path(name)))
    spec_by_label = {spec.label: spec for spec in specs}
    triaxial_exact_root = spec_by_label["triaxial_exact"].root
    spheroid_exact_root = spec_by_label["spheroid_exact"].root

    out: list[dict[str, float | str]] = []
    for case in cases:
        row: dict[str, float | str] = {"case": case}
        meta = parse_case_name(case)
        row.update(meta)
        for source in [
            "triaxial_nd2",
            "triaxial_nd3",
            "triaxial_exact",
            "spheroid_nd2",
            "spheroid_exact",
        ]:
            item = by_key.get((case, source))
            if item:
                row[f"{source}_class"] = item["class"]
                row[f"{source}_behaviour_family"] = item["behaviour_family"]
                row[f"{source}_det"] = item["det"]
                row[f"{source}_max_diag_fraction"] = item["max_diag_fraction"]
                row[f"{source}_spectral_entropy"] = item["spectral_entropy"]
                row[f"{source}_top5_power_fraction"] = item["top5_power_fraction"]
            else:
                row[f"{source}_class"] = ""

        row["triaxial_nd2_rms_to_exact"] = rms_to_exact(
            case,
            spec_by_label["triaxial_nd2"].root,
            "single_body_complete.dat",
            ("ofix1_1", "ofix2_1", "ofix3_1"),
            triaxial_exact_root,
        )
        row["triaxial_nd3_rms_to_exact"] = rms_to_exact(
            case,
            spec_by_label["triaxial_nd3"].root,
            "single_body_complete.dat",
            ("ofix1_1", "ofix2_1", "ofix3_1"),
            triaxial_exact_root,
        )
        row["spheroid_nd2_rms_to_exact"] = rms_to_exact(
            case,
            spec_by_label["spheroid_nd2"].root,
            "single_body_complete.dat",
            ("ofix1_1", "ofix2_1", "ofix3_1"),
            spheroid_exact_root,
        )
        row["triaxial_mesh_class_changed"] = (
            row.get("triaxial_nd2_class", "") != row.get("triaxial_nd3_class", "")
        )
        row["triaxial_mesh_family_changed"] = (
            row.get("triaxial_nd2_behaviour_family", "")
            != row.get("triaxial_nd3_behaviour_family", "")
        )
        row["triaxial_nd3_exact_class_changed"] = (
            row.get("triaxial_nd3_class", "") != row.get("triaxial_exact_class", "")
        )
        row["triaxial_nd3_exact_family_changed"] = (
            row.get("triaxial_nd3_behaviour_family", "")
            != row.get("triaxial_exact_behaviour_family", "")
        )
        row["spheroid_exact_class_changed"] = (
            row.get("spheroid_nd2_class", "") != row.get("spheroid_exact_class", "")
        )
        row["spheroid_exact_family_changed"] = (
            row.get("spheroid_nd2_behaviour_family", "")
            != row.get("spheroid_exact_behaviour_family", "")
        )
        row["triaxial_mesh_metric_shift"] = metric_shift(row, "triaxial_nd2", "triaxial_nd3")
        row["triaxial_nd3_exact_metric_shift"] = metric_shift(row, "triaxial_nd3", "triaxial_exact")
        row["spheroid_exact_metric_shift"] = metric_shift(row, "spheroid_nd2", "spheroid_exact")
        out.append(row)
    return out


def metric_shift(row: dict[str, float | str], a: str, b: str) -> bool:
    try:
        dl = abs(float(row[f"{a}_max_diag_fraction"]) - float(row[f"{b}_max_diag_fraction"]))
        dh = abs(float(row[f"{a}_spectral_entropy"]) - float(row[f"{b}_spectral_entropy"]))
        dt = abs(float(row[f"{a}_top5_power_fraction"]) - float(row[f"{b}_top5_power_fraction"]))
    except (KeyError, TypeError, ValueError):
        return False
    return bool(dl > 0.12 or dh > 0.12 or dt > 0.18)


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def grouped_summary(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    groups: dict[tuple[str, str, float, float], list[dict[str, float | str]]] = {}
    for row in rows:
        try:
            key = (
                str(row["family"]),
                str(row["source"]),
                float(row["rho"]),
                float(row["ratio"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        groups.setdefault(key, []).append(row)

    out: list[dict[str, float | str]] = []
    metric_names = [
        "chaos_score",
        "broadband_chaos_score",
        "spectral_entropy",
        "top5_power_fraction",
        "max_diag_fraction",
        "neighbor_divergence_slope",
        "neighbor_divergence_gain",
    ]
    for (family, source, rho, ratio), items in sorted(groups.items()):
        row: dict[str, float | str] = {
            "family": family,
            "source": source,
            "rho": rho,
            "ratio": ratio,
            "n": len(items),
        }
        classes = [str(item["class"]) for item in items]
        chaotic_count = sum(str(item["behaviour_family"]) == "chaotic-candidate" for item in items)
        row["chaotic_like_count"] = chaotic_count
        row["chaotic_like_fraction"] = chaotic_count / len(items)
        row["classes"] = ";".join(classes)
        for name in metric_names:
            values = np.asarray([float(item[name]) for item in items], dtype=float)
            row[f"{name}_mean"] = float(np.mean(values))
            row[f"{name}_min"] = float(np.min(values))
            row[f"{name}_max"] = float(np.max(values))
        row["regime"] = "chaotic-regime" if row["broadband_chaos_score_mean"] >= 0.42 or chaotic_count >= 3 else "regular-regime"
        out.append(row)
    return out


def plot_overview(rows: list[dict[str, float | str]], agreement: list[dict[str, float | str]], out_path: Path) -> None:
    sources = ["triaxial_nd2", "triaxial_nd3", "triaxial_exact", "spheroid_nd2", "spheroid_exact"]
    cases = sorted({str(row["case"]) for row in rows}, key=lambda name: case_sort_key(Path(name)))
    by_key = {(str(row["case"]), str(row["source"])): row for row in rows}
    class_order = {
        "periodic": 0,
        "quasi-periodic": 1,
        "complex-regular": 2,
        "weakly-regular": 3,
        "sensitive-regular": 4,
        "ambiguous/irregular": 5,
        "chaotic-like": 6,
        "chaotic-candidate": 7,
        "": np.nan,
    }
    class_grid = np.full((len(sources), len(cases)), np.nan)
    entropy_grid = np.full_like(class_grid, np.nan, dtype=float)
    lmax_grid = np.full_like(class_grid, np.nan, dtype=float)
    chaos_grid = np.full_like(class_grid, np.nan, dtype=float)
    for i, source in enumerate(sources):
        for j, case in enumerate(cases):
            row = by_key.get((case, source))
            if not row:
                continue
            class_grid[i, j] = class_order.get(str(row["class"]), np.nan)
            entropy_grid[i, j] = float(row["spectral_entropy"])
            lmax_grid[i, j] = float(row["max_diag_fraction"])
            chaos_grid[i, j] = float(row["chaos_score"])

    fig, axes = plt.subplots(4, 1, figsize=(max(10.5, 0.55 * len(cases)), 10.8), constrained_layout=True)
    finite_classes = [value for value in class_order.values() if np.isfinite(value)]
    im0 = axes[0].imshow(
        class_grid,
        aspect="auto",
        interpolation="nearest",
        vmin=min(finite_classes),
        vmax=max(finite_classes),
        cmap="viridis",
    )
    axes[0].set_title("Recurrence behaviour class")
    axes[0].set_yticks(range(len(sources)), sources)
    axes[0].set_xticks(range(len(cases)), cases, rotation=60, ha="right", fontsize=7)
    cbar = fig.colorbar(im0, ax=axes[0], pad=0.01)
    cbar.set_ticks(list(class_order.values())[:-1])
    cbar.set_ticklabels(list(class_order.keys())[:-1])

    im1 = axes[1].imshow(entropy_grid, aspect="auto", interpolation="nearest", cmap="magma")
    axes[1].set_title("Spectral entropy of orientation trajectory")
    axes[1].set_yticks(range(len(sources)), sources)
    axes[1].set_xticks(range(len(cases)), cases, rotation=60, ha="right", fontsize=7)
    fig.colorbar(im1, ax=axes[1], pad=0.01)

    im2 = axes[2].imshow(lmax_grid, aspect="auto", interpolation="nearest", cmap="cividis", vmin=0, vmax=1)
    axes[2].set_title("Longest diagonal recurrence line, Lmax/N")
    axes[2].set_yticks(range(len(sources)), sources)
    axes[2].set_xticks(range(len(cases)), cases, rotation=60, ha="right", fontsize=7)
    fig.colorbar(im2, ax=axes[2], pad=0.01)

    im3 = axes[3].imshow(chaos_grid, aspect="auto", interpolation="nearest", cmap="inferno", vmin=0, vmax=1)
    axes[3].set_title("Chaos-likeness score")
    axes[3].set_yticks(range(len(sources)), sources)
    axes[3].set_xticks(range(len(cases)), cases, rotation=60, ha="right", fontsize=7)
    fig.colorbar(im3, ax=axes[3], pad=0.01)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)

    flags = [row for row in agreement if any(bool(row.get(key)) for key in (
        "triaxial_mesh_class_changed",
        "triaxial_mesh_family_changed",
        "triaxial_nd3_exact_class_changed",
        "triaxial_nd3_exact_family_changed",
        "spheroid_exact_class_changed",
        "spheroid_exact_family_changed",
        "triaxial_mesh_metric_shift",
        "triaxial_nd3_exact_metric_shift",
        "spheroid_exact_metric_shift",
    ))]
    if flags:
        text_path = out_path.with_name("ke_ratio_density_recurrence_current_comparison_flags.txt")
        with text_path.open("w") as f:
            for row in flags:
                f.write(f"{row['case']}: {row}\n")


def plot_grouped_regimes(group_rows: list[dict[str, float | str]], out_path: Path) -> None:
    triaxial = [row for row in group_rows if str(row["family"]).startswith("triaxial")]
    if not triaxial:
        return
    sources = ["triaxial_exact", "triaxial_nd2", "triaxial_nd3"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for source in sources:
        items = sorted([row for row in triaxial if row["source"] == source], key=lambda r: float(r["rho"]))
        if not items:
            continue
        rho = np.asarray([float(row["rho"]) for row in items])
        broad = np.asarray([float(row["broadband_chaos_score_mean"]) for row in items])
        full = np.asarray([float(row["chaos_score_mean"]) for row in items])
        axes[0].plot(rho, broad, marker="o", label=source)
        axes[1].plot(rho, full, marker="o", label=source)
    for ax in axes:
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.axhline(0.42, color="0.35", lw=1.0, ls="--")
        ax.set_xlabel("density rho")
        ax.grid(True, which="both", color="0.9")
        ax.legend(fontsize=8)
    axes[0].set_title("Broadband recurrence chaos score")
    axes[0].set_ylabel("group mean score")
    axes[1].set_title("Combined chaos-likeness score")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def default_specs() -> list[SourceSpec]:
    return [
        SourceSpec(
            "triaxial_nd2",
            SCRIPT_DIR / "runs_impulse_nd2",
            "single_body_complete.dat",
            ("ofix1_1", "ofix2_1", "ofix3_1"),
            "triaxial_1_0p8_0p6",
            "impulse_nd2",
            "marker",
        ),
        SourceSpec(
            "triaxial_nd3",
            SCRIPT_DIR / "runs_impulse_nd3",
            "single_body_complete.dat",
            ("ofix1_1", "ofix2_1", "ofix3_1"),
            "triaxial_1_0p8_0p6",
            "impulse_nd3",
            "marker",
        ),
        SourceSpec(
            "triaxial_exact",
            SCRIPT_DIR / "runs_impulse_nd2",
            "exact_added_mass.csv",
            ("ofx", "ofy", "ofz"),
            "triaxial_1_0p8_0p6",
            "exact_added_mass",
            "marker",
        ),
        SourceSpec(
            "spheroid_nd2",
            SCRIPT_DIR / "runs_impulse_spheroid_1_0p7_0p7_nd2",
            "single_body_complete.dat",
            ("ofix1_1", "ofix2_1", "ofix3_1"),
            "spheroid_1_0p7_0p7",
            "impulse_nd2",
            "axis",
        ),
        SourceSpec(
            "spheroid_exact",
            SCRIPT_DIR / "runs_impulse_spheroid_1_0p7_0p7_nd2",
            "exact_added_mass.csv",
            ("ofx", "ofy", "ofz"),
            "spheroid_1_0p7_0p7",
            "exact_added_mass",
            "axis",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    specs = default_specs()
    rows: list[dict[str, float | str]] = []
    for spec in specs:
        source_rows = analyse_source(spec, args.out)
        print(f"{spec.label}: {len(source_rows)} cases")
        rows.extend(source_rows)

    summary_path = args.out / "ke_ratio_density_recurrence_current_comparison_summary.csv"
    write_csv(summary_path, rows)
    agreement = build_agreement(rows, specs)
    agreement_path = args.out / "ke_ratio_density_recurrence_current_comparison_agreement.csv"
    write_csv(agreement_path, agreement)
    groups = grouped_summary(rows)
    group_path = args.out / "ke_ratio_density_recurrence_current_comparison_grouped.csv"
    write_csv(group_path, groups)
    plot_overview(rows, agreement, args.out / "ke_ratio_density_recurrence_current_comparison_overview.png")
    plot_grouped_regimes(groups, args.out / "ke_ratio_density_recurrence_current_comparison_grouped.png")
    print(f"summary: {summary_path}")
    print(f"agreement: {agreement_path}")
    print(f"grouped: {group_path}")


if __name__ == "__main__":
    main()
