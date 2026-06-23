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
OUT_BROADBAND_PNG = STUDY / "two_body_dynamics_broadband_score.png"

DEFAULT_TRANSIENT_FRACTION = 0.5
RECURRENCE_DIM = 3
RECURRENCE_TAU = 10
RECURRENCE_RATE = 0.03
RECURRENCE_THEILER = 50
MIN_DIAG_LEN = 4
TOP_SPECTRAL_PEAKS = 12
DIVERGENCE_HORIZON = 35
STATE_MODES = ("marker", "quaternion", "axis")

CLASS_ORDER = [
    "periodic",
    "quasi-periodic",
    "sensitive-regular",
    "complex-regular",
    "chaotic-like",
    "chaotic",
    "chaotic-candidate",
    "ambiguous",
    "mixed",
    "incomplete",
]
CLASS_COLOR = {
    "periodic": "#2ca02c",
    "quasi-periodic": "#1f77b4",
    "sensitive-regular": "#9467bd",
    "complex-regular": "#17becf",
    "chaotic-like": "#d62728",
    "chaotic": "#d62728",
    "chaotic-candidate": "#ff7f0e",
    "ambiguous": "#ffbf00",
    "mixed": "#7f7f7f",
    "incomplete": "#9e9e9e",
}
CLASS_CODE = {
    "periodic": "P",
    "quasi-periodic": "Q",
    "sensitive-regular": "S",
    "complex-regular": "R",
    "chaotic-like": "C",
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


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
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


def quaternion(data: np.ndarray, body: int) -> np.ndarray:
    q = np.column_stack([data[f"q0_{body}"], data[f"q1_{body}"], data[f"q2_{body}"], data[f"q3_{body}"]])
    norm = np.linalg.norm(q, axis=1)
    norm[norm < 1.0e-14] = 1.0
    return q / norm[:, None]


def continuity_orient_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.array(q, dtype=float, copy=True)
    for i in range(1, len(q)):
        if float(np.dot(q[i - 1], q[i])) < 0.0:
            q[i] *= -1.0
    return q


def continuity_orient_axis(axis: np.ndarray) -> np.ndarray:
    axis = np.array(axis, dtype=float, copy=True)
    norm = np.linalg.norm(axis, axis=1)
    norm[norm < 1.0e-14] = 1.0
    axis /= norm[:, None]
    for i in range(1, len(axis)):
        if float(np.dot(axis[i - 1], axis[i])) < 0.0:
            axis[i] *= -1.0
    return axis


def angular_velocity(data: np.ndarray, body: int) -> np.ndarray:
    return np.column_stack([data[f"w1_{body}"], data[f"w2_{body}"], data[f"w3_{body}"]])


def position(data: np.ndarray, body: int) -> np.ndarray:
    return np.column_stack([data[f"px_{body}"], data[f"py_{body}"], data[f"pz_{body}"]])


def body_state(data: np.ndarray, body: int, state_mode: str) -> np.ndarray:
    omega = angular_velocity(data, body)
    if state_mode == "marker":
        return np.column_stack([marker(data, body), omega])
    if state_mode == "quaternion":
        return np.column_stack([continuity_orient_quaternion(quaternion(data, body)), omega])
    if state_mode == "axis":
        return np.column_stack([continuity_orient_axis(marker(data, body)), omega])
    raise ValueError(f"unknown state mode {state_mode}")


def combined_state(data: np.ndarray, state_mode: str) -> np.ndarray:
    p1 = position(data, 1)
    p2 = position(data, 2)
    rel = p2 - p1
    sep = np.linalg.norm(rel, axis=1)[:, None]
    if state_mode == "marker":
        orientation = [marker(data, 1), marker(data, 2)]
    elif state_mode == "quaternion":
        orientation = [continuity_orient_quaternion(quaternion(data, 1)), continuity_orient_quaternion(quaternion(data, 2))]
    elif state_mode == "axis":
        orientation = [continuity_orient_axis(marker(data, 1)), continuity_orient_axis(marker(data, 2))]
    else:
        raise ValueError(f"unknown state mode {state_mode}")
    return np.column_stack([*orientation, rel, sep])


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


def pairwise_angular_distances(vectors: np.ndarray, double_cover: bool) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(vectors, axis=1)
    norms[norms < 1.0e-14] = 1.0
    unit = vectors / norms[:, None]
    dots = np.clip(unit @ unit.T, -1.0, 1.0)
    if double_cover:
        dots = np.abs(dots)
    if unit.shape[1] == 4:
        return 2.0 * np.arccos(dots)
    return np.arccos(dots)


def distance_scale(dist: np.ndarray) -> float:
    n = dist.shape[0]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    values = dist[mask & np.isfinite(dist)]
    values = values[values > 1.0e-12]
    if values.size == 0:
        return 1.0
    scale = float(np.median(values))
    return scale if math.isfinite(scale) and scale > 1.0e-12 else 1.0


def pairwise_state_distances(
    series: np.ndarray,
    state_mode: str,
    combined: bool = False,
    dim: int = RECURRENCE_DIM,
    tau: int = RECURRENCE_TAU,
) -> np.ndarray:
    if state_mode == "marker":
        return pairwise_distances(delay_embed(series, dim=dim, tau=tau))

    series = np.asarray(series, dtype=float)
    n = len(series) - (dim - 1) * tau
    if n <= 2:
        raise ValueError("not enough samples for delay embedding")

    if state_mode == "quaternion":
        orient_width = 4
        orient_groups = [(0, 4), (4, 8)] if combined else [(0, 4)]
        double_cover = True
    elif state_mode == "axis":
        orient_width = 3
        orient_groups = [(0, 3), (3, 6)] if combined else [(0, 3)]
        double_cover = True
    else:
        raise ValueError(f"unknown state mode {state_mode}")

    orient_cols = {idx for start, stop in orient_groups for idx in range(start, stop)}
    euclidean_cols = [idx for idx in range(series.shape[1]) if idx not in orient_cols]
    euclidean = standardize(series[:, euclidean_cols]) if euclidean_cols else np.zeros((len(series), 0))
    dist2 = np.zeros((n, n), dtype=float)
    for k in range(dim):
        start = k * tau
        stop = start + n
        for group_start, group_stop in orient_groups:
            if group_stop - group_start != orient_width:
                raise ValueError("invalid orientation group")
            angle = pairwise_angular_distances(series[start:stop, group_start:group_stop], double_cover)
            scale = distance_scale(angle)
            dist2 += (angle / scale) ** 2
        if euclidean_cols:
            block = euclidean[start:stop]
            diff = block[:, None, :] - block[None, :, :]
            dist2 += np.sum(diff * diff, axis=2)
    return np.sqrt(dist2)


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


def rqa_metrics(series: np.ndarray, state_mode: str = "marker", combined: bool = False) -> dict[str, float]:
    dist = pairwise_state_distances(series, state_mode=state_mode, combined=combined)
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


def broadband_chaos_score(metrics: dict[str, float]) -> float:
    lmax = metrics["lmax_fraction"]
    sent = metrics["spectral_entropy"]
    peak = metrics["peak_fraction"]
    short_diagonal = max(0.0, min(1.0, (0.32 - lmax) / 0.32))
    broad_spectrum = max(0.0, min(1.0, (0.985 - peak) / 0.45))
    entropy_score = max(0.0, min(1.0, sent / 0.60))
    return float(0.40 * broad_spectrum + 0.35 * entropy_score + 0.25 * short_diagonal)


def chaos_score(metrics: dict[str, float]) -> float:
    broad = broadband_chaos_score(metrics)
    div = max(0.0, metrics["divergence_slope"])
    gain = max(0.0, metrics["divergence_gain"])
    divergence_score = max(0.0, min(1.0, div / 0.018))
    gain_score = max(0.0, min(1.0, gain / 0.25))
    return float(0.75 * broad + 0.15 * divergence_score + 0.10 * gain_score)


def behaviour_family(label: str) -> str:
    if label == "periodic":
        return "periodic"
    if label in {"quasi-periodic", "sensitive-regular", "complex-regular"}:
        return "regular"
    if label in {"chaotic-like", "chaotic", "chaotic-candidate"}:
        return "chaotic-candidate"
    return label


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
    broad_score = metrics["broadband_chaos_score"]
    full_score = metrics["chaos_score"]

    strong_divergence = div > 0.018 and gain > 0.25
    weak_divergence = div > 0.010 and gain > 0.18
    narrow_regular_spectrum = peak > 0.94 and sent < 0.32

    if lmax < 0.25 and (broad_score >= 0.42 or peak < 0.78 or sent > 0.42):
        return "chaotic-like"
    if det < 0.78 and lmax < 0.22 and sent > 0.42 and peak < 0.68:
        return "chaotic-like"
    if sent > 0.62 and peak < 0.50:
        return "chaotic-like"

    if det > 0.90 and lmax > 0.35 and sent < 0.30 and peak > 0.82 and harmonicity > 0.78:
        return "periodic"
    if det > 0.80 and lmax > 0.20 and sent < 0.50 and peak > 0.58:
        if harmonicity > 0.82 and dominant > 0.28 and sent < 0.38:
            return "periodic"
        return "quasi-periodic"
    if full_score >= 0.50 and lmax < 0.32:
        return "sensitive-regular" if narrow_regular_spectrum else "chaotic-like"
    if weak_divergence and narrow_regular_spectrum:
        return "sensitive-regular"
    if det > 0.80 or peak > 0.72:
        return "complex-regular"
    return "ambiguous"


def analyze_series(series: np.ndarray, state_mode: str = "marker", combined: bool = False) -> dict[str, object]:
    metrics = {**rqa_metrics(series, state_mode=state_mode, combined=combined), **spectral_metrics(series)}
    metrics["broadband_chaos_score"] = broadband_chaos_score(metrics)
    metrics["chaos_score"] = chaos_score(metrics)
    metrics["class"] = classify(metrics)
    metrics["behaviour_family"] = behaviour_family(str(metrics["class"]))
    return metrics


def consensus(classes: list[str]) -> str:
    useful = [base_class(c) for c in classes if base_class(c) != "incomplete"]
    if not useful:
        return "incomplete"
    counts = Counter(useful)
    top, n_top = counts.most_common(1)[0]
    if n_top == len(useful):
        return top
    chaotic_votes = sum(counts[name] for name in ("chaotic-like", "chaotic", "chaotic-candidate"))
    if chaotic_votes > 0 and counts["periodic"] > 0:
        return "mixed"
    if chaotic_votes >= math.ceil(len(useful) / 2):
        return "chaotic-like"
    if n_top >= math.ceil(len(useful) / 2):
        return top
    return "mixed"


def analyze_run(row: dict[str, str], transient_fraction: float, state_mode: str) -> dict[str, object]:
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
            "state_mode": state_mode,
            "class_body1": "incomplete",
            "class_body2": "incomplete",
            "class_combined": "incomplete",
            "class_consensus": "incomplete",
        }

    data = load_output(row)
    names = data.dtype.names or ()
    required = {"time", "ofix1_1", "ofix1_2", "w1_1", "w1_2", "px_1", "px_2"}
    if state_mode == "quaternion":
        required.update({f"q{i}_{body}" for body in (1, 2) for i in range(4)})
    if not required.issubset(names):
        raise KeyError(f"{row['name']} is missing required columns: {sorted(required - set(names))}")
    if not np.all(np.isfinite(data["time"])):
        return {**base, "status": "nonfinite", "class_consensus": "incomplete"}

    t = data["time"]
    keep = t >= (float(row["tend"]) * transient_fraction)
    data = data[keep]
    analyses = {
        "body1": analyze_series(body_state(data, 1, state_mode), state_mode=state_mode),
        "body2": analyze_series(body_state(data, 2, state_mode), state_mode=state_mode),
        "combined": analyze_series(combined_state(data, state_mode), state_mode=state_mode, combined=True),
    }
    classes = [str(analyses[key]["class"]) for key in ("body1", "body2", "combined")]
    body_classes = [str(analyses[key]["class"]) for key in ("body1", "body2")]
    out: dict[str, object] = {
        **base,
        "status": "ok",
        "state_mode": state_mode,
        "class_body1": classes[0],
        "class_body2": classes[1],
        "class_combined": classes[2],
        "class_consensus": consensus(body_classes),
        "class_consensus_with_combined": consensus(classes),
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
        classes: list[str] = []
        for row in group:
            if row.get("status") == "ok":
                classes.extend([str(row.get("class_body1", "incomplete")), str(row.get("class_body2", "incomplete"))])
            else:
                classes.append(str(row.get("class_consensus", "incomplete")))
        counts = Counter(classes)
        ok_group = [row for row in group if row.get("status") == "ok"]

        def mean_metric(name: str) -> float:
            values: list[float] = []
            for row in ok_group:
                for body in ("body1", "body2"):
                    value = row.get(f"{body}_{name}")
                    try:
                        fvalue = float(value)  # type: ignore[arg-type]
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(fvalue):
                        values.append(fvalue)
            return float(np.mean(values)) if values else float("nan")

        chaotic_like_count = sum(counts[name] for name in ("chaotic-like", "chaotic", "chaotic-candidate"))
        n_body_observations = len(classes)
        chaotic_like_fraction = chaotic_like_count / n_body_observations if n_body_observations else 0.0
        group_class = consensus(classes)
        broadband_mean = mean_metric("broadband_chaos_score")
        chaos_mean = mean_metric("chaos_score")
        regime = (
            "chaotic-regime"
            if chaotic_like_fraction >= 0.5 or broadband_mean >= 0.42 or chaos_mean >= 0.50
            else "regular-regime"
        )
        out.append(
            {
                "shape_name": shape,
                "rho": rho,
                "energy_ratio": energy,
                "separation": sep,
                "n_repeats": len(group),
                "n_ok": len(ok_group),
                "n_body_observations": n_body_observations,
                "repeat_classes": ";".join(classes),
                "group_class": group_class,
                "regime": regime,
                "chaotic_like_count": chaotic_like_count,
                "chaotic_like_fraction": chaotic_like_fraction,
                "body_broadband_chaos_score_mean": broadband_mean,
                "body_chaos_score_mean": chaos_mean,
                "body_spectral_entropy_mean": mean_metric("spectral_entropy"),
                "body_peak_fraction_mean": mean_metric("peak_fraction"),
                "body_lmax_fraction_mean": mean_metric("lmax_fraction"),
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

    present_classes = {
        base_class(piece)
        for row in summary_rows
        for piece in str(row.get("repeat_classes", "")).split(";") + [str(row.get("group_class", ""))]
        if piece
    }
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
        if cls in present_classes
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=min(5, max(1, len(handles))), fontsize=9)
    fig.suptitle(f"Two-body dynamics classification from {title_suffix} trajectories", fontsize=14)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def plot_broadband_score(summary_rows: list[dict[str, object]], out_png: Path, title_suffix: str) -> None:
    shapes = sorted({str(row["shape_name"]) for row in summary_rows})
    fig, axes = plt.subplots(len(shapes), 1, figsize=(13, 4.8 * len(shapes)), constrained_layout=True)
    if len(shapes) == 1:
        axes = [axes]

    last_mesh = None
    for ax, shape in zip(axes, shapes):
        rows = [row for row in summary_rows if row["shape_name"] == shape]
        ys = sorted({(float(row["rho"]), float(row["separation"])) for row in rows}, reverse=True)
        xs = sorted({float(row["energy_ratio"]) for row in rows})
        grid = np.full((len(ys), len(xs)), np.nan)
        labels = [["" for _ in xs] for _ in ys]
        y_index = {key: i for i, key in enumerate(ys)}
        x_index = {key: i for i, key in enumerate(xs)}

        for row in rows:
            y = y_index[(float(row["rho"]), float(row["separation"]))]
            x = x_index[float(row["energy_ratio"])]
            try:
                score = float(row["body_broadband_chaos_score_mean"])
            except (KeyError, TypeError, ValueError):
                score = float("nan")
            grid[y, x] = score
            labels[y][x] = f"{score:.2f}" if math.isfinite(score) else "NA"

        last_mesh = ax.imshow(grid, vmin=0.0, vmax=1.0, cmap="inferno", aspect="auto")
        ax.set_title(shape)
        ax.set_xticks(range(len(xs)), [f"E={x:g}" for x in xs])
        ax.set_yticks(range(len(ys)), [f"rho={rho:g}, sep={sep:g}" for rho, sep in ys])
        ax.set_xlim(-0.5, len(xs) - 0.5)
        ax.set_ylim(len(ys) - 0.5, -0.5)
        ax.set_xticks(np.arange(-0.5, len(xs), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(ys), 1), minor=True)
        ax.grid(which="minor", color="white", lw=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)

        for y in range(len(ys)):
            for x in range(len(xs)):
                value = grid[y, x]
                color = "white" if math.isfinite(value) and value > 0.45 else "black"
                ax.text(x, y, labels[y][x], ha="center", va="center", fontsize=10, color=color, weight="bold")

    if last_mesh is not None:
        cbar = fig.colorbar(last_mesh, ax=axes, shrink=0.88, pad=0.015)
        cbar.set_label("mean body broadband recurrence chaos score")
    fig.suptitle(f"Two-body broadband recurrence score from {title_suffix} trajectories", fontsize=14)
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
    parser.add_argument("--manifest", type=Path, default=MANIFEST, help="manifest CSV to classify")
    parser.add_argument("--out-dir", type=Path, default=STUDY, help="directory for classification outputs")
    parser.add_argument(
        "--state-mode",
        choices=STATE_MODES,
        default="marker",
        help=(
            "recurrence state: marker uses the tracked body marker; quaternion uses the full sign-invariant "
            "orientation quaternion; axis uses the marker as an antipodal axis for axisymmetric bodies"
        ),
    )
    args = parser.parse_args()

    manifest = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(manifest)
    if args.limit is not None:
        rows = rows[: args.limit]

    if not 0.0 <= args.transient_fraction < 1.0:
        raise SystemExit("--transient-fraction must be in [0, 1)")

    out_csv = out_dir / f"{OUT_CSV.stem}{args.suffix}{OUT_CSV.suffix}"
    out_summary = out_dir / f"{OUT_SUMMARY.stem}{args.suffix}{OUT_SUMMARY.suffix}"
    out_png = out_dir / f"{OUT_PNG.stem}{args.suffix}{OUT_PNG.suffix}"
    out_broadband_png = out_dir / f"{OUT_BROADBAND_PNG.stem}{args.suffix}{OUT_BROADBAND_PNG.suffix}"
    time_suffix = "full-run" if args.transient_fraction == 0.0 else f"post-transient ({args.transient_fraction:g} tend)"
    title_suffix = f"{time_suffix}, {args.state_mode} state"

    results = []
    for i, row in enumerate(rows, start=1):
        print(f"[{i:03d}/{len(rows):03d}] {row['name']}", flush=True)
        results.append(analyze_run(row, args.transient_fraction, args.state_mode))

    summary = summarize(results)
    write_rows(out_csv, results)
    write_rows(out_summary, summary)
    plot_summary(summary, out_png, title_suffix)
    plot_broadband_score(summary, out_broadband_png, title_suffix)
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_summary}")
    print(f"Wrote {out_png}")
    print(f"Wrote {out_broadband_png}")


if __name__ == "__main__":
    main()
