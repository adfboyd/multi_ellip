from __future__ import annotations

import argparse
import csv
import itertools
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "studies" / "multibody_energy_balance"
DEFAULT_ROOTS = [
    ROOT / "studies" / "three_body" / "runs",
    ROOT / "studies" / "density_sweep" / "runs",
    ROOT / "studies" / "impulse_coupling" / "runs",
]


@dataclass(frozen=True)
class RunCase:
    name: str
    path: Path
    output: Path
    nbody: int
    density: float
    ndiv: int
    dt: float
    tend: float
    source: str


def parse_value(text: str) -> float | int | str:
    text = text.strip()
    try:
        value = float(text)
    except ValueError:
        return text
    if value.is_integer():
        return int(value)
    return value


def read_input(path: Path) -> dict[str, float | int | str]:
    values: dict[str, float | int | str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = parse_value(value)
    return values


def infer_float(values: dict[str, object], key: str, fallback: float = float("nan")) -> float:
    try:
        return float(values[key])
    except (KeyError, TypeError, ValueError):
        return fallback


def infer_int(values: dict[str, object], key: str, fallback: int = -1) -> int:
    try:
        return int(values[key])
    except (KeyError, TypeError, ValueError):
        return fallback


def natural_sort_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", str(path))
    return [int(part) if part.isdigit() else part for part in parts]


def find_cases(roots: list[Path], include_single: bool) -> list[RunCase]:
    cases: list[RunCase] = []
    for root in roots:
        if not root.exists():
            continue
        for out in sorted(root.rglob("*_body_complete.dat"), key=natural_sort_key):
            values = read_input(out.parent / "input.txt")
            names = set(np.genfromtxt(out, delimiter=",", names=True, max_rows=1).dtype.names or [])
            body_ids = sorted(
                int(match.group(1))
                for name in names
                if (match := re.fullmatch(r"px_(\d+)", name))
            )
            nbody = infer_int(values, "nbody", max(body_ids, default=1))
            if nbody < 2 and not include_single:
                continue
            rho_values = [
                infer_float(values, f"rhos{idx}") for idx in range(1, nbody + 1)
            ]
            finite_rhos = [rho for rho in rho_values if np.isfinite(rho)]
            density = float(np.mean(finite_rhos)) if finite_rhos else float("nan")
            cases.append(
                RunCase(
                    name=out.parent.name,
                    path=out.parent,
                    output=out,
                    nbody=nbody,
                    density=density,
                    ndiv=infer_int(values, "ndiv"),
                    dt=infer_float(values, "dt"),
                    tend=infer_float(values, "tend"),
                    source=str(root.relative_to(ROOT.parent) if ROOT.parent in root.parents else root),
                )
            )
    return cases


def load(case: RunCase) -> np.ndarray:
    data = np.genfromtxt(case.output, delimiter=",", names=True)
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    return data


def case_key(case: RunCase) -> str:
    return str(case.output.relative_to(ROOT))


def vec(data: np.ndarray, prefix: str, body: int) -> np.ndarray:
    return np.column_stack(
        [data[f"{prefix}_x_{body}"], data[f"{prefix}_y_{body}"], data[f"{prefix}_z_{body}"]]
    )


def position(data: np.ndarray, body: int) -> np.ndarray:
    return np.column_stack([data[f"px_{body}"], data[f"py_{body}"], data[f"pz_{body}"]])


def velocity(data: np.ndarray, body: int) -> np.ndarray:
    return np.column_stack([data[f"vx_{body}"], data[f"vy_{body}"], data[f"vz_{body}"]])


def omega(data: np.ndarray, body: int) -> np.ndarray:
    return np.column_stack([data[f"w1_{body}"], data[f"w2_{body}"], data[f"w3_{body}"]])


def total_vec(data: np.ndarray, prefix: str, nbody: int) -> np.ndarray:
    total = np.zeros((len(data), 3))
    for body in range(1, nbody + 1):
        total += vec(data, prefix, body)
    return total


def max_individual_span(data: np.ndarray, prefix: str, nbody: int) -> float:
    return max(max_norm_delta(vec(data, prefix, body)) for body in range(1, nbody + 1))


def max_individual_span_rel(data: np.ndarray, prefix: str, nbody: int) -> float:
    return max(max_norm_delta_rel(vec(data, prefix, body)) for body in range(1, nbody + 1))


def max_norm_delta(values: np.ndarray) -> float:
    return float(np.max(np.linalg.norm(values - values[0], axis=1)))


def max_norm_delta_rel(values: np.ndarray) -> float:
    scale = max(float(np.linalg.norm(values[0])), np.finfo(float).eps)
    return max_norm_delta(values) / scale


def drift_pct(data: np.ndarray) -> np.ndarray:
    return 100.0 * (data["ke_total"] - data["ke_total"][0]) / data["ke_total"][0]


def min_separation(data: np.ndarray, nbody: int) -> tuple[float, float]:
    if nbody < 2:
        return float("nan"), float("nan")
    seps = []
    for a, b in itertools.combinations(range(1, nbody + 1), 2):
        seps.append(np.linalg.norm(position(data, a) - position(data, b), axis=1))
    sep = np.min(np.column_stack(seps), axis=1)
    idx = int(np.argmin(sep))
    return float(sep[idx]), float(data["time"][idx])


def transport_integral(data: np.ndarray, nbody: int) -> np.ndarray:
    """Integral of sum_b v_b x p_b.

    This is the centre-transport term that appears when angular momentum is
    conserved about a fixed origin instead of each body's moving centre.
    """
    integrand = np.zeros((len(data), 3))
    for body in range(1, nbody + 1):
        integrand += np.cross(velocity(data, body), vec(data, "pcon", body))
    dt = np.diff(data["time"])
    increments = 0.5 * (integrand[1:] + integrand[:-1]) * dt[:, None]
    out = np.zeros_like(integrand)
    out[1:] = np.cumsum(increments, axis=0)
    return out


def fluid_ke_from_impulses(data: np.ndarray, nbody: int) -> np.ndarray:
    """Fluid KE from generalized impulse identity.

    With this code's sign convention, positive body velocity gives a negative
    fluid impulse, so K_f = -0.5 * sum_b(v_b.L_b + omega_b.Lambda_b). This uses
    exactly the L/Lambda columns written by the impulse-mode BEM solve, making it
    a direct consistency check against the separately integrated ke_fluid.
    """
    total = np.zeros(len(data))
    for body in range(1, nbody + 1):
        total += np.einsum("ij,ij->i", velocity(data, body), vec(data, "lfluid", body))
        total += np.einsum("ij,ij->i", omega(data, body), vec(data, "lambdafluid", body))
    return -0.5 * total


def discrete_work_balance(data: np.ndarray, nbody: int) -> dict[str, np.ndarray]:
    """Per-step solid-energy residuals against impulse work.

    The linear midpoint update should satisfy
        Delta K_lin = v_mid . Delta L
    with the code's L convention. The angular impulse update uses the
    body-centred angular impulse plus the centre-transport term, so a useful
    diagnostic is
        Delta K_rot ~= omega_mid . (Delta Lambda - dt v_mid x pcon_mid).
    This is not a replacement for the integrator's exact discrete energy law,
    but it localizes whether the observed drift is linear, angular, or coupled.
    """
    linear_work = np.zeros(len(data) - 1)
    angular_work = np.zeros(len(data) - 1)
    linear_delta = np.diff(data["ke_lin_solid"])
    angular_delta = np.diff(data["ke_rot_solid"])
    dt = np.diff(data["time"])

    for body in range(1, nbody + 1):
        v_mid = 0.5 * (velocity(data, body)[1:] + velocity(data, body)[:-1])
        omega_mid = 0.5 * (omega(data, body)[1:] + omega(data, body)[:-1])
        delta_l = np.diff(vec(data, "lfluid", body), axis=0)
        delta_lambda = np.diff(vec(data, "lambdafluid", body), axis=0)
        p_mid = 0.5 * (vec(data, "pcon", body)[1:] + vec(data, "pcon", body)[:-1])
        transport = np.cross(v_mid, p_mid) * dt[:, None]

        linear_work += np.einsum("ij,ij->i", v_mid, delta_l)
        angular_work += np.einsum("ij,ij->i", omega_mid, delta_lambda - transport)

    return {
        "linear_residual": linear_delta - linear_work,
        "angular_residual": angular_delta - angular_work,
        "total_solid_residual": (linear_delta + angular_delta) - (linear_work + angular_work),
        "linear_work": linear_work,
        "angular_work": angular_work,
    }


def corr_abs(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(np.corrcoef(np.abs(a), np.abs(b))[0, 1])


def summarize(case: RunCase, data: np.ndarray) -> dict[str, object]:
    drift = drift_pct(data)
    p_total = total_vec(data, "pcon", case.nbody)
    h_total = total_vec(data, "hcon", case.nbody)
    p_err = np.linalg.norm(p_total - p_total[0], axis=1)
    h_err = np.linalg.norm(h_total - h_total[0], axis=1)
    transport = transport_integral(data, case.nbody)
    ke_fluid_impulse = fluid_ke_from_impulses(data, case.nbody)
    ke_fluid = data["ke_fluid"]
    ke_impulse_err = ke_fluid_impulse - ke_fluid
    ke_impulse_scale = max(float(np.max(np.abs(ke_fluid))), np.finfo(float).eps)
    work = discrete_work_balance(data, case.nbody)
    ke0 = max(float(data["ke_total"][0]), np.finfo(float).eps)
    min_sep, min_sep_t = min_separation(data, case.nbody)
    return {
        "case": case.name,
        "source": case.source,
        "nbody": case.nbody,
        "density_mean": case.density,
        "ndiv": case.ndiv,
        "dt": case.dt,
        "tend": case.tend,
        "rows": len(data),
        "ke0": float(data["ke_total"][0]),
        "drift_end_pct": float(drift[-1]),
        "drift_max_abs_pct": float(np.max(np.abs(drift))),
        "p_total_span": max_norm_delta(p_total),
        "p_total_span_rel": max_norm_delta_rel(p_total),
        "h_total_span": max_norm_delta(h_total),
        "h_total_span_rel": max_norm_delta_rel(h_total),
        "p_individual_max_span": max_individual_span(data, "pcon", case.nbody),
        "p_individual_max_span_rel": max_individual_span_rel(data, "pcon", case.nbody),
        "h_individual_max_span": max_individual_span(data, "hcon", case.nbody),
        "h_individual_max_span_rel": max_individual_span_rel(data, "hcon", case.nbody),
        "transport_integral_span": max_norm_delta(transport),
        "fluid_ke_impulse_mean_rel_err": float(np.mean(ke_impulse_err) / ke_impulse_scale),
        "fluid_ke_impulse_max_rel_err": float(np.max(np.abs(ke_impulse_err)) / ke_impulse_scale),
        "fluid_ke_impulse_err_span_rel": max_norm_delta(ke_impulse_err[:, None]) / ke_impulse_scale,
        "corr_abs_energy_fluid_impulse_err": corr_abs(drift, ke_impulse_err),
        "linear_work_residual_sum_pct": float(100.0 * np.sum(work["linear_residual"]) / ke0),
        "angular_work_residual_sum_pct": float(100.0 * np.sum(work["angular_residual"]) / ke0),
        "solid_work_residual_sum_pct": float(100.0 * np.sum(work["total_solid_residual"]) / ke0),
        "linear_work_residual_l1_pct": float(100.0 * np.sum(np.abs(work["linear_residual"])) / ke0),
        "angular_work_residual_l1_pct": float(100.0 * np.sum(np.abs(work["angular_residual"])) / ke0),
        "corr_abs_energy_p": corr_abs(drift, p_err),
        "corr_abs_energy_h": corr_abs(drift, h_err),
        "min_separation": min_sep,
        "min_separation_time": min_sep_t,
        "output": str(case.output.relative_to(ROOT)),
    }


def write_summary(rows: list[dict[str, object]]) -> Path:
    out = STUDY / "multibody_energy_balance_summary.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out


def plot_case_timeseries(axis: plt.Axes, case: RunCase, data: np.ndarray, title: str) -> None:
    t = data["time"]
    p_total = total_vec(data, "pcon", case.nbody)
    h_total = total_vec(data, "hcon", case.nbody)
    p_err = np.linalg.norm(p_total - p_total[0], axis=1)
    h_err = np.linalg.norm(h_total - h_total[0], axis=1)
    ke_impulse = fluid_ke_from_impulses(data, case.nbody)
    ke_scale = max(float(data["ke_total"][0]), np.finfo(float).eps)
    axis.plot(t, drift_pct(data), lw=1.2, label="KE drift (%)")
    axis.plot(t, 100.0 * p_err / max(np.linalg.norm(p_total[0]), np.finfo(float).eps), lw=1.0, label="P drift (%)")
    axis.plot(t, 100.0 * h_err / max(np.linalg.norm(h_total[0]), np.finfo(float).eps), lw=1.0, label="H drift (%)")
    axis.plot(
        t,
        100.0 * (ke_impulse - data["ke_fluid"]) / ke_scale,
        lw=1.0,
        ls=":",
        label="fluid KE identity err (%)",
    )
    axis.axhline(0.0, color="k", lw=0.7)
    axis.set(title=title, xlabel="t", ylabel="relative change")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=7)


def cumulative_step(values: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(values)))


def plot_dashboard(cases: list[RunCase], data_by_case: dict[str, np.ndarray], rows: list[dict[str, object]]) -> Path:
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    rows_sorted = sorted(rows, key=lambda row: (str(row["source"]), int(row["nbody"]), float(row["density_mean"]), int(row["ndiv"]), float(row["dt"])))
    labels = [str(row["case"]) for row in rows_sorted]
    y = np.arange(len(rows_sorted))

    ax[0, 0].barh(y, [float(row["drift_max_abs_pct"]) for row in rows_sorted], color="tab:purple")
    ax[0, 0].set_yticks(y)
    ax[0, 0].set_yticklabels(labels, fontsize=7)
    ax[0, 0].invert_yaxis()
    ax[0, 0].set(title="Max total-energy drift", xlabel="percent")
    ax[0, 0].grid(alpha=0.3, axis="x")

    ax[0, 1].scatter(
        [float(row["h_total_span_rel"]) for row in rows_sorted],
        [float(row["drift_max_abs_pct"]) for row in rows_sorted],
        c=[int(row["ndiv"]) for row in rows_sorted],
        cmap="viridis",
        s=55,
    )
    ax[0, 1].set_xscale("log")
    ax[0, 1].set(title="Energy drift vs global H drift", xlabel="relative H span", ylabel="max |KE drift| (%)")
    ax[0, 1].grid(alpha=0.3)

    scatter = ax[0, 2].scatter(
        [float(row["p_total_span_rel"]) for row in rows_sorted],
        [float(row["drift_max_abs_pct"]) for row in rows_sorted],
        c=[int(row["ndiv"]) for row in rows_sorted],
        cmap="viridis",
        s=55,
    )
    ax[0, 2].set_xscale("log")
    ax[0, 2].set(title="Energy drift vs global P drift", xlabel="relative P span", ylabel="max |KE drift| (%)")
    ax[0, 2].grid(alpha=0.3)
    cbar = fig.colorbar(scatter, ax=ax[0, 2], fraction=0.046, pad=0.04)
    cbar.set_label("ndiv")

    ax[1, 0].scatter(
        [float(row["fluid_ke_impulse_err_span_rel"]) for row in rows_sorted],
        [float(row["drift_max_abs_pct"]) for row in rows_sorted],
        c=[int(row["nbody"]) for row in rows_sorted],
        cmap="plasma",
        s=55,
    )
    ax[1, 0].set_xscale("log")
    ax[1, 0].set(
        title="Energy drift vs fluid KE identity error",
        xlabel="relative span of K_impulse - K_integral",
        ylabel="max |KE drift| (%)",
    )
    ax[1, 0].grid(alpha=0.3)

    representative = sorted(
        cases,
        key=lambda case: (
            -case.nbody,
            case.density if np.isfinite(case.density) else 1e9,
            -case.ndiv,
            case.dt,
        ),
    )[:2]
    for idx, case in enumerate(representative):
        data = data_by_case[case_key(case)]
        if idx == 0:
            work = discrete_work_balance(data, case.nbody)
            ke0 = max(float(data["ke_total"][0]), np.finfo(float).eps)
            t = data["time"]
            ax[1, 1].plot(
                t,
                100.0 * cumulative_step(work["linear_residual"]) / ke0,
                lw=1.1,
                label="linear residual",
            )
            ax[1, 1].plot(
                t,
                100.0 * cumulative_step(work["angular_residual"]) / ke0,
                lw=1.1,
                label="angular residual",
            )
            ax[1, 1].plot(t, drift_pct(data), lw=1.2, ls="--", label="total KE drift")
            ax[1, 1].set(title=f"Work residual split ({case.name})", xlabel="t", ylabel="% initial KE")
            ax[1, 1].grid(alpha=0.3)
            ax[1, 1].legend(fontsize=7)
        else:
            plot_case_timeseries(ax[1, 2], case, data, case.name)

    fig.suptitle("Multi-body energy balance diagnostics", fontsize=14)
    fig.tight_layout()
    out = STUDY / "multibody_energy_balance_dashboard.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    return out


def print_summary(rows: list[dict[str, object]]) -> None:
    print(
        f"{'case':<24}{'n':>3}{'rho':>8}{'nd':>4}{'dt':>8}"
        f"{'max KE%':>11}{'P rel':>11}{'H rel':>11}{'Kimp err':>11}"
        f"{'P_i rel':>11}{'lin W%':>10}{'ang W%':>10}{'min sep':>10}"
    )
    for row in sorted(rows, key=lambda r: (str(r["source"]), int(r["nbody"]), float(r["density_mean"]), int(r["ndiv"]), float(r["dt"]))):
        print(
            f"{str(row['case']):<24}{int(row['nbody']):>3}"
            f"{float(row['density_mean']):>8.3g}{int(row['ndiv']):>4}{float(row['dt']):>8.3g}"
            f"{float(row['drift_max_abs_pct']):>11.4g}"
            f"{float(row['p_total_span_rel']):>11.3e}"
            f"{float(row['h_total_span_rel']):>11.3e}"
            f"{float(row['fluid_ke_impulse_err_span_rel']):>11.3e}"
            f"{float(row['p_individual_max_span_rel']):>11.3e}"
            f"{float(row['linear_work_residual_sum_pct']):>10.3g}"
            f"{float(row['angular_work_residual_sum_pct']):>10.3g}"
            f"{float(row['min_separation']):>10.3g}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize multi-body energy drift against global P/H conservation diagnostics."
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        type=Path,
        default=DEFAULT_ROOTS,
        help="Run directories to scan for *_body_complete.dat outputs.",
    )
    parser.add_argument("--include-single", action="store_true", help="include single-body outputs too")
    args = parser.parse_args()

    STUDY.mkdir(parents=True, exist_ok=True)
    roots = [root if root.is_absolute() else ROOT / root for root in args.roots]
    cases = find_cases(roots, args.include_single)
    if not cases:
        raise SystemExit("No matching outputs found.")

    data_by_case = {case_key(case): load(case) for case in cases}
    rows = [summarize(case, data_by_case[case_key(case)]) for case in cases]
    summary = write_summary(rows)
    dashboard = plot_dashboard(cases, data_by_case, rows)
    print_summary(rows)
    print(f"\nSaved {summary.relative_to(ROOT)}")
    print(f"Saved {dashboard.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
