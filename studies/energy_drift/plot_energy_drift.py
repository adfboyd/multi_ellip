from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import exact_added_mass as exact  # noqa: E402

STUDY = ROOT / "studies" / "energy_drift"
RUNS = STUDY / "runs"

CASES = [
    {"name": "nd2_dt0p2", "ndiv": 2, "dt": 0.2, "group": "mesh"},
    {"name": "nd3_dt0p2", "ndiv": 3, "dt": 0.2, "group": "mesh"},
    {"name": "nd4_dt0p2", "ndiv": 4, "dt": 0.2, "group": "mesh"},
    {"name": "nd2_dt0p1", "ndiv": 2, "dt": 0.1, "group": "dt_check"},
    {"name": "nd2_dt0p05", "ndiv": 2, "dt": 0.05, "group": "dt_check"},
]

COL = {
    "t": "time",
    "ke_total": "ke_total",
    "ke_fluid": "ke_fluid",
    "ke_solid": "ke_solid",
    "ke_lin": "ke_lin_solid",
    "ke_rot": "ke_rot_solid",
    "px": "px_1",
    "py": "py_1",
    "pz": "pz_1",
    "vx": "vx_1",
    "vy": "vy_1",
    "vz": "vz_1",
    "q1": "q1_1",
    "q2": "q2_1",
    "q3": "q3_1",
    "q0": "q0_1",
    "w1": "w1_1",
    "w2": "w2_1",
    "w3": "w3_1",
    "ofx": "ofix1_1",
    "ofy": "ofix2_1",
    "ofz": "ofix3_1",
    "pcon_x": "pcon_x_1",
    "pcon_y": "pcon_y_1",
    "pcon_z": "pcon_z_1",
    "hcon_x": "hcon_x_1",
    "hcon_y": "hcon_y_1",
    "hcon_z": "hcon_z_1",
}


def load_case(case: dict[str, object]) -> np.ndarray:
    path = RUNS / str(case["name"]) / "single_body_complete.dat"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    return data


def drift_pct(data: np.ndarray) -> np.ndarray:
    ke = data[COL["ke_total"]]
    return 100.0 * (ke - ke[0]) / ke[0]


def speed(data: np.ndarray) -> np.ndarray:
    return np.sqrt(data[COL["vx"]] ** 2 + data[COL["vy"]] ** 2 + data[COL["vz"]] ** 2)


def invariant_span(data: np.ndarray, cols: tuple[str, str, str]) -> float:
    if not all(col in data.dtype.names for col in cols):
        return float("nan")
    values = np.column_stack([data[col] for col in cols])
    return float(np.max(np.linalg.norm(values - values[0], axis=1)))


def analytic_energy_at_state(case: dict[str, object], data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals = exact.parse_input(RUNS / str(case["name"]) / "input.txt")
    bp = exact.body_params(vals)
    tensors = exact.tensors(bp["abc"], bp["rho_s"], vals.get("rhof", 1.0))
    m_s = tensors["m_s"]
    i_s = tensors["I_s"]
    m_a = tensors["M_a"]
    i_a = tensors["I_a"]

    ke_fluid = np.zeros(len(data))
    ke_total = np.zeros(len(data))
    for i, row in enumerate(data):
        q = np.array([row[COL["q0"]], row[COL["q1"]], row[COL["q2"]], row[COL["q3"]]], dtype=float)
        q /= np.linalg.norm(q)
        rot = exact.quat_to_R(q)
        u_lab = np.array([row[COL["vx"]], row[COL["vy"]], row[COL["vz"]]], dtype=float)
        w_lab = np.array([row[COL["w1"]], row[COL["w2"]], row[COL["w3"]]], dtype=float)
        u_body = rot.T @ u_lab
        w_body = rot.T @ w_lab
        solid = 0.5 * m_s * float(u_lab @ u_lab) + 0.5 * float(w_body @ i_s @ w_body)
        fluid = 0.5 * float(u_body @ m_a @ u_body) + 0.5 * float(w_body @ i_a @ w_body)
        ke_fluid[i] = fluid
        ke_total[i] = solid + fluid
    return ke_total, ke_fluid


def pct_drift(values: np.ndarray) -> np.ndarray:
    return 100.0 * (values - values[0]) / values[0]


def summarize(results: dict[str, np.ndarray]) -> list[dict[str, object]]:
    rows = []
    for case in CASES:
        data = results[str(case["name"])]
        drift = drift_pct(data)
        exact_total, _ = analytic_energy_at_state(case, data)
        exact_drift = pct_drift(exact_total)
        rows.append(
            {
                "case": case["name"],
                "ndiv": case["ndiv"],
                "dt": case["dt"],
                "rows": len(data),
                "ke0": data[COL["ke_total"]][0],
                "ke_end": data[COL["ke_total"]][-1],
                "drift_end_pct": drift[-1],
                "drift_max_abs_pct": np.max(np.abs(drift)),
                "analytic_state_drift_max_abs_pct": np.max(np.abs(exact_drift)),
                "pcon_span": invariant_span(data, (COL["pcon_x"], COL["pcon_y"], COL["pcon_z"])),
                "hcon_span": invariant_span(data, (COL["hcon_x"], COL["hcon_y"], COL["hcon_z"])),
                "speed_min": np.min(speed(data)),
                "speed_max": np.max(speed(data)),
            }
        )
    return rows


def write_summary(rows: list[dict[str, object]]) -> None:
    out = STUDY / "energy_drift_summary.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out.relative_to(ROOT)}")

    print()
    print("Energy drift summary")
    print("case          ndiv  dt     rows  end drift %  max |drift| %  analytic max %")
    for row in rows:
        print(
            f"{row['case']:<13} {row['ndiv']:>4}  {row['dt']:<5} "
            f"{row['rows']:>5}  {row['drift_end_pct']:>11.5f}  {row['drift_max_abs_pct']:>13.5f}"
            f"  {row['analytic_state_drift_max_abs_pct']:>14.5f}"
        )


def main() -> None:
    STUDY.mkdir(parents=True, exist_ok=True)
    results = {str(case["name"]): load_case(case) for case in CASES}
    rows = summarize(results)
    write_summary(rows)

    mesh_cases = [case for case in CASES if case["group"] == "mesh"]
    dt_cases = [
        {"name": "nd2_dt0p2", "ndiv": 2, "dt": 0.2, "group": "dt_check"},
        *[case for case in CASES if case["group"] == "dt_check"],
    ]
    colors = {"nd2_dt0p2": "C0", "nd3_dt0p2": "C1", "nd4_dt0p2": "C2", "nd2_dt0p1": "C3", "nd2_dt0p05": "C4"}

    fig, ax = plt.subplots(3, 3, figsize=(17, 13))

    for case in mesh_cases:
        data = results[str(case["name"])]
        label = f"ndiv={case['ndiv']}, dt={case['dt']}"
        ax[0, 0].plot(data[COL["t"]], drift_pct(data), color=colors[str(case["name"])], lw=1.4, label=label)
    ax[0, 0].axhline(0, color="k", lw=0.8)
    ax[0, 0].set(title="Energy drift vs mesh (single body, impulse scheme)", xlabel="t", ylabel="drift (%)")
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].grid(alpha=0.3)

    ndivs = [row["ndiv"] for row in rows if row["case"] in [case["name"] for case in mesh_cases]]
    final_drift = [row["drift_end_pct"] for row in rows if row["case"] in [case["name"] for case in mesh_cases]]
    max_drift = [row["drift_max_abs_pct"] for row in rows if row["case"] in [case["name"] for case in mesh_cases]]
    width = 0.35
    x = np.arange(len(ndivs))
    ax[0, 1].bar(x - width / 2, final_drift, width, label="final drift")
    ax[0, 1].bar(x + width / 2, max_drift, width, label="max |drift|")
    ax[0, 1].set(title="Drift summary by mesh", xlabel="ndiv", ylabel="percent")
    ax[0, 1].set_xticks(x, [str(n) for n in ndivs])
    ax[0, 1].legend(fontsize=8)
    ax[0, 1].grid(axis="y", alpha=0.3)

    for case in dt_cases:
        data = results[str(case["name"])]
        label = f"dt={case['dt']}"
        ax[0, 2].plot(data[COL["t"]], drift_pct(data), color=colors[str(case["name"])], lw=1.2, label=label)
    ax[0, 2].axhline(0, color="k", lw=0.8)
    ax[0, 2].set(title="dt sensitivity check at ndiv=2", xlabel="t", ylabel="drift (%)")
    ax[0, 2].legend(fontsize=8)
    ax[0, 2].grid(alpha=0.3)

    ref = results["nd2_dt0p2"]
    ref_case = next(case for case in CASES if case["name"] == "nd2_dt0p2")
    ref_exact_total, ref_exact_fluid = analytic_energy_at_state(ref_case, ref)
    ax[1, 0].plot(ref[COL["t"]], ref[COL["ke_total"]], lw=1.4, label="total")
    ax[1, 0].plot(ref[COL["t"]], ref[COL["ke_fluid"]], lw=1.0, label="fluid")
    ax[1, 0].plot(ref[COL["t"]], ref[COL["ke_solid"]], lw=1.0, label="solid")
    ax[1, 0].plot(ref[COL["t"]], ref_exact_total, lw=1.0, ls=":", label="analytic-at-state total")
    ax[1, 0].plot(ref[COL["t"]], ref_exact_fluid, lw=1.0, ls=":", label="analytic-at-state fluid")
    ax[1, 0].plot(ref[COL["t"]], ref[COL["ke_lin"]], lw=0.9, ls="--", label="solid linear")
    ax[1, 0].plot(ref[COL["t"]], ref[COL["ke_rot"]], lw=0.9, ls="--", label="solid rotational")
    ax[1, 0].set(title="Energy components (ndiv=2, dt=0.2)", xlabel="t", ylabel="KE")
    ax[1, 0].legend(fontsize=7, ncol=2)
    ax[1, 0].grid(alpha=0.3)

    for case in mesh_cases:
        data = results[str(case["name"])]
        ax[1, 1].plot(data[COL["t"]], speed(data), color=colors[str(case["name"])], lw=1.2, label=f"ndiv={case['ndiv']}")
    ax[1, 1].set(title="Speed magnitude vs mesh", xlabel="t", ylabel="|v|")
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].grid(alpha=0.3)

    for comp, label in ((COL["vx"], "vx"), (COL["vy"], "vy"), (COL["vz"], "vz")):
        ax[1, 2].plot(ref[COL["t"]], ref[comp], lw=1.2, label=label)
    ax[1, 2].set(title="Velocity components (ndiv=2, dt=0.2)", xlabel="t", ylabel="velocity")
    ax[1, 2].legend(fontsize=8)
    ax[1, 2].grid(alpha=0.3)

    ax_path = fig.add_subplot(3, 3, 7, projection="3d")
    for case in mesh_cases:
        data = results[str(case["name"])]
        ax_path.plot(data[COL["px"]], data[COL["py"]], data[COL["pz"]], color=colors[str(case["name"])], lw=1.0, label=f"ndiv={case['ndiv']}")
    ax_path.set(title="Centre path", xlabel="x", ylabel="y", zlabel="z")
    ax_path.legend(fontsize=7)

    ax_orient = fig.add_subplot(3, 3, 8, projection="3d")
    for case in mesh_cases:
        data = results[str(case["name"])]
        ax_orient.plot(data[COL["ofx"]], data[COL["ofy"]], data[COL["ofz"]], color=colors[str(case["name"])], lw=1.0, label=f"ndiv={case['ndiv']}")
    ax_orient.set(title="Orientation marker path", xlabel="ofix1", ylabel="ofix2", zlabel="ofix3")
    ax_orient.legend(fontsize=7)

    if all(col in ref.dtype.names for col in (COL["pcon_x"], COL["pcon_y"], COL["pcon_z"], COL["hcon_x"], COL["hcon_y"], COL["hcon_z"])):
        p_values = np.column_stack([ref[COL["pcon_x"]], ref[COL["pcon_y"]], ref[COL["pcon_z"]]])
        h_values = np.column_stack([ref[COL["hcon_x"]], ref[COL["hcon_y"]], ref[COL["hcon_z"]]])
        ax[2, 2].plot(ref[COL["t"]], np.linalg.norm(p_values - p_values[0], axis=1), lw=1.2, label="|P-P0|")
        ax[2, 2].plot(ref[COL["t"]], np.linalg.norm(h_values - h_values[0], axis=1), lw=1.2, label="|H-H0|")
        ax[2, 2].set(title="Impulse momentum invariant errors (ndiv=2)", xlabel="t", ylabel="absolute error")
        ax[2, 2].legend(fontsize=8)
        ax[2, 2].grid(alpha=0.3)
    else:
        marker_norm = np.sqrt(ref[COL["ofx"]] ** 2 + ref[COL["ofy"]] ** 2 + ref[COL["ofz"]] ** 2)
        ax[2, 2].plot(ref[COL["t"]], marker_norm - 1.0, lw=1.1, label="ndiv=2")
        ax[2, 2].set(title="Orientation marker norm error", xlabel="t", ylabel="|ofix| - 1")
        ax[2, 2].legend(fontsize=8)
        ax[2, 2].grid(alpha=0.3)

    fig.suptitle("Single ellipsoid energy drift study, t=0..50, impulse scheme", fontsize=14)
    fig.tight_layout()
    out = STUDY / "energy_drift_dashboard.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
