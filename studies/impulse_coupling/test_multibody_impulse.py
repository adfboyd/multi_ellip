from __future__ import annotations

import csv
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "studies" / "impulse_coupling"
RUNS = STUDY / "runs"


BASE_INPUT = {
    "cex1": 4.0,
    "cey1": 0.0,
    "cez1": 0.0,
    "oriw1": 1.0,
    "orii1": 2.0,
    "orij1": 0.0,
    "orik1": 0.0,
    "lvx1": -0.4,
    "lvy1": 0.1,
    "lvz1": 0.0,
    "avx1": 1.0,
    "avy1": 1.0,
    "avz1": 0.0,
    "shx1": 1.0,
    "shy1": 0.8,
    "shz1": 0.6,
    "req1": 1.0,
    "rhos1": 1.0,
    "cex2": -4.0,
    "cey2": 1.0,
    "cez2": 0.0,
    "oriw2": 1.0,
    "orii2": 0.0,
    "orij2": 1.0,
    "orik2": 0.0,
    "lvx2": -0.4,
    "lvy2": 0.1,
    "lvz2": 0.0,
    "avx2": 0.0,
    "avy2": 1.0,
    "avz2": 1.0,
    "shx2": 1.0,
    "shy2": 0.8,
    "shz2": 0.6,
    "req2": 1.0,
    "rhos2": 1.0,
    "rhof": 1.0,
    "ndiv": 2,
    "tend": 50.0,
    "dt": 0.2,
    "tprint": 1,
    "logevery": 10,
    "nbody": 2,
    "impulse_scheme": 1,
}

CASES = [
    {"name": "two_body_dt0p2", "dt": 0.2},
    {"name": "two_body_dt0p1", "dt": 0.1},
]


def fmt_hms(seconds: float) -> str:
    seconds = int(round(seconds))
    m, s = divmod(seconds, 60)
    return f"{m}m {s}s" if m else f"{s}s"


def write_input(case: dict[str, object]) -> Path:
    case_dir = RUNS / str(case["name"])
    case_dir.mkdir(parents=True, exist_ok=True)
    input_path = case_dir / "input.txt"
    values = dict(BASE_INPUT)
    values["dt"] = case["dt"]
    with input_path.open("w", encoding="utf-8", newline="\n") as f:
        for key, value in values.items():
            f.write(f"{key}={value}\n")
    return input_path


def stream_run(cmd: list[str], log_path: Path | None = None) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    log = log_path.open("w", encoding="utf-8", newline="\n") if log_path else None
    started = time.monotonic()
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            if log:
                log.write(line)
                log.flush()
        rc = proc.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)
    finally:
        if log:
            log.close()
    print(f"Finished in {fmt_hms(time.monotonic() - started)}", flush=True)


def load(case: dict[str, object]) -> np.ndarray:
    path = RUNS / str(case["name"]) / "multiple_body_complete.dat"
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    return data


def vector(data: np.ndarray, prefix: str, body: int) -> np.ndarray:
    return np.column_stack(
        [data[f"{prefix}_x_{body}"], data[f"{prefix}_y_{body}"], data[f"{prefix}_z_{body}"]]
    )


def position(data: np.ndarray, body: int) -> np.ndarray:
    return np.column_stack(
        [data[f"px_{body}"], data[f"py_{body}"], data[f"pz_{body}"]]
    )


def ofix(data: np.ndarray, body: int) -> np.ndarray:
    return np.column_stack(
        [
            data[f"ofix1_{body}"],
            data[f"ofix2_{body}"],
            data[f"ofix3_{body}"],
        ]
    )


def span(values: np.ndarray) -> float:
    return float(np.max(np.linalg.norm(values - values[0], axis=1)))


def drift_pct(data: np.ndarray) -> np.ndarray:
    ke = data["ke_total"]
    return 100.0 * (ke - ke[0]) / ke[0]


def summarize(case: dict[str, object], data: np.ndarray) -> dict[str, object]:
    drift = drift_pct(data)
    p1 = vector(data, "pcon", 1)
    p2 = vector(data, "pcon", 2)
    h1 = vector(data, "hcon", 1)
    h2 = vector(data, "hcon", 2)
    row = {
        "case": case["name"],
        "dt": case["dt"],
        "rows": len(data),
        "ke0": data["ke_total"][0],
        "ke_end": data["ke_total"][-1],
        "drift_end_pct": drift[-1],
        "drift_max_abs_pct": np.max(np.abs(drift)),
        "p1_span": span(p1),
        "p2_span": span(p2),
        "ptotal_span": span(p1 + p2),
        "h1_span": span(h1),
        "h2_span": span(h2),
        "htotal_span": span(h1 + h2),
    }
    return row


def write_summary(rows: list[dict[str, object]]) -> None:
    out = STUDY / "multibody_impulse_summary.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out.relative_to(ROOT)}")
    print(
        "case              dt    end drift %  max |drift| %    P total span    H total span"
    )
    for row in rows:
        print(
            f"{row['case']:<16} {row['dt']:<5} {row['drift_end_pct']:>12.5f}"
            f" {row['drift_max_abs_pct']:>14.5f}"
            f" {row['ptotal_span']:>15.4e} {row['htotal_span']:>15.4e}"
        )


def plot(results: dict[str, np.ndarray]) -> None:
    fig = plt.figure(figsize=(18, 14))
    ax_drift = fig.add_subplot(3, 3, 1)
    ax_ke = fig.add_subplot(3, 3, 2)
    ax_mom = fig.add_subplot(3, 3, 3)
    ax_xy = fig.add_subplot(3, 3, 4)
    ax_path3d = fig.add_subplot(3, 3, 5, projection="3d")
    ax_coords = fig.add_subplot(3, 3, 6)
    ax_sep = fig.add_subplot(3, 3, 7)
    ax_ofix1 = fig.add_subplot(3, 3, 8, projection="3d")
    ax_ofix2 = fig.add_subplot(3, 3, 9, projection="3d")

    for idx, case in enumerate(CASES):
        name = str(case["name"])
        data = results[name]
        t = data["time"]
        x1 = position(data, 1)
        x2 = position(data, 2)
        rel = x1 - x2
        sep = np.linalg.norm(rel, axis=1)
        ofix1 = ofix(data, 1)
        ofix2 = ofix(data, 2)
        p1 = vector(data, "pcon", 1)
        p2 = vector(data, "pcon", 2)
        h1 = vector(data, "hcon", 1)
        h2 = vector(data, "hcon", 2)
        label = f"dt={case['dt']}"

        ax_drift.plot(t, drift_pct(data), lw=1.3, label=label)
        ax_ke.plot(t, data["ke_total"], lw=1.2, label=label)
        ax_mom.plot(
            t,
            np.linalg.norm(p1 + p2 - (p1[0] + p2[0]), axis=1),
            lw=1.4,
            label=f"{label} |P-P0|",
        )
        ax_mom.plot(
            t,
            np.linalg.norm(h1 + h2 - (h1[0] + h2[0]), axis=1),
            lw=1.4,
            ls="--",
            label=f"{label} |H-H0|",
        )

        ax_xy.plot(x1[:, 0], x1[:, 1], lw=1.2, color=f"C{idx}", label=f"{label} body 1")
        ax_xy.plot(x2[:, 0], x2[:, 1], lw=1.2, ls="--", color=f"C{idx}", label=f"{label} body 2")
        ax_xy.scatter([x1[0, 0], x2[0, 0]], [x1[0, 1], x2[0, 1]], color=f"C{idx}", s=16)

        ax_path3d.plot(x1[:, 0], x1[:, 1], x1[:, 2], lw=1.0, color=f"C{idx}", label=f"{label} body 1")
        ax_path3d.plot(x2[:, 0], x2[:, 1], x2[:, 2], lw=1.0, ls="--", color=f"C{idx}", label=f"{label} body 2")

        if idx == 0:
            ax_coords.plot(t, x1[:, 0], lw=1.0, color="C0", label="body 1 x")
            ax_coords.plot(t, x1[:, 1], lw=1.0, color="C1", label="body 1 y")
            ax_coords.plot(t, x1[:, 2], lw=1.0, color="C2", label="body 1 z")
            ax_coords.plot(t, x2[:, 0], lw=1.0, ls="--", color="C0", label="body 2 x")
            ax_coords.plot(t, x2[:, 1], lw=1.0, ls="--", color="C1", label="body 2 y")
            ax_coords.plot(t, x2[:, 2], lw=1.0, ls="--", color="C2", label="body 2 z")

        ax_sep.plot(t, sep, lw=1.3, label=label)
        ax_ofix1.plot(ofix1[:, 0], ofix1[:, 1], ofix1[:, 2], lw=1.2, color=f"C{idx}", label=label)
        ax_ofix1.scatter(ofix1[0, 0], ofix1[0, 1], ofix1[0, 2], color=f"C{idx}", s=18)
        ax_ofix2.plot(ofix2[:, 0], ofix2[:, 1], ofix2[:, 2], lw=1.2, color=f"C{idx}", label=label)
        ax_ofix2.scatter(ofix2[0, 0], ofix2[0, 1], ofix2[0, 2], color=f"C{idx}", s=18)

    ax_drift.axhline(0, color="k", lw=0.8)
    ax_drift.set(title="Total KE drift", xlabel="t", ylabel="drift (%)")
    ax_drift.legend(fontsize=8)
    ax_drift.grid(alpha=0.3)

    ax_ke.set(title="Total energy", xlabel="t", ylabel="KE")
    ax_ke.legend(fontsize=8)
    ax_ke.grid(alpha=0.3)

    ax_mom.set(title="System momentum invariant errors", xlabel="t", ylabel="absolute error")
    ax_mom.legend(fontsize=7)
    ax_mom.grid(alpha=0.3)

    ax_xy.set(title="Centre paths in xy", xlabel="x", ylabel="y")
    ax_xy.axis("equal")
    ax_xy.legend(fontsize=7)
    ax_xy.grid(alpha=0.3)

    ax_path3d.set(title="Centre paths in 3D", xlabel="x", ylabel="y", zlabel="z")
    ax_path3d.legend(fontsize=6)

    ax_coords.set(title="Body coordinates over time (dt=0.2)", xlabel="t", ylabel="position")
    ax_coords.legend(fontsize=7, ncol=2)
    ax_coords.grid(alpha=0.3)

    ax_sep.set(title="Centre separation", xlabel="t", ylabel="|x1 - x2|")
    ax_sep.legend(fontsize=8)
    ax_sep.grid(alpha=0.3)

    ax_ofix1.set(title="Body 1 ofix path", xlabel="ofix1", ylabel="ofix2", zlabel="ofix3")
    ax_ofix1.legend(fontsize=8)

    ax_ofix2.set(title="Body 2 ofix path", xlabel="ofix1", ylabel="ofix2", zlabel="ofix3")
    ax_ofix2.legend(fontsize=8)

    fig.suptitle("Multi-body impulse diagnostic", fontsize=13)
    fig.tight_layout()
    out = STUDY / "multibody_impulse_dashboard.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved {out.relative_to(ROOT)}")


def main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    stream_run(["cargo", "build", "--release"])
    exe = ROOT / "target" / "release" / "multi_ellip.exe"
    if not exe.exists():
        exe = ROOT / "target" / "release" / "multi_ellip"

    results = {}
    rows = []
    for idx, case in enumerate(CASES, start=1):
        print()
        print("=" * 72)
        print(f"Case {idx}/{len(CASES)}: {case['name']} (dt={case['dt']})")
        print("=" * 72)
        input_path = write_input(case)
        out_dir = RUNS / str(case["name"])
        stream_run([str(exe), str(input_path), str(out_dir)], out_dir / "run.log")
        data = load(case)
        results[str(case["name"])] = data
        rows.append(summarize(case, data))

    write_summary(rows)
    plot(results)


if __name__ == "__main__":
    main()
