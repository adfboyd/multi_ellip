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
    "cex1": 0.0,
    "cey1": 0.0,
    "cez1": 0.0,
    "oriw1": 1.0,
    "orii1": 2.0,
    "orij1": 0.0,
    "orik1": 0.0,
    "shx1": 1.0,
    "shy1": 0.8,
    "shz1": 0.6,
    "req1": 1.0,
    "rhos1": 1.0,
    "rhof": 1.0,
    "ndiv": 2,
    "tend": 20.0,
    "dt": 0.2,
    "tprint": 1,
    "logevery": 25,
    "nbody": 1,
    "impulse_scheme": 1,
}

CASES = [
    {
        "name": "translation_default",
        "lv": (-1.0, 0.0, 0.0),
        "av": (0.0, 0.0, 0.0),
    },
    {
        "name": "rotation_default",
        "lv": (0.0, 0.0, 0.0),
        "av": (1.0, 1.0, 0.0),
    },
    {
        "name": "both_default",
        "lv": (-1.0, 0.0, 0.0),
        "av": (1.0, 1.0, 0.0),
    },
]


def fmt_hms(seconds: float) -> str:
    seconds = int(round(seconds))
    m, s = divmod(seconds, 60)
    return f"{m}m {s}s" if m else f"{s}s"


def write_input(case: dict[str, object]) -> Path:
    case_dir = RUNS / str(case["name"])
    case_dir.mkdir(parents=True, exist_ok=True)
    values = dict(BASE_INPUT)
    lvx, lvy, lvz = case["lv"]
    avx, avy, avz = case["av"]
    values.update(
        {
            "lvx1": lvx,
            "lvy1": lvy,
            "lvz1": lvz,
            "avx1": avx,
            "avy1": avy,
            "avz1": avz,
        }
    )
    input_path = case_dir / "input.txt"
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
        rc = proc.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)
    finally:
        if log:
            log.close()
    print(f"Finished in {fmt_hms(time.monotonic() - started)}", flush=True)


def load(case: dict[str, object]) -> np.ndarray:
    return np.genfromtxt(
        RUNS / str(case["name"]) / "single_body_complete.dat",
        delimiter=",",
        names=True,
    )


def drift(data: np.ndarray) -> np.ndarray:
    ke = data["ke_total"]
    return 100.0 * (ke - ke[0]) / ke[0]


def invariant_span(data: np.ndarray, prefix: str) -> float:
    cols = tuple(f"{prefix}_{axis}_1" for axis in ("x", "y", "z"))
    if not all(col in data.dtype.names for col in cols):
        return float("nan")
    values = np.column_stack([data[col] for col in cols])
    return float(np.max(np.linalg.norm(values - values[0], axis=1)))


def summarize() -> list[dict[str, object]]:
    rows = []
    for case in CASES:
        data = load(case)
        d = drift(data)
        rows.append(
            {
                "case": case["name"],
                "lv": case["lv"],
                "av": case["av"],
                "rows": len(data),
                "ke0": data["ke_total"][0],
                "ke_end": data["ke_total"][-1],
                "drift_end_pct": d[-1],
                "drift_max_abs_pct": np.max(np.abs(d)),
                "solid_range": data["ke_solid"].max() - data["ke_solid"].min(),
                "fluid_range": data["ke_fluid"].max() - data["ke_fluid"].min(),
                "pcon_span": invariant_span(data, "pcon"),
                "hcon_span": invariant_span(data, "hcon"),
            }
        )
    return rows


def write_summary(rows: list[dict[str, object]]) -> None:
    out = STUDY / "impulse_coupling_summary.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out.relative_to(ROOT)}")
    print("case                    end drift %  max |drift| %      P span      H span")
    for row in rows:
        print(
            f"{row['case']:<24} {row['drift_end_pct']:>12.5f}"
            f" {row['drift_max_abs_pct']:>14.5f}"
            f" {row['pcon_span']:>11.4e} {row['hcon_span']:>11.4e}"
        )


def plot(rows: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    for case in CASES:
        data = load(case)
        ax[0, 0].plot(data["time"], drift(data), lw=1.2, label=case["name"])
    ax[0, 0].axhline(0, color="k", lw=0.8)
    ax[0, 0].set(title="Total KE drift", xlabel="t", ylabel="drift (%)")
    ax[0, 0].legend(fontsize=7)
    ax[0, 0].grid(alpha=0.3)

    labels = [row["case"] for row in rows]
    max_drift = [row["drift_max_abs_pct"] for row in rows]
    ax[0, 1].barh(labels, max_drift)
    ax[0, 1].set(title="Max absolute drift", xlabel="percent")
    ax[0, 1].grid(axis="x", alpha=0.3)

    for case in CASES:
        if not str(case["name"]).startswith("both"):
            continue
        data = load(case)
        ax[1, 0].plot(data["time"], data["ke_solid"], lw=1.2, label=f"{case['name']} solid")
        ax[1, 0].plot(data["time"], data["ke_fluid"], lw=1.2, ls="--", label=f"{case['name']} fluid")
    ax[1, 0].set(title="Both-motion solid/fluid exchange", xlabel="t", ylabel="KE")
    ax[1, 0].legend(fontsize=7)
    ax[1, 0].grid(alpha=0.3)

    for case in CASES:
        data = load(case)
        ax[1, 1].plot(data["time"], data["ke_total"], lw=1.2, label=case["name"])
    if all(col in load(CASES[-1]).dtype.names for col in ("pcon_x_1", "pcon_y_1", "pcon_z_1", "hcon_x_1", "hcon_y_1", "hcon_z_1")):
        data = load(CASES[-1])
        p_values = np.column_stack([data["pcon_x_1"], data["pcon_y_1"], data["pcon_z_1"]])
        h_values = np.column_stack([data["hcon_x_1"], data["hcon_y_1"], data["hcon_z_1"]])
        ax[1, 1].plot(data["time"], np.linalg.norm(p_values - p_values[0], axis=1), lw=1.2, ls=":", label="both |P-P0|")
        ax[1, 1].plot(data["time"], np.linalg.norm(h_values - h_values[0], axis=1), lw=1.2, ls=":", label="both |H-H0|")
    ax[1, 1].set(title="Total KE and combined invariant errors", xlabel="t", ylabel="KE / abs error")
    ax[1, 1].legend(fontsize=7)
    ax[1, 1].grid(alpha=0.3)

    fig.suptitle("Impulse coupling diagnostic: rotation/translation split", fontsize=13)
    fig.tight_layout()
    out = STUDY / "impulse_coupling_dashboard.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved {out.relative_to(ROOT)}")


def main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    stream_run(["cargo", "build", "--release"])
    exe = ROOT / "target" / "release" / "multi_ellip.exe"
    if not exe.exists():
        exe = ROOT / "target" / "release" / "multi_ellip"

    for i, case in enumerate(CASES, start=1):
        print()
        print("=" * 72)
        print(f"Case {i}/{len(CASES)}: {case['name']}")
        print("=" * 72)
        input_path = write_input(case)
        out_dir = RUNS / str(case["name"])
        stream_run([str(exe), str(input_path), str(out_dir)], out_dir / "run.log")

    rows = summarize()
    write_summary(rows)
    plot(rows)


if __name__ == "__main__":
    main()
