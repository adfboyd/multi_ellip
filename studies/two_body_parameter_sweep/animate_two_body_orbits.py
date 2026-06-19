from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = ROOT / "two_body_parameter_sweep_task_runs"


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
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


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def output_complete(row: dict[str, str]) -> bool:
    output = resolve(row["output"])
    if not output.exists():
        return False
    try:
        tend = float(row.get("tend", "nan"))
        dt = float(row.get("dt", "nan"))
    except ValueError:
        return False
    if not np.isfinite(tend) or not np.isfinite(dt) or dt <= 0.0:
        return False
    expected_rows = int(round(tend / dt)) + 1
    with output.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f) >= expected_rows + 1


def load_output(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    return data


def position(data: np.ndarray, body: int) -> np.ndarray:
    return np.column_stack([data[f"px_{body}"], data[f"py_{body}"], data[f"pz_{body}"]])


def drift_pct(data: np.ndarray) -> np.ndarray:
    e0 = float(data["ke_total"][0])
    return 100.0 * (data["ke_total"] - e0) / e0


def finite_complete_data(data: np.ndarray) -> bool:
    cols = ["time", "ke_total", "px_1", "py_1", "pz_1", "px_2", "py_2", "pz_2"]
    return len(data) >= 2 and all(col in data.dtype.names and np.all(np.isfinite(data[col])) for col in cols)


def dedupe_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_name: dict[str, dict[str, str]] = {}
    for row in rows:
        name = row.get("name", "")
        if not name:
            continue
        current = by_name.get(name)
        if current is None:
            by_name[name] = row
            continue
        if current.get("status") != "OK" or row.get("status") == "OK":
            by_name[name] = row
    return list(by_name.values())


def animation_path(row: dict[str, str], out_dir: Path | None, fmt: str) -> Path:
    run_dir = resolve(row["output"]).parent
    if out_dir is None:
        return run_dir / f"orbit_animation.{fmt}"
    return out_dir / "animations" / row["name"] / f"orbit_animation.{fmt}"


def set_equal_3d_bounds(ax, points: np.ndarray) -> None:
    mins = np.nanmin(points, axis=0)
    maxs = np.nanmax(points, axis=0)
    centre = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    radius = max(radius, 1.0)
    pad = 0.08 * radius
    for setter, c in ((ax.set_xlim, centre[0]), (ax.set_ylim, centre[1]), (ax.set_zlim, centre[2])):
        setter(c - radius - pad, c + radius + pad)


def colourline(ax, xyz: np.ndarray, values: np.ndarray, colour: str, alpha: float) -> Line3DCollection:
    if len(xyz) < 2:
        segments = np.zeros((0, 2, 3))
    else:
        segments = np.stack([xyz[:-1], xyz[1:]], axis=1)
    collection = Line3DCollection(segments, colors=colour, linewidth=0.6, alpha=alpha)
    ax.add_collection3d(collection)
    return collection


def downsample_indices(n: int, frames: int) -> np.ndarray:
    frames = min(max(frames, 2), n)
    return np.unique(np.linspace(0, n - 1, frames).astype(int))


def make_animation(row: dict[str, str], out_path: Path, frames: int, fps: int, tail_fraction: float, dpi: int) -> dict[str, str]:
    output = resolve(row["output"])
    data = load_output(output)
    if not finite_complete_data(data):
        return {**row, "animation": "", "animation_status": "non-finite-or-too-short"}

    t = data["time"]
    p1 = position(data, 1)
    p2 = position(data, 2)
    sep = np.linalg.norm(p1 - p2, axis=1)
    drift = drift_pct(data)
    idx = downsample_indices(len(t), frames)
    tail_len = max(2, int(round(len(t) * tail_fraction)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_points = np.vstack([p1, p2])

    fig = plt.figure(figsize=(11.5, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[2.0, 1.0, 1.0])
    ax3 = fig.add_subplot(gs[:, 0], projection="3d")
    ax_sep = fig.add_subplot(gs[0, 1:])
    ax_ke = fig.add_subplot(gs[1, 1:])

    colourline(ax3, p1, t, "#2f6fb0", 0.16)
    colourline(ax3, p2, t, "#b84a3a", 0.16)
    tail1, = ax3.plot([], [], [], color="#2f6fb0", lw=1.8, label="body 1")
    tail2, = ax3.plot([], [], [], color="#b84a3a", lw=1.8, label="body 2")
    marker1, = ax3.plot([], [], [], "o", color="#2f6fb0", ms=5)
    marker2, = ax3.plot([], [], [], "o", color="#b84a3a", ms=5)
    connector, = ax3.plot([], [], [], color="#59636e", lw=0.6, alpha=0.55)
    set_equal_3d_bounds(ax3, all_points)
    ax3.set(title="Orbit evolution", xlabel="x", ylabel="y", zlabel="z")
    ax3.legend(loc="upper left", fontsize=8)

    ax_sep.plot(t, sep, color="#59636e", lw=1.0)
    sep_marker, = ax_sep.plot([], [], "o", color="#111827", ms=4)
    ax_sep.set(title="Body separation", xlabel="t", ylabel=r"$|\mathbf{x}_1-\mathbf{x}_2|$")
    ax_sep.grid(alpha=0.25)

    ax_ke.plot(t, drift, color="#6f5aa8", lw=1.0)
    ke_marker, = ax_ke.plot([], [], "o", color="#111827", ms=4)
    ax_ke.set(title="Total KE drift", xlabel="t", ylabel="drift (%)")
    ax_ke.grid(alpha=0.25)

    title = fig.suptitle("", fontsize=12)

    def update(frame: int):
        k = int(idx[frame])
        start = max(0, k - tail_len)
        for line, path in ((tail1, p1), (tail2, p2)):
            segment = path[start : k + 1]
            line.set_data(segment[:, 0], segment[:, 1])
            line.set_3d_properties(segment[:, 2])
        for marker, path in ((marker1, p1), (marker2, p2)):
            marker.set_data([path[k, 0]], [path[k, 1]])
            marker.set_3d_properties([path[k, 2]])
        connector.set_data([p1[k, 0], p2[k, 0]], [p1[k, 1], p2[k, 1]])
        connector.set_3d_properties([p1[k, 2], p2[k, 2]])
        sep_marker.set_data([t[k]], [sep[k]])
        ke_marker.set_data([t[k]], [drift[k]])
        title.set_text(f"{row.get('name', '')}   t={t[k]:.2f}")
        return tail1, tail2, marker1, marker2, connector, sep_marker, ke_marker, title

    if out_path.suffix.lower() == ".mp4":
        if shutil.which("ffmpeg") is None:
            plt.close(fig)
            return {**row, "animation": "", "animation_status": "ffmpeg-not-found"}
        writer = FFMpegWriter(fps=fps, bitrate=1800)
    else:
        writer = PillowWriter(fps=fps)

    anim = FuncAnimation(fig, update, frames=len(idx), interval=1000 / fps, blit=False)
    anim.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)
    return {
        **row,
        "animation": str(out_path),
        "animation_status": "OK",
        "animation_frames": str(len(idx)),
        "animation_fps": str(fps),
    }


def rel(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def write_index(rows: list[dict[str, str]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    ok = [
        row
        for row in rows
        if row.get("animation_status") in {"OK", "existing"} and row.get("animation")
    ]
    cards = []
    for row in ok:
        animation = resolve(row["animation"])
        dashboard = resolve(row.get("dashboard", "")) if row.get("dashboard") else None
        fields = [
            ("shape", row.get("shape_name", "")),
            ("rho", row.get("rho", "")),
            ("E", row.get("energy_ratio", "")),
            ("sep", row.get("separation", "")),
            ("repeat", row.get("repeat", "")),
            ("max KE drift %", row.get("max_abs_ke_drift_pct", "")),
        ]
        meta = "\n".join(f"<li><b>{label}</b>: {value}</li>" for label, value in fields)
        dash = f'<a href="{rel(dashboard, out.parent)}">dashboard</a>' if dashboard else ""
        cards.append(
            f"""
            <article class="card">
              <h2>{row.get("name", "")}</h2>
              <a href="{rel(animation, out.parent)}"><img src="{rel(animation, out.parent)}" alt="orbit animation for {row.get("name", "")}"></a>
              <ul>{meta}</ul>
              <p><a href="{rel(animation, out.parent)}">animation</a> {dash}</p>
            </article>
            """
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Two-body orbit animation index</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; background: #f6f7f8; color: #1f2328; }}
    h1 {{ margin: 0 0 6px; font-size: 24px; }}
    .summary {{ color: #5b626b; margin-bottom: 20px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 18px; }}
    .card {{ background: white; border: 1px solid #d8dee4; border-radius: 8px; padding: 14px; }}
    .card h2 {{ font-size: 14px; margin: 0 0 10px; overflow-wrap: anywhere; }}
    img {{ width: 100%; height: auto; border: 1px solid #d8dee4; border-radius: 4px; }}
    ul {{ columns: 2; padding-left: 18px; font-size: 12px; line-height: 1.45; }}
    a {{ color: #0969da; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <h1>Two-body orbit animation index</h1>
  <div class="summary">{len(ok)} animations available from {len(rows)} attempted rows.</div>
  <main class="grid">
    {"".join(cards)}
  </main>
</body>
</html>
"""
    out.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate orbit animations for completed two-body sweep cases.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None, help="optional mirror output directory; defaults to each run dir")
    parser.add_argument("--format", choices=("gif", "mp4"), default="gif")
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--tail-fraction", type=float, default=0.16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--missing-only", action="store_true")
    args = parser.parse_args()

    run_root = args.run_root if args.run_root.is_absolute() else ROOT / args.run_root
    summary = args.summary or run_root / "run_summary.csv"
    if not summary.is_absolute():
        summary = ROOT / summary
    out_dir = args.out_dir if args.out_dir is None or args.out_dir.is_absolute() else ROOT / args.out_dir

    rows = dedupe_rows(read_rows(summary))
    candidates = [
        row
        for row in rows
        if row.get("status", "OK") == "OK" and row.get("output") and output_complete(row)
    ]
    if args.limit is not None:
        candidates = candidates[: args.limit]

    results = []
    for i, row in enumerate(candidates, start=1):
        out_path = animation_path(row, out_dir, args.format)
        if args.missing_only and out_path.exists():
            print(f"[{i:03d}/{len(candidates):03d}] SKIP existing {row['name']}", flush=True)
            results.append({**row, "animation": str(out_path), "animation_status": "existing"})
            continue
        print(f"[{i:03d}/{len(candidates):03d}] ANIMATE {row['name']}", flush=True)
        results.append(make_animation(row, out_path, args.frames, args.fps, args.tail_fraction, args.dpi))

    analysis_dir = out_dir if out_dir is not None else run_root / "analysis"
    result_csv = analysis_dir / "orbit_animation_summary.csv"
    index_html = analysis_dir / "orbit_animation_index.html"
    write_rows(result_csv, results)
    write_index(results, index_html)
    print(f"Wrote {result_csv}")
    print(f"Wrote {index_html}")


if __name__ == "__main__":
    main()
