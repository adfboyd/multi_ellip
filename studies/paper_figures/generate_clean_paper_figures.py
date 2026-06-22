from __future__ import annotations

import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "paper_figures"
SECTION3 = DOCS / "section3_clean"
SECTION4 = DOCS / "section4_setup"


def copy_asset(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(dst)


def read_csv_numeric(path: Path) -> dict[str, np.ndarray]:
    import csv

    cols: dict[str, list[float]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for key, value in row.items():
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    number = math.nan
                cols.setdefault(key, []).append(number)
    return {key: np.asarray(values, dtype=float) for key, values in cols.items()}


def plot_geometric_error_convergence() -> None:
    data = read_csv_numeric(DOCS / "default_grid_error_summary.csv")
    h = np.sqrt(8.0 / data["triangles"])

    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.6), constrained_layout=True)
    ax[0].loglog(h, data["surface_l2_abs_distance"], "o-", label=r"$L^2$ surface distance")
    ax[0].loglog(h, data["p95_abs_distance"], "s-", label="95th percentile distance")
    ax[0].invert_xaxis()
    ax[0].set_xlabel(r"panel scale $h$")
    ax[0].set_ylabel("surface distance error")
    ax[0].grid(alpha=0.25, which="both")
    ax[0].legend(fontsize=8)

    ax[1].loglog(h, np.degrees(data["normal_rms_angle_rad"]), "o-", label="RMS normal angle")
    ax[1].loglog(h, np.degrees(data["normal_p95_angle_rad"]), "s-", label="95th percentile angle")
    ax[1].invert_xaxis()
    ax[1].set_xlabel(r"panel scale $h$")
    ax[1].set_ylabel("normal error (degrees)")
    ax[1].grid(alpha=0.25, which="both")
    ax[1].legend(fontsize=8)

    out = SECTION3 / "section3_geometric_error_convergence.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240)
    plt.close(fig)
    print(out)


def image_panel(paths: list[Path], labels: list[str], out: Path, figsize: tuple[float, float]) -> None:
    fig, axes = plt.subplots(1, len(paths), figsize=figsize, constrained_layout=True)
    axes_arr = np.atleast_1d(axes)
    for axis, path, label in zip(axes_arr, paths, labels):
        axis.imshow(mpimg.imread(path))
        axis.set_title(label, fontsize=10)
        axis.set_axis_off()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(out)


def make_section3() -> None:
    copy_asset(DOCS / "default_grid_overlay_panel.png", SECTION3 / "section3_grid_discretisation.png")
    copy_asset(DOCS / "exact_singular_convergence.png", SECTION3 / "section3_bem_convergence.png")
    plot_geometric_error_convergence()

    recurrence_root = DOCS / "ke_ratio_recurrence_current_comparison" / "plots" / "triaxial_nd2"
    image_panel(
        [
            recurrence_root / "ratio1_rho4_run01_recurrence.png",
            recurrence_root / "ratio20_rho0p25_run04_recurrence.png",
        ],
        [r"Regular/quasiperiodic, $\rho=4$, $E=1$", r"Chaotic-like, $\rho=0.25$, $E=20$"],
        SECTION3 / "section3_recurrence_examples.png",
        (8.2, 4.2),
    )

    copy_asset(
        DOCS / "representative_single_body_orbits" / "triaxial_nd2_representative_marker_orbits.png",
        SECTION3 / "section3_marker_orbit_examples.png",
    )


def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    cc = 1.0 - c
    return np.array(
        [
            [c + x * x * cc, x * y * cc - z * s, x * z * cc + y * s],
            [y * x * cc + z * s, c + y * y * cc, y * z * cc - x * s],
            [z * x * cc - y * s, z * y * cc + x * s, c + z * z * cc],
        ]
    )


def ellipsoid_surface(
    axes: np.ndarray,
    center: np.ndarray,
    rot: np.ndarray,
    n_u: int = 64,
    n_v: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0.0, 2.0 * math.pi, n_u)
    v = np.linspace(0.0, math.pi, n_v)
    uu, vv = np.meshgrid(u, v)
    local = np.stack(
        [
            axes[0] * np.cos(uu) * np.sin(vv),
            axes[1] * np.sin(uu) * np.sin(vv),
            axes[2] * np.cos(vv),
        ],
        axis=-1,
    )
    xyz = local @ rot.T + center
    return xyz[..., 0], xyz[..., 1], xyz[..., 2]


def arrow3d(ax: plt.Axes, start: np.ndarray, vec: np.ndarray, color: str, label: str, lw: float = 1.6) -> None:
    ax.quiver(
        start[0],
        start[1],
        start[2],
        vec[0],
        vec[1],
        vec[2],
        color=color,
        linewidth=lw,
        arrow_length_ratio=0.14,
        normalize=False,
    )
    end = start + vec
    ax.text(end[0], end[1], end[2], label, color=color, fontsize=10)


def set_equal_3d(ax: plt.Axes, limits: tuple[float, float, float, float, float, float]) -> None:
    xmin, xmax, ymin, ymax, zmin, zmax = limits
    center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0])
    radius = max(xmax - xmin, ymax - ymin, zmax - zmin) / 2.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def make_two_body_setup() -> None:
    shape = np.array([1.0, 0.8, 0.6])
    axes = shape / np.cbrt(np.prod(shape))
    c1 = np.array([-2.2, 0.0, 0.0])
    c2 = np.array([2.2, 0.0, 0.0])
    r1 = rotation_matrix(np.array([0.2, 1.0, 0.3]), math.radians(32.0))
    r2 = rotation_matrix(np.array([0.9, 0.1, 0.5]), math.radians(-48.0))

    fig = plt.figure(figsize=(8.6, 4.9), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    for center, rot, color in [(c1, r1, "#4e79a7"), (c2, r2, "#f28e2b")]:
        x, y, z = ellipsoid_surface(axes, center, rot)
        ax.plot_surface(x, y, z, color=color, alpha=0.64, linewidth=0.0, shade=True)
        ax.plot_wireframe(x, y, z, color="0.22", linewidth=0.25, rstride=4, cstride=6, alpha=0.34)

    ax.plot([c1[0], c2[0]], [c1[1], c2[1]], [c1[2], c2[2]], color="0.15", lw=1.1, ls="--")
    ax.text(0.0, 0.08, -0.35, r"$d$", fontsize=12, color="0.12", ha="center")
    arrow3d(ax, c1 + np.array([-0.45, -1.35, -0.75]), np.array([1.15, 0.0, 0.0]), "#2ca02c", r"$\mathbf{U}_0$")
    arrow3d(ax, c2 + np.array([-0.45, -1.35, -0.75]), np.array([1.15, 0.0, 0.0]), "#2ca02c", r"$\mathbf{U}_0$")
    arrow3d(ax, c1 + np.array([0.0, 0.0, 1.1]), np.array([-0.45, 0.48, 0.55]), "#9467bd", r"$\boldsymbol{\Omega}_1$")
    arrow3d(ax, c2 + np.array([0.0, 0.0, 1.1]), np.array([0.32, 0.58, -0.45]), "#9467bd", r"$\boldsymbol{\Omega}_2$")

    label_box = {"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9, "pad": 3.0}
    ax.text2D(0.18, 0.66, "body 1\naxes 1:0.8:0.6", transform=ax.transAxes, fontsize=9, bbox=label_box)
    ax.text2D(0.74, 0.42, "body 2\naxes 1:0.8:0.6", transform=ax.transAxes, fontsize=9, bbox=label_box)

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    ax.view_init(elev=20, azim=-58)
    set_equal_3d(ax, (-3.6, 3.6, -1.8, 1.8, -1.6, 1.7))
    ax.grid(alpha=0.14)

    out = SECTION4 / "section4_two_body_setup_schematic.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=260)
    plt.close(fig)
    print(out)


def main() -> None:
    make_section3()
    make_two_body_setup()


if __name__ == "__main__":
    main()
