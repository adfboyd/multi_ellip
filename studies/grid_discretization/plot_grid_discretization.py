from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "studies" / "grid_discretization"
OUT = STUDY / "figures"


BASE_FACES = [
    [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
    [(-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)],
    [(-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)],
    [(0.0, 0.0, 1.0), (0.0, -1.0, 0.0), (1.0, 0.0, 0.0)],
    [(1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)],
    [(0.0, 0.0, -1.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
    [(0.0, 0.0, -1.0), (0.0, -1.0, 0.0), (-1.0, 0.0, 0.0)],
    [(1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)],
]


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / n


def element_from_corners(corners: np.ndarray) -> np.ndarray:
    p1, p2, p3 = corners
    return np.array(
        [
            p1,
            p2,
            p3,
            0.5 * (p1 + p2),
            0.5 * (p2 + p3),
            0.5 * (p3 + p1),
        ],
        dtype=float,
    )


def unit_elements(ndiv: int) -> np.ndarray:
    elems = np.array([element_from_corners(np.array(face, dtype=float)) for face in BASE_FACES])
    elems = normalize(elems)

    for _ in range(ndiv):
        children = []
        for e in elems:
            p1, p2, p3, p4, p5, p6 = e
            child_corners = [
                [p1, p4, p6],
                [p4, p2, p5],
                [p6, p5, p3],
                [p4, p5, p6],
            ]
            children.extend(element_from_corners(np.array(c)) for c in child_corners)
        elems = normalize(np.array(children))
    return elems


def deduplicate_nodes(elems: np.ndarray, decimals: int = 12) -> tuple[np.ndarray, np.ndarray]:
    nodes: list[np.ndarray] = []
    lookup: dict[tuple[float, float, float], int] = {}
    conn = np.zeros((len(elems), 6), dtype=int)
    for i, elem in enumerate(elems):
        for j, point in enumerate(elem):
            key = tuple(np.round(point, decimals))
            idx = lookup.get(key)
            if idx is None:
                idx = len(nodes)
                lookup[key] = idx
                nodes.append(point.copy())
            conn[i, j] = idx
    return np.array(nodes), conn


def solver_axes(req: float, shape: np.ndarray) -> np.ndarray:
    a, b, c = shape
    boa = b / a
    coa = c / a
    scale = req / (boa * coa) ** (1.0 / 3.0)
    return np.array([scale, scale * boa, scale * coa], dtype=float)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def transform(points: np.ndarray, axes: np.ndarray, center: np.ndarray, quat: np.ndarray) -> np.ndarray:
    rot = quat_to_matrix(quat)
    scaled = points * axes
    return scaled @ rot.T + center


def body_frame(points: np.ndarray, center: np.ndarray, quat: np.ndarray) -> np.ndarray:
    rot = quat_to_matrix(quat)
    return (points - center) @ rot


def mesh(ndiv: int, axes: np.ndarray, center: np.ndarray, quat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    elems_unit = unit_elements(ndiv)
    nodes_unit, conn = deduplicate_nodes(elems_unit)
    nodes = transform(nodes_unit, axes, center, quat)
    elems = nodes[conn]
    return nodes, conn, elems


def abc(elem: np.ndarray) -> tuple[float, float, float]:
    p1, p2, p3, p4, p5, p6 = elem
    d42 = np.linalg.norm(p4 - p2)
    d41 = np.linalg.norm(p4 - p1)
    d63 = np.linalg.norm(p6 - p3)
    d61 = np.linalg.norm(p6 - p1)
    d52 = np.linalg.norm(p5 - p2)
    d53 = np.linalg.norm(p5 - p3)
    alpha = 1.0 / (1.0 + d42 / d41)
    beta = 1.0 / (1.0 + d63 / d61)
    gamma = 1.0 / (1.0 + d52 / d53)
    return alpha, beta, gamma


def quadratic_basis(xi: float, eta: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
    alc = 1.0 - alpha
    bec = 1.0 - beta
    gac = 1.0 - gamma
    ph2 = xi * (xi - alpha + eta * (alpha - gamma) / gac) / alc
    ph3 = eta * (eta - beta + xi * (beta + gamma - 1.0) / gamma) / bec
    ph4 = xi * (1.0 - xi - eta) / (alpha * alc)
    ph5 = xi * eta / (gamma * gac)
    ph6 = eta * (1.0 - xi - eta) / (beta * bec)
    ph1 = 1.0 - ph2 - ph3 - ph4 - ph5 - ph6
    return np.array([ph1, ph2, ph3, ph4, ph5, ph6], dtype=float)


def quadratic_basis_derivatives(
    xi: float, eta: float, alpha: float, beta: float, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    alc = 1.0 - alpha
    bec = 1.0 - beta
    gac = 1.0 - gamma

    dph2_dxi = (2.0 * xi - alpha + eta * (alpha - gamma) / gac) / alc
    dph2_deta = xi * (alpha - gamma) / (alc * gac)
    dph3_dxi = eta * (beta + gamma - 1.0) / (gamma * bec)
    dph3_deta = (2.0 * eta - beta + xi * (beta + gamma - 1.0) / gamma) / bec
    dph4_dxi = (1.0 - 2.0 * xi - eta) / (alpha * alc)
    dph4_deta = -xi / (alpha * alc)
    dph5_dxi = eta / (gamma * gac)
    dph5_deta = xi / (gamma * gac)
    dph6_dxi = -eta / (beta * bec)
    dph6_deta = (1.0 - xi - 2.0 * eta) / (beta * bec)

    dph1_dxi = -(dph2_dxi + dph3_dxi + dph4_dxi + dph5_dxi + dph6_dxi)
    dph1_deta = -(dph2_deta + dph3_deta + dph4_deta + dph5_deta + dph6_deta)
    return (
        np.array([dph1_dxi, dph2_dxi, dph3_dxi, dph4_dxi, dph5_dxi, dph6_dxi], dtype=float),
        np.array([dph1_deta, dph2_deta, dph3_deta, dph4_deta, dph5_deta, dph6_deta], dtype=float),
    )


def quadratic_point(elem: np.ndarray, xi: float, eta: float) -> np.ndarray:
    basis = quadratic_basis(xi, eta, *abc(elem))
    return basis @ elem


def quadratic_normal(elem: np.ndarray, xi: float, eta: float) -> np.ndarray:
    dxi, deta = quadratic_basis_derivatives(xi, eta, *abc(elem))
    tangent_xi = dxi @ elem
    tangent_eta = deta @ elem
    normal = np.cross(tangent_xi, tangent_eta)
    return normal / max(np.linalg.norm(normal), 1.0e-15)


def triangle_samples(resolution: int) -> tuple[list[tuple[int, int]], list[tuple[int, int, int]]]:
    ij = [(i, j) for i in range(resolution + 1) for j in range(resolution + 1 - i)]
    index = {v: k for k, v in enumerate(ij)}
    faces = []
    for i in range(resolution):
        for j in range(resolution - i):
            faces.append((index[(i, j)], index[(i + 1, j)], index[(i, j + 1)]))
            if i + j < resolution - 1:
                faces.append((index[(i + 1, j)], index[(i + 1, j + 1)], index[(i, j + 1)]))
    return ij, faces


def tri_area(tri: np.ndarray) -> float:
    return 0.5 * float(np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])))


def sampled_quadratic_surface(
    elems: np.ndarray,
    axes: np.ndarray,
    center: np.ndarray,
    quat: np.ndarray,
    resolution: int,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    ij, faces = triangle_samples(resolution)
    xi_eta = np.array([(i / resolution, j / resolution) for i, j in ij])
    tris: list[np.ndarray] = []
    errors: list[float] = []
    areas: list[float] = []
    normal_angles: list[float] = []
    for elem in elems:
        pts = []
        for i, j in ij:
            xi = i / resolution
            eta = j / resolution
            pts.append(quadratic_point(elem, xi, eta))
        pts = np.array(pts)
        for face in faces:
            tri = pts[list(face)]
            centroid = np.mean(tri, axis=0)
            dist = ellipsoid_signed_distance_approx(centroid, axes, center, quat)
            xi_centroid, eta_centroid = np.mean(xi_eta[list(face)], axis=0)
            discrete_normal = quadratic_normal(elem, xi_centroid, eta_centroid)
            exact_normal = analytic_normal(centroid, axes, center, quat)
            dot = float(np.clip(abs(np.dot(discrete_normal, exact_normal)), 0.0, 1.0))
            tris.append(tri)
            errors.append(abs(dist))
            areas.append(tri_area(tri))
            normal_angles.append(math.acos(dot))
    return tris, np.array(errors), np.array(areas), np.array(normal_angles)


def ellipsoid_signed_distance_approx(point: np.ndarray, axes: np.ndarray, center: np.ndarray, quat: np.ndarray) -> float:
    x = body_frame(np.asarray(point)[None, :], center, quat)[0]
    f = np.sum((x / axes) ** 2) - 1.0
    grad = np.array([2.0 * x[0] / axes[0] ** 2, 2.0 * x[1] / axes[1] ** 2, 2.0 * x[2] / axes[2] ** 2])
    return f / max(np.linalg.norm(grad), 1.0e-15)


def analytic_normal(point: np.ndarray, axes: np.ndarray, center: np.ndarray, quat: np.ndarray) -> np.ndarray:
    x = body_frame(np.asarray(point)[None, :], center, quat)[0]
    normal_body = np.array([x[0] / axes[0] ** 2, x[1] / axes[1] ** 2, x[2] / axes[2] ** 2])
    rot = quat_to_matrix(quat)
    normal_lab = normal_body @ rot.T
    return normal_lab / max(np.linalg.norm(normal_lab), 1.0e-15)


def analytic_surface(axes: np.ndarray, center: np.ndarray, quat: np.ndarray, nu: int = 80, nv: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0.0, 2.0 * math.pi, nu)
    v = np.linspace(0.0, math.pi, nv)
    uu, vv = np.meshgrid(u, v)
    body = np.column_stack(
        [
            axes[0] * np.sin(vv).ravel() * np.cos(uu).ravel(),
            axes[1] * np.sin(vv).ravel() * np.sin(uu).ravel(),
            axes[2] * np.cos(vv).ravel(),
        ]
    )
    rot = quat_to_matrix(quat)
    lab = body @ rot.T + center
    x = lab[:, 0].reshape(vv.shape)
    y = lab[:, 1].reshape(vv.shape)
    z = lab[:, 2].reshape(vv.shape)
    return x, y, z


def curved_edges(elems: np.ndarray, samples: int = 9) -> list[np.ndarray]:
    edges = []
    tvals = np.linspace(0.0, 1.0, samples)
    for elem in elems:
        edge_pts = []
        edge_pts.append([quadratic_point(elem, t, 0.0) for t in tvals])
        edge_pts.append([quadratic_point(elem, 1.0 - t, t) for t in tvals])
        edge_pts.append([quadratic_point(elem, 0.0, 1.0 - t) for t in tvals])
        edges.extend(np.array(e) for e in edge_pts)
    return edges


def set_equal_axes(ax, center: np.ndarray, axes: np.ndarray) -> None:
    radius = float(np.max(axes)) * 1.08
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=22, azim=-42)


def plot_overlay_panel(ndivs: list[int], axes: np.ndarray, center: np.ndarray, quat: np.ndarray, out: Path) -> None:
    fig = plt.figure(figsize=(4.7 * len(ndivs), 4.5))
    sx, sy, sz = analytic_surface(axes, center, quat, nu=64, nv=32)
    for idx, ndiv in enumerate(ndivs, start=1):
        _, conn, elems = mesh(ndiv, axes, center, quat)
        ax = fig.add_subplot(1, len(ndivs), idx, projection="3d")
        ax.plot_surface(sx, sy, sz, color="#6aaed6", alpha=0.18, linewidth=0, shade=True)
        ax.add_collection3d(Line3DCollection(curved_edges(elems, samples=8), colors="#222222", linewidths=0.55))
        ax.scatter(elems[:, :3, 0].ravel(), elems[:, :3, 1].ravel(), elems[:, :3, 2].ravel(), s=3, color="#b2182b", alpha=0.65)
        ax.set_title(f"ndiv={ndiv}\n{len(conn)} quadratic triangles")
        set_equal_axes(ax, center, axes)
    fig.suptitle("Solver ellipsoid grid over analytical surface", fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_error_panel(
    ndivs: list[int],
    axes: np.ndarray,
    center: np.ndarray,
    quat: np.ndarray,
    sample_resolution: int,
    out: Path,
) -> list[dict[str, float]]:
    stats = []
    sampled = []
    positive_errors = []
    for ndiv in ndivs:
        _, conn, elems = mesh(ndiv, axes, center, quat)
        tris, err, area, normal_angle = sampled_quadratic_surface(elems, axes, center, quat, sample_resolution)
        sampled.append((ndiv, conn, tris, err, area, normal_angle))
        positive_errors.extend(err[err > 0.0])
        area_sum = float(np.sum(area))
        stats.append(
            {
                "ndiv": ndiv,
                "triangles": len(conn),
                "nodes": len(np.unique(conn)),
                "sampled_faces": len(err),
                "rms_abs_distance": float(np.sqrt(np.mean(err * err))),
                "surface_l2_abs_distance": float(np.sqrt(np.sum(area * err * err) / area_sum)),
                "surface_l1_abs_distance": float(np.sum(area * err) / area_sum),
                "p95_abs_distance": float(np.percentile(err, 95)),
                "max_abs_distance": float(np.max(err)),
                "normal_rms_angle_rad": float(np.sqrt(np.sum(area * normal_angle * normal_angle) / area_sum)),
                "normal_p95_angle_rad": float(np.percentile(normal_angle, 95)),
                "normal_max_angle_rad": float(np.max(normal_angle)),
            }
        )

    if positive_errors:
        vmin = max(min(positive_errors), 1.0e-10)
        vmax = max(max(positive_errors), vmin * 10.0)
    else:
        vmin, vmax = 1.0e-10, 1.0e-9
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = cm.magma

    fig = plt.figure(figsize=(4.9 * len(ndivs), 4.8))
    for idx, (ndiv, _conn, tris, err, _area, _normal_angle) in enumerate(sampled, start=1):
        ax = fig.add_subplot(1, len(ndivs), idx, projection="3d")
        facecolors = cmap(norm(np.maximum(err, vmin)))
        poly = Poly3DCollection(tris, facecolors=facecolors, linewidths=0.02, edgecolors=(0, 0, 0, 0.08))
        ax.add_collection3d(poly)
        stat = next(row for row in stats if row["ndiv"] == ndiv)
        ax.set_title(
            f"ndiv={ndiv}\nRMS {stat['rms_abs_distance']:.2e}, max {stat['max_abs_distance']:.2e}"
        )
        set_equal_axes(ax, center, axes)

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=fig.axes, shrink=0.72, pad=0.03)
    cbar.set_label("approx. absolute distance to analytical ellipsoid")
    fig.suptitle("Quadratic element surface error between mesh nodes", fontsize=14)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return stats


def plot_normal_panel(
    ndivs: list[int],
    axes: np.ndarray,
    center: np.ndarray,
    quat: np.ndarray,
    sample_resolution: int,
    out: Path,
) -> None:
    sampled = []
    positive_angles = []
    for ndiv in ndivs:
        _, conn, elems = mesh(ndiv, axes, center, quat)
        tris, _err, area, normal_angle = sampled_quadratic_surface(
            elems, axes, center, quat, sample_resolution
        )
        area_sum = float(np.sum(area))
        rms_angle = float(np.sqrt(np.sum(area * normal_angle * normal_angle) / area_sum))
        sampled.append((ndiv, conn, tris, normal_angle, rms_angle, float(np.max(normal_angle))))
        positive_angles.extend(normal_angle[normal_angle > 0.0])

    if positive_angles:
        vmin = max(min(positive_angles), 1.0e-8)
        vmax = max(max(positive_angles), vmin * 10.0)
    else:
        vmin, vmax = 1.0e-8, 1.0e-7
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = cm.viridis

    fig = plt.figure(figsize=(4.9 * len(ndivs), 4.8))
    for idx, (ndiv, _conn, tris, normal_angle, rms_angle, max_angle) in enumerate(sampled, start=1):
        ax = fig.add_subplot(1, len(ndivs), idx, projection="3d")
        facecolors = cmap(norm(np.maximum(normal_angle, vmin)))
        poly = Poly3DCollection(tris, facecolors=facecolors, linewidths=0.02, edgecolors=(0, 0, 0, 0.08))
        ax.add_collection3d(poly)
        ax.set_title(f"ndiv={ndiv}\nRMS {math.degrees(rms_angle):.2f} deg, max {math.degrees(max_angle):.2f} deg")
        set_equal_axes(ax, center, axes)

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=fig.axes, shrink=0.72, pad=0.03)
    cbar.set_label("normal-direction error (radians)")
    fig.suptitle("Surface normal error against analytical ellipsoid", fontsize=14)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_summary(stats: list[dict[str, float]], out: Path) -> None:
    ndiv = np.array([row["ndiv"] for row in stats])
    h = 2.0 ** (-ndiv)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for key, label in [
        ("rms_abs_distance", "RMS"),
        ("p95_abs_distance", "95th percentile"),
        ("max_abs_distance", "max"),
    ]:
        y = np.array([row[key] for row in stats])
        ax.loglog(h, y, "o-", label=label)
    ax.invert_xaxis()
    ax.set_xlabel("nominal grid spacing h ~ 2^-ndiv")
    ax.set_ylabel("absolute distance to analytical ellipsoid")
    ax.set_title("Geometric convergence of quadratic surface")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_surface_norm_summary(stats: list[dict[str, float]], out: Path) -> None:
    ndiv = np.array([row["ndiv"] for row in stats])
    h = 2.0 ** (-ndiv)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for key, label in [
        ("surface_l2_abs_distance", r"$\|d\|_{L^2(\Gamma_h)}$"),
        ("surface_l1_abs_distance", r"$\|d\|_{L^1(\Gamma_h)}$ mean"),
        ("max_abs_distance", r"$\|d\|_{L^\infty}$"),
    ]:
        y = np.array([row[key] for row in stats])
        ax.loglog(h, y, "o-", label=label)
    ax.invert_xaxis()
    ax.set_xlabel("nominal grid spacing h ~ 2^-ndiv")
    ax.set_ylabel("surface norm of distance error")
    ax.set_title("Area-weighted surface norm error")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_normal_summary(stats: list[dict[str, float]], out: Path) -> None:
    ndiv = np.array([row["ndiv"] for row in stats])
    h = 2.0 ** (-ndiv)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for key, label in [
        ("normal_rms_angle_rad", "RMS angle"),
        ("normal_p95_angle_rad", "95th percentile angle"),
        ("normal_max_angle_rad", "max angle"),
    ]:
        y = np.degrees(np.array([row[key] for row in stats]))
        ax.loglog(h, y, "o-", label=label)
    ax.invert_xaxis()
    ax.set_xlabel("nominal grid spacing h ~ 2^-ndiv")
    ax.set_ylabel("normal direction error (degrees)")
    ax.set_title("Surface normal convergence")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_input(path: Path, body: int) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    values: dict[str, float] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.strip().startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        values[key.strip()] = float(value.strip())
    shape = np.array([values[f"shx{body}"], values[f"shy{body}"], values[f"shz{body}"]], dtype=float)
    req = values.get(f"req{body}", 1.0)
    center = np.array(
        [values.get(f"cex{body}", 0.0), values.get(f"cey{body}", 0.0), values.get(f"cez{body}", 0.0)],
        dtype=float,
    )
    quat = np.array(
        [
            values.get(f"oriw{body}", 1.0),
            values.get(f"orii{body}", 0.0),
            values.get(f"orij{body}", 0.0),
            values.get(f"orik{body}", 0.0),
        ],
        dtype=float,
    )
    return shape, req, center, quat


def write_stats(path: Path, stats: list[dict[str, float]], axes: np.ndarray, shape: np.ndarray, req: float) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "shape_x",
            "shape_y",
            "shape_z",
            "req",
            "axis_x",
            "axis_y",
            "axis_z",
            "ndiv",
            "triangles",
            "nodes",
            "sampled_faces",
            "rms_abs_distance",
            "surface_l2_abs_distance",
            "surface_l1_abs_distance",
            "p95_abs_distance",
            "max_abs_distance",
            "normal_rms_angle_rad",
            "normal_p95_angle_rad",
            "normal_max_angle_rad",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in stats:
            writer.writerow(
                {
                    "shape_x": shape[0],
                    "shape_y": shape[1],
                    "shape_z": shape[2],
                    "req": req,
                    "axis_x": axes[0],
                    "axis_y": axes[1],
                    "axis_z": axes[2],
                    **row,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate figures comparing the solver ellipsoid grid to the analytical ellipsoid."
    )
    parser.add_argument("--input", type=Path, help="optional solver input.txt to read shape/position/orientation")
    parser.add_argument("--body", type=int, default=1, help="body index when reading --input")
    parser.add_argument("--shape", nargs=3, type=float, default=[1.0, 0.8, 0.6], metavar=("A", "B", "C"))
    parser.add_argument("--req", type=float, default=1.0)
    parser.add_argument("--center", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--quat", nargs=4, type=float, default=[1.0, 0.0, 0.0, 0.0], metavar=("W", "I", "J", "K"))
    parser.add_argument("--ndivs", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--sample-resolution", type=int, default=5)
    parser.add_argument("--outdir", type=Path, default=OUT)
    args = parser.parse_args()

    if args.input:
        shape, req, center, quat = parse_input(args.input, args.body)
    else:
        shape = np.array(args.shape, dtype=float)
        req = float(args.req)
        center = np.array(args.center, dtype=float)
        quat = np.array(args.quat, dtype=float)
    axes = solver_axes(req, shape)
    quat = quat / np.linalg.norm(quat)

    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = f"body{args.body}" if args.input else "default"
    overlay_path = args.outdir / f"{stem}_grid_overlay_panel.png"
    error_path = args.outdir / f"{stem}_grid_error_panel.png"
    summary_path = args.outdir / f"{stem}_grid_error_summary.png"
    surface_norm_path = args.outdir / f"{stem}_grid_surface_norm_summary.png"
    normal_panel_path = args.outdir / f"{stem}_grid_normal_error_panel.png"
    normal_summary_path = args.outdir / f"{stem}_grid_normal_error_summary.png"
    csv_path = args.outdir / f"{stem}_grid_error_summary.csv"

    plot_overlay_panel(args.ndivs, axes, center, quat, overlay_path)
    stats = plot_error_panel(args.ndivs, axes, center, quat, args.sample_resolution, error_path)
    plot_summary(stats, summary_path)
    plot_surface_norm_summary(stats, surface_norm_path)
    plot_normal_panel(args.ndivs, axes, center, quat, args.sample_resolution, normal_panel_path)
    plot_normal_summary(stats, normal_summary_path)
    write_stats(csv_path, stats, axes, shape, req)

    print(f"shape ratios: {shape}")
    print(f"solver semi-axes: {axes}")
    print(f"Saved {overlay_path}")
    print(f"Saved {error_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {surface_norm_path}")
    print(f"Saved {normal_panel_path}")
    print(f"Saved {normal_summary_path}")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
