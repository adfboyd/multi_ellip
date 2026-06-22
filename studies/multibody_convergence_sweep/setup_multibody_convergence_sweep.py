import argparse
import csv
import itertools
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "studies" / "multibody_convergence_sweep"
RUNS = STUDY / "runs"
MANIFEST = STUDY / "manifest.csv"

SHAPES = {
    "spheroid_1_0p7_0p7": (1.0, 0.7, 0.7),
    "ellipsoid_1_0p8_0p6": (1.0, 0.8, 0.6),
}

REQ = 1.0
RHO_F = 1.0
NBODY = 2
ROTATIONS_PER_TIME = 1.0

PRIMARY_RHO = 4.0
PRIMARY_ENERGY_RATIO = 1.0
PRIMARY_SEPARATIONS = [3.0, 5.0, 8.0]
PRIMARY_TEMPORAL_NDIV = 3
PRIMARY_TEMPORAL_DTS = [0.2, 0.1, 0.05, 0.025]
PRIMARY_GRID_NDIVS = [1, 2, 3, 4]
PRIMARY_GRID_DT = 0.05

STRESS_SHAPES = ["spheroid_1_0p7_0p7"]
STRESS_RHOS = [1.0, 0.1]
STRESS_ENERGY_RATIO = 0.25
STRESS_SEPARATIONS = [3.0, 8.0]
STRESS_TEMPORAL_NDIV = 2
STRESS_TEMPORAL_DTS = [0.1, 0.05, 0.025, 0.0125]
STRESS_GRID_NDIVS = [1, 2, 3]
STRESS_GRID_DT = 0.025

T_END = 10.0
T_PRINT = 1

ORIENTATION_1 = (1.0, 2.0, 0.0, 0.0)
ORIENTATION_2 = (1.0, 0.0, 1.0, 0.0)

Vector3 = tuple[float, float, float]


@dataclass
class Case:
    family: str
    shape_name: str
    rho: float
    energy_ratio: float
    separation: float
    ndiv: int
    dt: float
    suites: set[str] = field(default_factory=set)
    method: str = "impulse_noproj"
    repeat: int = 1
    tend: float = T_END


def label_float(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def normalize(v: Vector3) -> Vector3:
    n = math.sqrt(sum(x * x for x in v))
    if n == 0.0:
        raise ValueError("zero vector")
    return tuple(x / n for x in v)


def normalized_shape(shape: Vector3, req: float = REQ) -> Vector3:
    scale = req / (shape[0] * shape[1] * shape[2]) ** (1.0 / 3.0)
    return tuple(scale * x for x in shape)


def random_unit(rng: random.Random) -> Vector3:
    z = rng.uniform(-1.0, 1.0)
    theta = rng.uniform(0.0, 2.0 * math.pi)
    r = math.sqrt(max(0.0, 1.0 - z * z))
    return (r * math.cos(theta), r * math.sin(theta), z)


def case_seed(case: Case) -> int:
    text = (
        f"{case.family}|{case.shape_name}|{case.rho}|{case.energy_ratio}|"
        f"{case.separation}|{case.ndiv}|{case.dt}|{case.repeat}"
    )
    seed = 0x4D554C54
    for ch in text:
        seed = ((seed * 1000003) ^ ord(ch)) & 0xFFFFFFFF
    return seed


def ellipsoid_mass(shape: Vector3, density: float) -> float:
    a, b, c = shape
    return density * (4.0 / 3.0) * math.pi * a * b * c


def inertia_diag(shape: Vector3, mass: float) -> Vector3:
    a, b, c = shape
    return (
        0.2 * mass * (b * b + c * c),
        0.2 * mass * (a * a + c * c),
        0.2 * mass * (a * a + b * b),
    )


def spin_for_axis(axis: Vector3) -> Vector3:
    unit = normalize(axis)
    omega_mag = 2.0 * math.pi * ROTATIONS_PER_TIME
    return tuple(omega_mag * x for x in unit)


def speed_for_ratio(shape: Vector3, density: float, energy_ratio: float, spin: Vector3) -> float:
    solver_shape = normalized_shape(shape)
    mass = ellipsoid_mass(solver_shape, density)
    inertia = inertia_diag(solver_shape, mass)
    rot_ke = 0.5 * sum(inertia[i] * spin[i] * spin[i] for i in range(3))
    target_lin_ke = rot_ke / energy_ratio
    return math.sqrt(2.0 * target_lin_ke / mass)


def case_key(case: Case) -> tuple[Any, ...]:
    return (
        case.family,
        case.method,
        case.shape_name,
        case.rho,
        case.energy_ratio,
        case.separation,
        case.repeat,
        case.ndiv,
        case.dt,
        case.tend,
    )


def add_case(case_map: dict[tuple[Any, ...], Case], case: Case, suite: str) -> None:
    key = case_key(case)
    if key not in case_map:
        case_map[key] = case
    case_map[key].suites.add(suite)


def cases() -> list[Case]:
    case_map: dict[tuple[Any, ...], Case] = {}

    for shape_name, separation in itertools.product(SHAPES, PRIMARY_SEPARATIONS):
        for dt in PRIMARY_TEMPORAL_DTS:
            add_case(
                case_map,
                Case(
                    family="periodic",
                    shape_name=shape_name,
                    rho=PRIMARY_RHO,
                    energy_ratio=PRIMARY_ENERGY_RATIO,
                    separation=separation,
                    ndiv=PRIMARY_TEMPORAL_NDIV,
                    dt=dt,
                ),
                "temporal",
            )
        for ndiv in PRIMARY_GRID_NDIVS:
            add_case(
                case_map,
                Case(
                    family="periodic",
                    shape_name=shape_name,
                    rho=PRIMARY_RHO,
                    energy_ratio=PRIMARY_ENERGY_RATIO,
                    separation=separation,
                    ndiv=ndiv,
                    dt=PRIMARY_GRID_DT,
                ),
                "grid",
            )

    for shape_name, rho, separation in itertools.product(STRESS_SHAPES, STRESS_RHOS, STRESS_SEPARATIONS):
        for dt in STRESS_TEMPORAL_DTS:
            add_case(
                case_map,
                Case(
                    family="stress",
                    shape_name=shape_name,
                    rho=rho,
                    energy_ratio=STRESS_ENERGY_RATIO,
                    separation=separation,
                    ndiv=STRESS_TEMPORAL_NDIV,
                    dt=dt,
                ),
                "temporal",
            )
        for ndiv in STRESS_GRID_NDIVS:
            add_case(
                case_map,
                Case(
                    family="stress",
                    shape_name=shape_name,
                    rho=rho,
                    energy_ratio=STRESS_ENERGY_RATIO,
                    separation=separation,
                    ndiv=ndiv,
                    dt=STRESS_GRID_DT,
                ),
                "grid",
            )

    return sorted(
        case_map.values(),
        key=lambda c: (
            c.family,
            c.shape_name,
            c.rho,
            c.energy_ratio,
            c.separation,
            c.ndiv,
            c.dt,
        ),
    )


def case_name(case: Case) -> str:
    suite_label = "both" if case.suites == {"temporal", "grid"} else sorted(case.suites)[0]
    return (
        f"{case.family}_{suite_label}_{case.shape_name}"
        f"_rho{label_float(case.rho)}"
        f"_E{label_float(case.energy_ratio)}"
        f"_sep{label_float(case.separation)}"
        f"_nd{case.ndiv}"
        f"_dt{label_float(case.dt)}"
    )


def method_values(case: Case) -> dict[str, Any]:
    if case.method != "impulse_noproj":
        raise ValueError(f"unsupported method {case.method!r}")
    return {
        "impulse_scheme": 1,
        "energy_projection": 0,
        "impulse_pair_metric_correction": 0,
    }


def input_values(case: Case) -> dict[str, Any]:
    shape = SHAPES[case.shape_name]
    rng = random.Random(case_seed(case))
    velocity_direction = random_unit(rng)
    spin1 = spin_for_axis(random_unit(rng))
    spin2 = spin_for_axis(random_unit(rng))
    speed1 = speed_for_ratio(shape, case.rho, case.energy_ratio, spin1)
    speed2 = speed_for_ratio(shape, case.rho, case.energy_ratio, spin2)
    linear_velocity1 = tuple(speed1 * x for x in velocity_direction)
    linear_velocity2 = tuple(speed2 * x for x in velocity_direction)
    nsteps = int(round(case.tend / case.dt))

    values: dict[str, Any] = {
        "cex1": -0.5 * case.separation,
        "cey1": 0.0,
        "cez1": 0.0,
        "oriw1": ORIENTATION_1[0],
        "orii1": ORIENTATION_1[1],
        "orij1": ORIENTATION_1[2],
        "orik1": ORIENTATION_1[3],
        "lvx1": linear_velocity1[0],
        "lvy1": linear_velocity1[1],
        "lvz1": linear_velocity1[2],
        "avx1": spin1[0],
        "avy1": spin1[1],
        "avz1": spin1[2],
        "shx1": shape[0],
        "shy1": shape[1],
        "shz1": shape[2],
        "req1": REQ,
        "rhos1": case.rho,
        "cex2": 0.5 * case.separation,
        "cey2": 0.0,
        "cez2": 0.0,
        "oriw2": ORIENTATION_2[0],
        "orii2": ORIENTATION_2[1],
        "orij2": ORIENTATION_2[2],
        "orik2": ORIENTATION_2[3],
        "lvx2": linear_velocity2[0],
        "lvy2": linear_velocity2[1],
        "lvz2": linear_velocity2[2],
        "avx2": spin2[0],
        "avy2": spin2[1],
        "avz2": spin2[2],
        "shx2": shape[0],
        "shy2": shape[1],
        "shz2": shape[2],
        "req2": REQ,
        "rhos2": case.rho,
        "rhof": RHO_F,
        "ndiv": case.ndiv,
        "tend": case.tend,
        "dt": case.dt,
        "tprint": T_PRINT,
        "logevery": max(1, nsteps // 10),
        "nbody": NBODY,
    }
    values.update(method_values(case))
    return values


def write_input(path: Path, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for key, value in values.items():
            f.write(f"{key}={value}\n")


def format_manifest_path(path: Path, portable: bool) -> str:
    return path.relative_to(ROOT).as_posix() if portable else str(path)


def write_manifest(case_list: list[Case], portable: bool) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "index",
        "name",
        "family",
        "suites",
        "method",
        "shape_name",
        "rho",
        "energy_ratio",
        "separation",
        "repeat",
        "ndiv",
        "dt",
        "tend",
        "input",
        "output",
    ]
    with MANIFEST.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, case in enumerate(case_list, start=1):
            name = case_name(case)
            run_dir = RUNS / name
            writer.writerow(
                {
                    "index": idx,
                    "name": name,
                    "family": case.family,
                    "suites": ";".join(sorted(case.suites)),
                    "method": case.method,
                    "shape_name": case.shape_name,
                    "rho": case.rho,
                    "energy_ratio": case.energy_ratio,
                    "separation": case.separation,
                    "repeat": case.repeat,
                    "ndiv": case.ndiv,
                    "dt": case.dt,
                    "tend": case.tend,
                    "input": format_manifest_path(run_dir / "input.txt", portable),
                    "output": format_manifest_path(run_dir / "multiple_body_complete.dat", portable),
                }
            )


def setup(case_list: list[Case], portable_manifest: bool) -> None:
    for case in case_list:
        write_input(RUNS / case_name(case) / "input.txt", input_values(case))
    write_manifest(case_list, portable_manifest)


def print_plan(case_list: list[Case]) -> None:
    n_periodic = sum(1 for case in case_list if case.family == "periodic")
    n_stress = sum(1 for case in case_list if case.family == "stress")
    n_ndiv4 = sum(1 for case in case_list if case.ndiv == 4)
    print("Multibody convergence sweep")
    print(f"  Study dir:        {STUDY}")
    print(f"  Manifest:         {MANIFEST}")
    print(f"  Output root:      {RUNS}")
    print("  Method:           impulse_scheme=1, energy_projection=0")
    print(f"  End time:         {T_END}")
    print(f"  Spin rate:        {ROTATIONS_PER_TIME:g} rotation per unit time")
    print(f"  Primary cases:    {n_periodic} high-density self-convergence cases")
    print(f"  Stress cases:     {n_stress} close/far low-energy cases")
    print(f"  ndiv=4 cases:     {n_ndiv4}")
    print(f"  Total cases:      {len(case_list)}")
    print()
    print("Primary sweep:")
    print(f"  shapes:           {', '.join(SHAPES)}")
    print(f"  rho, E:           {PRIMARY_RHO}, {PRIMARY_ENERGY_RATIO}")
    print(f"  separations:      {PRIMARY_SEPARATIONS}")
    print(f"  temporal:         ndiv={PRIMARY_TEMPORAL_NDIV}, dt={PRIMARY_TEMPORAL_DTS}")
    print(f"  grid:             dt={PRIMARY_GRID_DT}, ndiv={PRIMARY_GRID_NDIVS}")
    print()
    print("Stress sweep:")
    print(f"  shapes:           {', '.join(STRESS_SHAPES)}")
    print(f"  rho, E:           {STRESS_RHOS}, {STRESS_ENERGY_RATIO}")
    print(f"  separations:      {STRESS_SEPARATIONS}")
    print(f"  temporal:         ndiv={STRESS_TEMPORAL_NDIV}, dt={STRESS_TEMPORAL_DTS}")
    print(f"  grid:             dt={STRESS_GRID_DT}, ndiv={STRESS_GRID_NDIVS}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up the multibody convergence sweep.")
    parser.add_argument("--write", action="store_true", help="write input folders and manifest")
    parser.add_argument(
        "--portable-manifest",
        action="store_true",
        help="write manifest input/output paths relative to the repository root",
    )
    args = parser.parse_args()

    case_list = cases()
    print_plan(case_list)
    if args.write:
        setup(case_list, args.portable_manifest)
        print()
        print(f"Wrote {len(case_list)} inputs and manifest.")
    else:
        print()
        print("Dry run only. Re-run with --write after confirming the matrix.")


if __name__ == "__main__":
    main()
