import argparse
import csv
import itertools
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "studies" / "two_body_paper_sweeps"
RUNS = STUDY / "runs"

SHAPES = {
    "spheroid_1_0p7_0p7": (1.0, 0.7, 0.7),
    "ellipsoid_1_0p8_0p6": (1.0, 0.8, 0.6),
}

DENSITIES = [10.0, 4.0, 1.0, 0.3, 0.1, 0.03]
ENERGY_RATIOS = [0.2, 0.55, 1.5, 4.0, 11.0, 30.0]
SEPARATIONS = [3.0, 5.0, 8.0, 11.0]
REPEATS = 2

NDIV = 2
DT = 0.05
T_END = 100.0
T_PRINT = 1
REQ = 1.0
RHO_F = 1.0
NBODY = 2
ROTATIONS_OVER_RUN = 100.0

ORIENTATIONS = [
    ((1.0, 2.0, 0.0, 0.0), (1.0, 0.0, 1.0, 0.0)),
    ((1.0, 1.0, 1.0, 0.0), (1.0, -1.0, 0.0, 1.0)),
]

Vector3 = Tuple[float, float, float]


class Case:
    def __init__(
        self,
        shape_name: str,
        rho: float,
        energy_ratio: float,
        separation: float,
        repeat: int,
        ndiv: int = NDIV,
        dt: float = DT,
        tend: float = T_END,
    ) -> None:
        self.shape_name = shape_name
        self.rho = rho
        self.energy_ratio = energy_ratio
        self.separation = separation
        self.repeat = repeat
        self.ndiv = ndiv
        self.dt = dt
        self.tend = tend


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
        f"{case.shape_name}|{case.rho}|{case.energy_ratio}|"
        f"{case.separation}|{case.repeat}|{case.ndiv}|{case.dt}|{case.tend}"
    )
    seed = 0x32505745
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


def spin_for_rotation_count(axis: Vector3, rotations: float, tend: float) -> Vector3:
    unit = normalize(axis)
    omega_mag = 2.0 * math.pi * rotations / tend
    return tuple(omega_mag * x for x in unit)


def speed_for_ratio(shape: Vector3, density: float, energy_ratio: float, spin: Vector3) -> float:
    solver_shape = normalized_shape(shape)
    mass = ellipsoid_mass(solver_shape, density)
    inertia = inertia_diag(solver_shape, mass)
    rot_ke = 0.5 * sum(inertia[i] * spin[i] * spin[i] for i in range(3))
    target_lin_ke = rot_ke / energy_ratio
    return math.sqrt(2.0 * target_lin_ke / mass)


def selected_shapes(shape: str) -> List[str]:
    if shape == "all":
        return list(SHAPES)
    if shape not in SHAPES:
        raise ValueError(f"unknown shape {shape!r}")
    return [shape]


def cases(shape: str) -> List[Case]:
    out: List[Case] = []
    for shape_name, rho, energy_ratio, separation, repeat in itertools.product(
        selected_shapes(shape),
        DENSITIES,
        ENERGY_RATIOS,
        SEPARATIONS,
        range(1, REPEATS + 1),
    ):
        out.append(
            Case(
                shape_name=shape_name,
                rho=rho,
                energy_ratio=energy_ratio,
                separation=separation,
                repeat=repeat,
            )
        )
    return out


def case_name(case: Case) -> str:
    return (
        f"{case.shape_name}"
        f"_rho{label_float(case.rho)}"
        f"_E{label_float(case.energy_ratio)}"
        f"_sep{label_float(case.separation)}"
        f"_run{case.repeat:02d}"
        f"_impulse_noproj"
    )


def run_dir(case: Case) -> Path:
    return RUNS / case.shape_name / case_name(case)


def input_values(case: Case) -> Dict[str, Any]:
    shape = SHAPES[case.shape_name]
    idx = case.repeat - 1
    ori1, ori2 = ORIENTATIONS[idx % len(ORIENTATIONS)]
    rng = random.Random(case_seed(case))
    velocity_direction = random_unit(rng)
    axis1 = random_unit(rng)
    axis2 = random_unit(rng)
    spin1 = spin_for_rotation_count(axis1, ROTATIONS_OVER_RUN, case.tend)
    spin2 = spin_for_rotation_count(axis2, ROTATIONS_OVER_RUN, case.tend)
    speed1 = speed_for_ratio(shape, case.rho, case.energy_ratio, spin1)
    speed2 = speed_for_ratio(shape, case.rho, case.energy_ratio, spin2)
    linear_velocity1 = tuple(speed1 * x for x in velocity_direction)
    linear_velocity2 = tuple(speed2 * x for x in velocity_direction)
    nsteps = int(round(case.tend / case.dt))

    return {
        "cex1": -0.5 * case.separation,
        "cey1": 0.0,
        "cez1": 0.0,
        "oriw1": ori1[0],
        "orii1": ori1[1],
        "orij1": ori1[2],
        "orik1": ori1[3],
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
        "oriw2": ori2[0],
        "orii2": ori2[1],
        "orij2": ori2[2],
        "orik2": ori2[3],
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
        "logevery": max(1, nsteps // 20),
        "nbody": NBODY,
        "impulse_scheme": 1,
        "energy_projection": 0,
        "impulse_pair_metric_correction": 0,
    }


def write_input(path: Path, values: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for key, value in values.items():
            f.write(f"{key}={value}\n")


def manifest_path(shape: str) -> Path:
    return STUDY / f"manifest_{shape}.csv"


def format_path(path: Path, portable: bool) -> str:
    return path.relative_to(ROOT).as_posix() if portable else str(path)


def write_manifest(case_list: List[Case], shape: str, portable: bool) -> Path:
    path = manifest_path(shape)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "index",
        "name",
        "shape_name",
        "rho",
        "energy_ratio",
        "separation",
        "repeat",
        "solver_mode",
        "ndiv",
        "dt",
        "tend",
        "input",
        "output",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, case in enumerate(case_list, start=1):
            directory = run_dir(case)
            writer.writerow(
                {
                    "index": idx,
                    "name": case_name(case),
                    "shape_name": case.shape_name,
                    "rho": case.rho,
                    "energy_ratio": case.energy_ratio,
                    "separation": case.separation,
                    "repeat": case.repeat,
                    "solver_mode": "impulse_noproj",
                    "ndiv": case.ndiv,
                    "dt": case.dt,
                    "tend": case.tend,
                    "input": format_path(directory / "input.txt", portable),
                    "output": format_path(directory / "multiple_body_complete.dat", portable),
                }
            )
    return path


def setup(case_list: List[Case], shape: str, portable: bool) -> Path:
    for case in case_list:
        write_input(run_dir(case) / "input.txt", input_values(case))
    return write_manifest(case_list, shape, portable)


def print_plan(shape: str, case_list: List[Case], manifest: Path) -> None:
    shape_names = sorted({case.shape_name for case in case_list})
    print("Two-body paper sweep")
    print(f"  Study dir:        {STUDY}")
    print(f"  Manifest:         {manifest}")
    print(f"  Shapes:           {', '.join(shape_names)}")
    print(f"  Densities rho:    {DENSITIES}")
    print(f"  Energy ratios E:  {ENERGY_RATIOS}")
    print(f"  Separations:      {SEPARATIONS}")
    print(f"  Repeats:          {REPEATS}")
    print("  Method:           impulse_scheme=1, energy_projection=0")
    print(f"  Mesh/time:        ndiv={NDIV}, dt={DT}, tend={T_END}")
    print(f"  Spin magnitude:   {ROTATIONS_OVER_RUN:g} rotations over tend")
    print(f"  Selected shape:   {shape}")
    print(f"  Total cases:      {len(case_list)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up two-body paper sweeps for spheroids/ellipsoids.")
    parser.add_argument("--shape", choices=["all"] + list(SHAPES.keys()), default="all")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--portable-manifest", action="store_true")
    args = parser.parse_args()

    case_list = cases(args.shape)
    manifest = manifest_path(args.shape)
    print_plan(args.shape, case_list, manifest)
    if args.write:
        written = setup(case_list, args.shape, args.portable_manifest)
        print()
        print(f"Wrote {len(case_list)} inputs and manifest: {written}")
    else:
        print()
        print("Dry run only. Re-run with --write after confirming the matrix.")


if __name__ == "__main__":
    main()
