import argparse
import csv
import itertools
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "studies" / "departing_sphericity_sweep"
RUNS = STUDY / "runs"

FAMILY_NAMES = ("spheroid_prolate", "spheroid_oblate", "triaxial")
SWEEP_NAMES = ("homogeneous", "heterogeneous_sphere", "heterogeneous_aspherical")
EPSILONS = [0.0, 1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1, 3.0e-1, 1.0]
REPEATS = 10

RHO = 1.0
ENERGY_RATIO = 0.5
SEPARATION = 8.0

NDIV = 2
DT = 0.05
T_END = 100.0
T_PRINT = 1
REQ = 1.0
RHO_F = 1.0
NBODY = 2
ROTATIONS_OVER_RUN = 100.0

Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]


class Case:
    def __init__(self, family: str, suite: str, epsilon: float, repeat: int) -> None:
        self.family = family
        self.suite = suite
        self.epsilon = epsilon
        self.repeat = repeat
        self.rho = RHO
        self.energy_ratio = ENERGY_RATIO
        self.separation = SEPARATION
        self.ndiv = NDIV
        self.dt = DT
        self.tend = T_END


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


def random_orientation(rng: random.Random) -> Quaternion:
    u1 = rng.random()
    u2 = rng.random()
    u3 = rng.random()
    s1 = math.sqrt(1.0 - u1)
    s2 = math.sqrt(u1)
    theta1 = 2.0 * math.pi * u2
    theta2 = 2.0 * math.pi * u3
    i = s1 * math.sin(theta1)
    j = s1 * math.cos(theta1)
    k = s2 * math.sin(theta2)
    w = s2 * math.cos(theta2)
    return (w, i, j, k)


def shape_for(family: str, epsilon: float) -> Vector3:
    lam = 1.0 + epsilon
    if family == "spheroid_prolate":
        return (lam * lam, 1.0 / lam, 1.0 / lam)
    if family == "spheroid_oblate":
        return (lam, lam, 1.0 / (lam * lam))
    if family == "triaxial":
        return (lam, 1.0, 1.0 / lam)
    raise ValueError("unknown family {!r}".format(family))


def eps_pair(case: Case) -> Tuple[float, float]:
    if case.suite == "homogeneous":
        return case.epsilon, case.epsilon
    if case.suite == "heterogeneous_sphere":
        return case.epsilon, 0.0
    if case.suite == "heterogeneous_aspherical":
        return case.epsilon, 1.0
    raise ValueError("unknown suite {!r}".format(case.suite))


def case_seed(case: Case) -> int:
    text = "{}|{}|{}|{}|{}|{}|{}|{}|{}".format(
        case.family,
        case.suite,
        case.epsilon,
        case.repeat,
        case.rho,
        case.energy_ratio,
        case.separation,
        case.ndiv,
        case.dt,
    )
    seed = 0xD5F34219
    for ch in text:
        seed = ((seed * 1000003) ^ ord(ch)) & 0xFFFFFFFF
    return seed


def orientation_seed(case: Case) -> int:
    return case_seed(case) ^ 0xA5366B4D


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


def selected(values: Tuple[str, ...], selection: str) -> List[str]:
    if selection == "all":
        return list(values)
    if selection not in values:
        raise ValueError("unknown selection {!r}".format(selection))
    return [selection]


def cases(family: str, suite: str) -> List[Case]:
    out: List[Case] = []
    for family_name, suite_name, epsilon, repeat in itertools.product(
        selected(FAMILY_NAMES, family),
        selected(SWEEP_NAMES, suite),
        EPSILONS,
        range(1, REPEATS + 1),
    ):
        out.append(Case(family_name, suite_name, epsilon, repeat))
    return out


def case_name(case: Case) -> str:
    return "{}_{}_eps{}_rho{}_E{}_sep{}_run{:02d}_impulse_noproj".format(
        case.family,
        case.suite,
        label_float(case.epsilon),
        label_float(case.rho),
        label_float(case.energy_ratio),
        label_float(case.separation),
        case.repeat,
    )


def run_dir(case: Case) -> Path:
    return RUNS / case.family / case.suite / case_name(case)


def input_values(case: Case) -> Dict[str, Any]:
    eps1, eps2 = eps_pair(case)
    shape1 = shape_for(case.family, eps1)
    shape2 = shape_for(case.family, eps2)
    rng = random.Random(case_seed(case))
    orientation_rng = random.Random(orientation_seed(case))
    ori1 = random_orientation(orientation_rng)
    ori2 = random_orientation(orientation_rng)
    velocity_direction = random_unit(rng)
    spin1 = spin_for_rotation_count(random_unit(rng), ROTATIONS_OVER_RUN, case.tend)
    spin2 = spin_for_rotation_count(random_unit(rng), ROTATIONS_OVER_RUN, case.tend)
    speed1 = speed_for_ratio(shape1, case.rho, case.energy_ratio, spin1)
    speed2 = speed_for_ratio(shape2, case.rho, case.energy_ratio, spin2)
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
        "shx1": shape1[0],
        "shy1": shape1[1],
        "shz1": shape1[2],
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
        "shx2": shape2[0],
        "shy2": shape2[1],
        "shz2": shape2[2],
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
            f.write("{}={}\n".format(key, value))


def manifest_path(family: str, suite: str) -> Path:
    return STUDY / "manifest_{}_{}.csv".format(family, suite)


def format_path(path: Path, portable: bool) -> str:
    return path.relative_to(ROOT).as_posix() if portable else str(path)


def write_manifest(case_list: List[Case], family: str, suite: str, portable: bool) -> Path:
    path = manifest_path(family, suite)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "index",
        "name",
        "family",
        "suite",
        "epsilon",
        "epsilon_body1",
        "epsilon_body2",
        "shape1_x",
        "shape1_y",
        "shape1_z",
        "shape2_x",
        "shape2_y",
        "shape2_z",
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
            eps1, eps2 = eps_pair(case)
            shape1 = shape_for(case.family, eps1)
            shape2 = shape_for(case.family, eps2)
            writer.writerow(
                {
                    "index": idx,
                    "name": case_name(case),
                    "family": case.family,
                    "suite": case.suite,
                    "epsilon": case.epsilon,
                    "epsilon_body1": eps1,
                    "epsilon_body2": eps2,
                    "shape1_x": shape1[0],
                    "shape1_y": shape1[1],
                    "shape1_z": shape1[2],
                    "shape2_x": shape2[0],
                    "shape2_y": shape2[1],
                    "shape2_z": shape2[2],
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


def setup(case_list: List[Case], family: str, suite: str, portable: bool) -> Path:
    for case in case_list:
        write_input(run_dir(case) / "input.txt", input_values(case))
    return write_manifest(case_list, family, suite, portable)


def print_plan(family: str, suite: str, case_list: List[Case], manifest: Path) -> None:
    families = sorted({case.family for case in case_list})
    suites = sorted({case.suite for case in case_list})
    print("Departing-sphericity two-body sweep")
    print("  Study dir:        {}".format(STUDY))
    print("  Manifest:         {}".format(manifest))
    print("  Families:         {}".format(", ".join(families)))
    print("  Suites:           {}".format(", ".join(suites)))
    print("  eps values:       {}".format(EPSILONS))
    print("  rho, E, sep:      {}, {}, {}".format(RHO, ENERGY_RATIO, SEPARATION))
    print("  Repeats:          {}".format(REPEATS))
    print("  Method:           impulse_scheme=1, energy_projection=0")
    print("  Mesh/time:        ndiv={}, dt={}, tend={}".format(NDIV, DT, T_END))
    print("  Spin magnitude:   {} rotations over tend".format(ROTATIONS_OVER_RUN))
    print("  Initial orient.:  deterministic-random unit quaternions per body/repeat")
    print("  Selected family:  {}".format(family))
    print("  Selected suite:   {}".format(suite))
    print("  Total cases:      {}".format(len(case_list)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up departing-sphericity two-body sweeps.")
    parser.add_argument("--family", choices=["all"] + list(FAMILY_NAMES), default="all")
    parser.add_argument("--suite", choices=["all"] + list(SWEEP_NAMES), default="all")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--portable-manifest", action="store_true")
    args = parser.parse_args()

    case_list = cases(args.family, args.suite)
    manifest = manifest_path(args.family, args.suite)
    print_plan(args.family, args.suite, case_list, manifest)
    if args.write:
        written = setup(case_list, args.family, args.suite, args.portable_manifest)
        print()
        print("Wrote {} inputs and manifest: {}".format(len(case_list), written))
    else:
        print()
        print("Dry run only. Re-run with --write after confirming the matrix.")


if __name__ == "__main__":
    main()
