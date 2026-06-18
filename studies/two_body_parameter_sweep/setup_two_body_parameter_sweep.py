import argparse
import csv
import itertools
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "studies" / "two_body_parameter_sweep"
RUNS = STUDY / "runs"
COUPLED_RUNS = ROOT / "two_body_parameter_sweep_coupled_runs"
MANIFEST = STUDY / "two_body_parameter_sweep_manifest.csv"
COUPLED_MANIFEST = COUPLED_RUNS / "two_body_parameter_sweep_manifest.csv"

# Corner pilot before launching the full production matrix.
SHAPES = {
    "spheroid_1_0p7_0p7": (1.0, 0.7, 0.7),
    "ellipsoid_1_0p8_0p6": (1.0, 0.8, 0.6),
}
DENSITIES = [1.0, 0.1, 0.03]
ENERGY_RATIOS = [0.25, 4.0, 16.0]
SEPARATIONS = [3.0, 8.0]
REPEATS = 2

NDIV = 2
DT = 0.05
T_END = 100.0
REQ = 1.0
RHO_F = 1.0
NBODY = 2
DEFAULT_SOLVER_MODE = "coupled_endpoint"

SOLVER_MODES = ("impulse_projection", "coupled_endpoint")

ROTATIONS_OVER_RUN = 100.0
ORIENTATIONS = [
    ((1.0, 2.0, 0.0, 0.0), (1.0, 0.0, 1.0, 0.0)),
    ((1.0, 1.0, 1.0, 0.0), (1.0, -1.0, 0.0, 1.0)),
]


Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]
Case = Dict[str, Any]


def label_float(value: float) -> str:
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    return text


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
        f"{case['shape_name']}|{case['rho']}|{case['energy_ratio']}|"
        f"{case['separation']}|{case['repeat']}"
    )
    seed = 0x345678
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


def spin_for_rotation_count(
    axis: Vector3,
    rotations: float = ROTATIONS_OVER_RUN,
    tend: float = T_END,
) -> Vector3:
    unit = normalize(axis)
    omega_mag = 2.0 * math.pi * rotations / tend
    return tuple(omega_mag * x for x in unit)


def speed_for_ratio(
    shape: Vector3,
    density: float,
    energy_ratio: float,
    spin: Vector3,
) -> float:
    solver_shape = normalized_shape(shape)
    mass = ellipsoid_mass(solver_shape, density)
    inertia = inertia_diag(solver_shape, mass)
    rot_ke = 0.5 * sum(inertia[i] * spin[i] * spin[i] for i in range(3))
    target_lin_ke = rot_ke / energy_ratio
    return math.sqrt(2.0 * target_lin_ke / mass)


def case_name(case: Case, solver_mode: str = "impulse_projection") -> str:
    name = (
        f"{case['shape_name']}"
        f"_rho{label_float(float(case['rho']))}"
        f"_E{label_float(float(case['energy_ratio']))}"
        f"_sep{label_float(float(case['separation']))}"
        f"_run{int(case['repeat']):02d}"
    )
    if solver_mode != "impulse_projection":
        name += f"_{solver_mode}"
    return name


def cases() -> List[Case]:
    out: List[Case] = []
    for shape_name, rho, energy_ratio, separation, repeat in itertools.product(
        SHAPES, DENSITIES, ENERGY_RATIOS, SEPARATIONS, range(1, REPEATS + 1)
    ):
        out.append(
            {
                "shape_name": shape_name,
                "shape": SHAPES[shape_name],
                "rho": rho,
                "energy_ratio": energy_ratio,
                "separation": separation,
                "repeat": repeat,
            }
        )
    return out


def default_runs_root(solver_mode: str) -> Path:
    if solver_mode == "coupled_endpoint":
        return COUPLED_RUNS
    return RUNS


def default_manifest_path(solver_mode: str) -> Path:
    if solver_mode == "coupled_endpoint":
        return COUPLED_MANIFEST
    return MANIFEST


def solver_values(solver_mode: str) -> Dict[str, Any]:
    if solver_mode == "impulse_projection":
        return {
            "impulse_scheme": 1,
            "energy_projection": 1,
        }
    if solver_mode == "coupled_endpoint":
        return {
            "impulse_scheme": 1,
            "energy_projection": 0,
            "fluid_energy_gradient": 0,
            "hamiltonian_midpoint_scheme": 1,
            "hamiltonian_coupled_solve": 1,
            "hamiltonian_coupled_iters": 6,
            "hamiltonian_coupled_eps": 0.001,
            "hamiltonian_coupled_max_shift": 0.2,
            "hamiltonian_coupled_jacobian_interval": 6,
            "hamiltonian_coupled_broyden_update": 1,
            "hamiltonian_coupled_endpoint_velocity": 1,
            "hamiltonian_adaptive_substeps": 1,
            "hamiltonian_max_substeps": 8,
            "hamiltonian_floor_tol": 0.0001,
        }
    raise ValueError(f"unknown solver mode {solver_mode!r}")


def input_values(case: Case, solver_mode: str) -> Dict[str, Any]:
    shape = case["shape"]
    assert isinstance(shape, tuple)
    rho = float(case["rho"])
    energy_ratio = float(case["energy_ratio"])
    separation = float(case["separation"])
    repeat = int(case["repeat"])
    idx = repeat - 1
    ori1, ori2 = ORIENTATIONS[idx % len(ORIENTATIONS)]
    rng = random.Random(case_seed(case))
    velocity_direction = random_unit(rng)
    axis1 = random_unit(rng)
    axis2 = random_unit(rng)
    spin1 = spin_for_rotation_count(axis1)
    spin2 = spin_for_rotation_count(axis2)
    speed1 = speed_for_ratio(shape, rho, energy_ratio, spin1)
    speed2 = speed_for_ratio(shape, rho, energy_ratio, spin2)
    linear_velocity1 = tuple(speed1 * x for x in velocity_direction)
    linear_velocity2 = tuple(speed2 * x for x in velocity_direction)

    values: Dict[str, Any] = {
        "cex1": -0.5 * separation,
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
        "rhos1": rho,
        "cex2": 0.5 * separation,
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
        "rhos2": rho,
        "rhof": RHO_F,
        "ndiv": NDIV,
        "tend": T_END,
        "dt": DT,
        "tprint": 1,
        "logevery": 100,
        "nbody": NBODY,
    }
    values.update(solver_values(solver_mode))
    return values


def write_input(path: Path, values: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for key, value in values.items():
            f.write(f"{key}={value}\n")


def format_manifest_path(path: Path, portable: bool) -> str:
    if portable:
        return path.relative_to(ROOT).as_posix()
    return str(path)


def write_manifest(
    case_list: List[Case],
    solver_mode: str,
    manifest_path: Path,
    runs_root: Path,
    portable: bool = False,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
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
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for case in case_list:
            name = case_name(case, solver_mode)
            run_dir = runs_root / name
            writer.writerow(
                {
                    "name": name,
                    "shape_name": case["shape_name"],
                    "rho": case["rho"],
                    "energy_ratio": case["energy_ratio"],
                    "separation": case["separation"],
                    "repeat": case["repeat"],
                    "solver_mode": solver_mode,
                    "ndiv": NDIV,
                    "dt": DT,
                    "tend": T_END,
                    "input": format_manifest_path(run_dir / "input.txt", portable),
                    "output": format_manifest_path(run_dir / "multiple_body_complete.dat", portable),
                }
            )


def setup_inputs(
    case_list: List[Case],
    solver_mode: str,
    manifest_path: Path,
    runs_root: Path,
    portable_manifest: bool = False,
) -> None:
    for case in case_list:
        run_dir = runs_root / case_name(case, solver_mode)
        write_input(run_dir / "input.txt", input_values(case, solver_mode))
    write_manifest(
        case_list,
        solver_mode,
        manifest_path,
        runs_root,
        portable=portable_manifest,
    )


def print_plan(
    case_list: List[Case],
    solver_mode: str,
    manifest_path: Path,
    runs_root: Path,
) -> None:
    print("Two-body parameter sweep proposal")
    print(f"  Study dir:       {STUDY}")
    print(f"  Runs dir:        {runs_root}")
    print(f"  Manifest:        {manifest_path}")
    print(f"  Shapes:          {', '.join(SHAPES)}")
    print(f"  Densities rho:   {DENSITIES}")
    print(f"  Energy ratios E: {ENERGY_RATIOS}  (rotational KE / translational KE per body)")
    print(f"  Separations:     {SEPARATIONS} centre distances")
    print(f"  Repeats:         {REPEATS} per (shape, rho, E, separation)")
    print(f"  Mesh/time:       ndiv={NDIV}, dt={DT}, tend={T_END}")
    print(f"  Spin magnitude:  {ROTATIONS_OVER_RUN:g} rotations over tend")
    print(f"  Solver mode:     {solver_mode}")
    if solver_mode == "impulse_projection":
        print("  Coupling:        impulse_scheme=1, energy_projection=1")
    else:
        print("  Coupling:        Hamiltonian midpoint + coupled endpoint velocity")
    print("  Initial v:       both bodies parallel, deterministic-random direction; speed set by E")
    print("  Initial omega:   deterministic-random directions, fixed total rotation count")
    print(f"  Total cases:     {len(case_list)}")
    print()
    print("First 8 case names:")
    for case in case_list[:8]:
        print(f"  {case_name(case, solver_mode)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up the two-body E/rho/separation parameter sweep.")
    parser.add_argument("--write", action="store_true", help="write input folders and manifest")
    parser.add_argument(
        "--portable-manifest",
        action="store_true",
        help="write manifest input/output paths relative to the repository root",
    )
    parser.add_argument(
        "--solver-mode",
        choices=SOLVER_MODES,
        default=DEFAULT_SOLVER_MODE,
        help=(
            "solver settings to write. coupled_endpoint uses the energy-conserving "
            "Hamiltonian midpoint endpoint-velocity mode; impulse_projection writes "
            "the older impulse+energy_projection inputs"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="manifest path to write; defaults to the solver-mode-specific manifest",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="run directory root to write; defaults to the solver-mode-specific runs root",
    )
    args = parser.parse_args()

    case_list = cases()
    manifest_path = args.manifest or default_manifest_path(args.solver_mode)
    runs_root = args.runs_root or default_runs_root(args.solver_mode)
    print_plan(case_list, args.solver_mode, manifest_path, runs_root)
    if args.write:
        setup_inputs(
            case_list,
            args.solver_mode,
            manifest_path,
            runs_root,
            portable_manifest=args.portable_manifest,
        )
        print()
        print(f"Wrote {len(case_list)} inputs and manifest:")
        print(f"  {manifest_path}")
        print(f"  Solver mode: {args.solver_mode}")
        if args.portable_manifest:
            print("  Manifest paths are relative to the repository root.")
    else:
        print("Dry run only. Re-run with --write after confirming the matrix.")


if __name__ == "__main__":
    main()
