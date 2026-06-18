from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = ROOT / "two_body_parameter_sweep_user_runs"
DEFAULT_MANIFEST = DEFAULT_RUN_ROOT / "manifest.csv"
DEFAULT_SUMMARY = DEFAULT_RUN_ROOT / "run_summary.csv"
DEFAULT_STDOUT = DEFAULT_RUN_ROOT / "driver_stdout.log"
DEFAULT_STDERR = DEFAULT_RUN_ROOT / "driver_stderr.log"
DRIVER = ROOT / "studies" / "two_body_parameter_sweep" / "run_two_body_parameter_sweep.py"


def clean_env() -> dict[str, str]:
    env: dict[str, str] = {}
    path_value = ""
    for key, value in os.environ.items():
        if key.lower() == "path":
            if not path_value:
                path_value = value
        else:
            env[key] = value
    if path_value:
        env["Path"] = path_value
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the two-body sweep driver in the background.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--stdout", type=Path, default=DEFAULT_STDOUT)
    parser.add_argument("--stderr", type=Path, default=DEFAULT_STDERR)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--wait", action="store_true", help="wait for the driver and print its return code")
    args = parser.parse_args()

    args.stdout.parent.mkdir(parents=True, exist_ok=True)
    args.stderr.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(DRIVER),
        "--manifest",
        str(args.manifest),
        "--summary",
        str(args.summary),
    ]
    if args.rerun:
        cmd.append("--rerun")
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])

    stdout = args.stdout.open("wb")
    stderr = args.stderr.open("wb")
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    process = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=stdout,
        stderr=stderr,
        stdin=subprocess.DEVNULL,
        env=clean_env(),
        creationflags=creationflags,
        close_fds=False,
    )
    print(process.pid)
    if args.wait:
        print(process.wait())


if __name__ == "__main__":
    main()
