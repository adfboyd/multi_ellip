from __future__ import annotations

import argparse
import csv
from pathlib import Path

from run_two_body_parameter_sweep import (
    case_metrics,
    has_finite_output,
    load_output,
    log_metrics,
    output_complete,
    save_dashboard,
    save_recurrence,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "studies" / "two_body_parameter_sweep" / "two_body_parameter_sweep_manifest.csv"
DEFAULT_OUT_DIR = ROOT / "old_two_body_sweep_analysis"


def read_rows(path: Path) -> list[dict[str, str]]:
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


def mirror_row(row: dict[str, str], out_dir: Path) -> dict[str, str]:
    if not output_complete(row):
        return {**row, "status": "incomplete", "postprocess_message": "missing or incomplete output"}
    data = load_output(row)
    if len(data) < 2 or not has_finite_output(data):
        return {**row, "status": "nonfinite", "postprocess_message": "non-finite or too-short output"}

    mirror_dir = out_dir / "dashboards" / row["name"]
    mirror_dir.mkdir(parents=True, exist_ok=True)
    mirror_row_data = {
        **row,
        "output": str(mirror_dir / "multiple_body_complete.dat"),
    }
    dashboard = save_dashboard(mirror_row_data, data)
    rec_rows = []
    rec_message = "ok"
    for body in (1, 2):
        try:
            rec_rows.append(save_recurrence(mirror_row_data, data, body))
        except ValueError as exc:
            rec_message = f"recurrence skipped: {exc}"
    rec_csv = mirror_dir / "recurrence_metrics.csv"
    if rec_rows:
        write_rows(rec_csv, rec_rows)
    rec_by_body = {rec["body"]: rec for rec in rec_rows}
    post = {
        "dashboard": str(dashboard),
        "recurrence_metrics": str(rec_csv) if rec_rows else "",
        "recurrence_rr_body1": rec_by_body.get("1", {}).get("achieved_rr", ""),
        "recurrence_rr_body2": rec_by_body.get("2", {}).get("achieved_rr", ""),
        "recurrence_eps_body1": rec_by_body.get("1", {}).get("epsilon", ""),
        "recurrence_eps_body2": rec_by_body.get("2", {}).get("epsilon", ""),
        **case_metrics(data),
        **(log_metrics(Path(row["log"])) if row.get("log") else {}),
        "postprocess_message": rec_message,
    }
    return {**row, "status": "OK", **post}


def main() -> None:
    parser = argparse.ArgumentParser(description="Mirror completed run dashboards into a writable analysis directory.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    manifest = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    rows = read_rows(manifest)
    if args.limit is not None:
        rows = rows[: args.limit]

    mirrored = []
    for i, row in enumerate(rows, start=1):
        print(f"[{i:03d}/{len(rows):03d}] {row['name']}", flush=True)
        mirrored.append(mirror_row(row, out_dir))

    summary = out_dir / "dashboard_summary.csv"
    write_rows(summary, mirrored)
    print(f"Wrote {summary}")


if __name__ == "__main__":
    main()
