from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = ROOT / "two_body_parameter_sweep_task_runs"


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def rel(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def row_card(row: dict[str, str], out_dir: Path) -> str:
    dashboard = Path(row.get("dashboard", ""))
    if not dashboard.is_absolute():
        dashboard = ROOT / dashboard
    dashboard_href = rel(dashboard, out_dir)
    run_dir = dashboard.parent
    log_path = Path(row.get("log", ""))
    if not log_path.is_absolute():
        log_path = ROOT / log_path
    fields = [
        ("status", row.get("status", "")),
        ("shape", row.get("shape_name", "")),
        ("rho", row.get("rho", "")),
        ("E", row.get("energy_ratio", "")),
        ("sep", row.get("separation", "")),
        ("repeat", row.get("repeat", "")),
        ("max KE drift %", row.get("max_abs_ke_drift_pct", "")),
        ("max H drift %", row.get("max_hcon_drift_pct", "")),
        ("coupled residual", row.get("coupled_residual", "")),
        ("retries", row.get("hamiltonian_adaptive_retries", "")),
    ]
    meta = "\n".join(f"<li><b>{label}</b>: {value}</li>" for label, value in fields)
    return f"""
      <article class="card">
        <h2>{row.get("name", "")}</h2>
        <a href="{dashboard_href}"><img src="{dashboard_href}" alt="dashboard for {row.get("name", "")}"></a>
        <ul>{meta}</ul>
        <p class="links">
          <a href="{dashboard_href}">dashboard</a>
          <a href="{rel(log_path, out_dir)}">run log</a>
          <a href="{rel(run_dir / "recurrence_body1.png", out_dir)}">recurrence body 1</a>
          <a href="{rel(run_dir / "recurrence_body2.png", out_dir)}">recurrence body 2</a>
        </p>
      </article>
    """


def write_index(rows: list[dict[str, str]], out: Path, title: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    complete = [row for row in rows if row.get("dashboard")]
    cards = "\n".join(row_card(row, out.parent) for row in complete)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; background: #f6f7f8; color: #1f2328; }}
    header {{ margin-bottom: 20px; }}
    h1 {{ margin: 0 0 6px; font-size: 24px; }}
    .summary {{ color: #5b626b; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 18px; }}
    .card {{ background: white; border: 1px solid #d8dee4; border-radius: 8px; padding: 14px; }}
    .card h2 {{ font-size: 14px; margin: 0 0 10px; overflow-wrap: anywhere; }}
    img {{ width: 100%; height: auto; border: 1px solid #d8dee4; border-radius: 4px; }}
    ul {{ columns: 2; padding-left: 18px; font-size: 12px; line-height: 1.45; }}
    .links {{ display: flex; flex-wrap: wrap; gap: 10px; font-size: 13px; }}
    a {{ color: #0969da; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <div class="summary">{len(complete)} dashboards available from {len(rows)} summary rows.</div>
  </header>
  <main class="grid">
    {cards}
  </main>
</body>
</html>
"""
    out.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a browsable HTML dashboard index for two-body sweep runs.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    run_root = args.run_root if args.run_root.is_absolute() else ROOT / args.run_root
    summary = args.summary or run_root / "run_summary.csv"
    out = args.out or run_root / "analysis" / "dashboard_index.html"
    if not summary.is_absolute():
        summary = ROOT / summary
    if not out.is_absolute():
        out = ROOT / out
    rows = read_rows(summary)
    write_index(rows, out, "Two-body coupled endpoint dashboard index")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
