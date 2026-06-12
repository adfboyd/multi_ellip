"""Compare two multi_ellip output files numerically."""
import sys
import csv
import math

def load(path):
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows

def vec_norm(row, prefix):
    return math.sqrt(sum(row[f"{prefix}_{i}"]**2 for i in range(1,4)))

def compare(old_path, new_path):
    old = load(old_path)
    new = load(new_path)
    n = min(len(old), len(new))
    print(f"Comparing {n} timesteps  ({old_path}  vs  {new_path})\n")

    cols_to_diff = [
        ("body1 pos x",   "p1_1"),
        ("body1 pos y",   "p2_1"),
        ("body2 pos x",   "p1_2"),
        ("body1 vel x",   "v1_1"),
        ("body1 vel y",   "v2_1"),
        ("body2 vel x",   "v1_2"),
        ("ke_fluid",      "ke_fluid"),
        ("ke_solid",      "ke_solid"),
        ("ke_b1",         "ke_b1"),
        ("ke_b2",         "ke_b2"),
    ]

    # Print per-column max absolute difference
    print(f"{'Quantity':<18}  {'old_final':>14}  {'new_final':>14}  {'max|diff|':>14}  {'rel max diff':>14}")
    print("-" * 80)
    for label, col in cols_to_diff:
        if col not in old[0]:
            print(f"  {label:<18}  (not in file)")
            continue
        diffs = [abs(new[i][col] - old[i][col]) for i in range(n)]
        max_diff = max(diffs)
        old_final = old[n-1][col]
        new_final = new[n-1][col]
        denom = max(abs(old_final), abs(new_final), 1e-10)
        rel = max_diff / denom
        print(f"  {label:<18}  {old_final:>14.6g}  {new_final:>14.6g}  {max_diff:>14.3e}  {rel:>12.3e}")

    # Energy sanity checks at final time
    print("\n--- Energy sanity (final timestep, new run) ---")
    r = new[n-1]
    ke_solid = r.get("ke_solid", float('nan'))
    ke_fluid = r.get("ke_fluid", float('nan'))
    ke_total = ke_solid + ke_fluid
    print(f"  KE solid = {ke_solid:.6g}")
    print(f"  KE fluid = {ke_fluid:.6g}")
    print(f"  KE total = {ke_total:.6g}")
    ratio = ke_fluid / ke_solid if ke_solid != 0 else float('inf')
    print(f"  KE_fluid / KE_solid = {ratio:.4f}  (expect O(1) for comparable-density bodies)")

    # Time-series of body2 speed (should be non-trivial due to hydrodynamic interaction)
    print("\n--- Body 2 speed (10 evenly spaced timesteps, new run) ---")
    for i in range(0, n, max(1, n//10)):
        r = new[i]
        v2 = math.sqrt(r["v1_2"]**2 + r["v2_2"]**2 + r["v3_2"]**2)
        print(f"  t={r['time']:6.3f}  |v2|={v2:.6g}  pos2_x={r['p1_2']:.6g}")

if __name__ == "__main__":
    old = sys.argv[1] if len(sys.argv) > 1 else "output1/multiple_body_complete.dat"
    new = sys.argv[2] if len(sys.argv) > 2 else "output_fix3/multiple_body_complete.dat"
    compare(old, new)
