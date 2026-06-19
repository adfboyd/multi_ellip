@echo off
setlocal

set ROOT=C:\Users\User\Documents\multi_ellip
set RUN_ROOT=two_body_parameter_sweep_task_runs
set MANIFEST=%RUN_ROOT%\manifest.csv
set SUMMARY=%RUN_ROOT%\run_summary.csv
set ANALYSIS=%RUN_ROOT%\analysis

cd /d "%ROOT%"

echo [1/6] Refreshing per-case dashboards and recurrence files for completed cases
"C:\Users\User\anaconda3\python.exe" ^
  "studies\two_body_parameter_sweep\run_two_body_parameter_sweep.py" ^
  --manifest "%MANIFEST%" ^
  --summary "%RUN_ROOT%\postprocess_summary.csv" ^
  --postprocess-only ^
  --rerun

echo [2/6] Writing aggregate conservation/solver summaries
"C:\Users\User\anaconda3\python.exe" ^
  "studies\two_body_parameter_sweep\analyze_two_body_sweep_results.py" ^
  --run-root "%RUN_ROOT%" ^
  --manifest "%MANIFEST%" ^
  --summary "%SUMMARY%" ^
  --out-dir "%ANALYSIS%"

echo [3/6] Building per-run dashboard gallery
"C:\Users\User\anaconda3\python.exe" ^
  "studies\two_body_parameter_sweep\build_dashboard_index.py" ^
  --run-root "%RUN_ROOT%" ^
  --summary "%SUMMARY%" ^
  --out "%ANALYSIS%\dashboard_index.html"

echo [4/6] Classifying dynamics on the second half of each trajectory
"C:\Users\User\anaconda3\python.exe" ^
  "studies\two_body_parameter_sweep\classify_two_body_dynamics.py" ^
  --manifest "%MANIFEST%" ^
  --out-dir "%ANALYSIS%" ^
  --suffix "_second_half" ^
  --transient-fraction 0.5

echo [5/6] Classifying dynamics on full trajectories
"C:\Users\User\anaconda3\python.exe" ^
  "studies\two_body_parameter_sweep\classify_two_body_dynamics.py" ^
  --manifest "%MANIFEST%" ^
  --out-dir "%ANALYSIS%" ^
  --suffix "_full_run" ^
  --transient-fraction 0

echo [6/6] Generating orbit animations for completed cases
"C:\Users\User\anaconda3\python.exe" ^
  "studies\two_body_parameter_sweep\animate_two_body_orbits.py" ^
  --run-root "%RUN_ROOT%" ^
  --summary "%SUMMARY%" ^
  --missing-only

echo Analysis complete: %ROOT%\%ANALYSIS%
