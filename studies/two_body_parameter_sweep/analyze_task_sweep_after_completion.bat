@echo off
setlocal

set ROOT=C:\Users\User\Documents\multi_ellip
set RUN_ROOT=two_body_parameter_sweep_task_runs
set MANIFEST=%RUN_ROOT%\manifest.csv
set SUMMARY=%RUN_ROOT%\run_summary.csv
set ANALYSIS=%RUN_ROOT%\analysis

cd /d "%ROOT%"

echo [1/4] Refreshing per-case dashboards and recurrence files for completed cases
"C:\Users\User\anaconda3\python.exe" ^
  "studies\two_body_parameter_sweep\run_two_body_parameter_sweep.py" ^
  --manifest "%MANIFEST%" ^
  --summary "%RUN_ROOT%\postprocess_summary.csv" ^
  --postprocess-only ^
  --rerun

echo [2/4] Writing aggregate conservation/solver summaries
"C:\Users\User\anaconda3\python.exe" ^
  "studies\two_body_parameter_sweep\analyze_two_body_sweep_results.py" ^
  --run-root "%RUN_ROOT%" ^
  --manifest "%MANIFEST%" ^
  --summary "%SUMMARY%" ^
  --out-dir "%ANALYSIS%"

echo [3/4] Classifying dynamics on the second half of each trajectory
"C:\Users\User\anaconda3\python.exe" ^
  "studies\two_body_parameter_sweep\classify_two_body_dynamics.py" ^
  --manifest "%MANIFEST%" ^
  --out-dir "%ANALYSIS%" ^
  --suffix "_second_half" ^
  --transient-fraction 0.5

echo [4/4] Classifying dynamics on full trajectories
"C:\Users\User\anaconda3\python.exe" ^
  "studies\two_body_parameter_sweep\classify_two_body_dynamics.py" ^
  --manifest "%MANIFEST%" ^
  --out-dir "%ANALYSIS%" ^
  --suffix "_full_run" ^
  --transient-fraction 0

echo Analysis complete: %ROOT%\%ANALYSIS%
