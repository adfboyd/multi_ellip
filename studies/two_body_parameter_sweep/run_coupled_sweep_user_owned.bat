@echo off
setlocal

set ROOT=C:\Users\User\Documents\multi_ellip
set RUN_ROOT=two_body_parameter_sweep_task_runs
set MANIFEST=%RUN_ROOT%\manifest.csv
set SUMMARY=%RUN_ROOT%\run_summary.csv
set STDOUT=%RUN_ROOT%\driver_stdout.log
set STDERR=%RUN_ROOT%\driver_stderr.log
set LAUNCH_LOG=%RUN_ROOT%\launch.log

cd /d "%ROOT%"
mkdir "%RUN_ROOT%" 2>nul

echo [%DATE% %TIME%] Launch started > "%LAUNCH_LOG%"

"C:\Users\User\anaconda3\python.exe" ^
  "studies\two_body_parameter_sweep\setup_two_body_parameter_sweep.py" ^
  --write ^
  --portable-manifest ^
  --solver-mode coupled_endpoint ^
  --runs-root "%RUN_ROOT%" ^
  --manifest "%MANIFEST%" ^
  >> "%LAUNCH_LOG%" 2>&1

if errorlevel 1 (
  echo [%DATE% %TIME%] Setup failed with errorlevel %ERRORLEVEL% >> "%LAUNCH_LOG%"
  exit /b %ERRORLEVEL%
)

echo [%DATE% %TIME%] Driver started >> "%LAUNCH_LOG%"

"C:\Users\User\anaconda3\python.exe" ^
  "studies\two_body_parameter_sweep\run_two_body_parameter_sweep.py" ^
  --manifest "%MANIFEST%" ^
  --summary "%SUMMARY%" ^
  --rerun ^
  > "%STDOUT%" 2> "%STDERR%"

set RC=%ERRORLEVEL%
echo [%DATE% %TIME%] Driver exited with errorlevel %RC% >> "%LAUNCH_LOG%"
exit /b %RC%
