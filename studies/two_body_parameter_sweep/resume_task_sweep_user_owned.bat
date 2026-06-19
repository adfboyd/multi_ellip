@echo off
setlocal

set ROOT=C:\Users\User\Documents\multi_ellip
set PYTHON=C:\Users\User\anaconda3\python.exe
set RUN_ROOT=two_body_parameter_sweep_task_runs
set MANIFEST=%RUN_ROOT%\manifest.csv
set SUMMARY=%RUN_ROOT%\run_summary.csv
set STDOUT=%RUN_ROOT%\driver_stdout.log
set STDERR=%RUN_ROOT%\driver_stderr.log
set LAUNCH_LOG=%RUN_ROOT%\launch.log

cd /d "%ROOT%"

echo [%DATE% %TIME%] Resume driver started >> "%LAUNCH_LOG%"

"%PYTHON%" ^
  "studies\two_body_parameter_sweep\run_two_body_parameter_sweep.py" ^
  --manifest "%MANIFEST%" ^
  --summary "%SUMMARY%" ^
  >> "%STDOUT%" 2>> "%STDERR%"

set RC=%ERRORLEVEL%
echo [%DATE% %TIME%] Driver exited with errorlevel %RC% >> "%LAUNCH_LOG%"
exit /b %RC%
