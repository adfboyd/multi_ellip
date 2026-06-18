@echo off
setlocal

set ROOT=C:\Users\User\Documents\multi_ellip
set PYTHON=C:\Users\User\anaconda3\python.exe
set TWO_BODY_ROOT=two_body_parameter_sweep_task_runs
set TWO_BODY_LAUNCH_LOG=%TWO_BODY_ROOT%\launch.log
set OUT_ROOT=single_body_dynamics_check_after_two_body
set STDOUT=%OUT_ROOT%\driver_stdout.log
set STDERR=%OUT_ROOT%\driver_stderr.log
set LAUNCH_LOG=%OUT_ROOT%\launch.log

cd /d "%ROOT%"
mkdir "%OUT_ROOT%" 2>nul

echo [%DATE% %TIME%] Single-body follow-on launcher started > "%LAUNCH_LOG%"
echo [%DATE% %TIME%] Waiting for %TWO_BODY_LAUNCH_LOG% to report driver exit >> "%LAUNCH_LOG%"

:WAIT_TWO_BODY
findstr /C:"Driver exited with errorlevel" "%TWO_BODY_LAUNCH_LOG%" >nul 2>nul
if errorlevel 1 (
  timeout /t 60 /nobreak >nul
  goto WAIT_TWO_BODY
)

echo [%DATE% %TIME%] Two-body launcher has exited; building release binary >> "%LAUNCH_LOG%"
cargo build --release >> "%LAUNCH_LOG%" 2>&1
if errorlevel 1 (
  echo [%DATE% %TIME%] cargo build failed with errorlevel %ERRORLEVEL% >> "%LAUNCH_LOG%"
  exit /b %ERRORLEVEL%
)

echo [%DATE% %TIME%] Running single-body dynamics check >> "%LAUNCH_LOG%"
"%PYTHON%" ^
  "studies\ke_ratio_density\run_single_body_dynamics_check.py" ^
  --source-root "studies\ke_ratio_density\runs" ^
  --out-root "%OUT_ROOT%" ^
  --rerun ^
  > "%STDOUT%" 2> "%STDERR%"

set RC=%ERRORLEVEL%
echo [%DATE% %TIME%] Single-body dynamics check exited with errorlevel %RC% >> "%LAUNCH_LOG%"
exit /b %RC%
