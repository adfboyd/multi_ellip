@echo off
setlocal

set ROOT=C:\Users\User\Documents\multi_ellip
set RUN_ROOT=two_body_parameter_sweep_task_runs
set TWO_BODY_LAUNCH_LOG=%RUN_ROOT%\launch.log
set SINGLE_BODY_LAUNCH_LOG=single_body_dynamics_check_after_two_body\launch.log
set PUBLISH_LOG=%RUN_ROOT%\publish_to_git.log

cd /d "%ROOT%"

echo [%DATE% %TIME%] Sweep output publisher started > "%PUBLISH_LOG%"
echo [%DATE% %TIME%] Waiting for two-body driver exit line >> "%PUBLISH_LOG%"

:WAIT_TWO_BODY
findstr /C:"Driver exited with errorlevel" "%TWO_BODY_LAUNCH_LOG%" >nul 2>nul
if errorlevel 1 (
  timeout /t 60 /nobreak >nul
  goto WAIT_TWO_BODY
)

echo [%DATE% %TIME%] Two-body driver has exited >> "%PUBLISH_LOG%"

if exist "%SINGLE_BODY_LAUNCH_LOG%" (
  echo [%DATE% %TIME%] Waiting for single-body follow-on to finish before post-processing >> "%PUBLISH_LOG%"
  :WAIT_SINGLE_BODY
  findstr /C:"Single-body dynamics check exited with errorlevel" "%SINGLE_BODY_LAUNCH_LOG%" >nul 2>nul
  if errorlevel 1 (
    timeout /t 60 /nobreak >nul
    goto WAIT_SINGLE_BODY
  )
)

echo [%DATE% %TIME%] Running post-run analysis, dashboards, classifications, and animations >> "%PUBLISH_LOG%"
call "studies\two_body_parameter_sweep\analyze_task_sweep_after_completion.bat" >> "%PUBLISH_LOG%" 2>&1
if errorlevel 1 (
  echo [%DATE% %TIME%] Analysis failed with errorlevel %ERRORLEVEL% >> "%PUBLISH_LOG%"
  exit /b %ERRORLEVEL%
)

echo [%DATE% %TIME%] Staging PDF/source, logs, summaries, and visualisations only >> "%PUBLISH_LOG%"
git add docs\solver_equations.tex docs\solver_equations.pdf >> "%PUBLISH_LOG%" 2>&1
for %%F in ("%RUN_ROOT%\*.log" "%RUN_ROOT%\*.csv" "%RUN_ROOT%\*.html") do (
  if exist "%%~F" git add -f "%%~F" >> "%PUBLISH_LOG%" 2>&1
)
if exist "%RUN_ROOT%\analysis" git add -f "%RUN_ROOT%\analysis" >> "%PUBLISH_LOG%" 2>&1
for /r "%RUN_ROOT%" %%F in (run.log dashboard.png recurrence_body*.png recurrence_metrics.csv orbit_animation.gif orbit_animation.mp4 *.html) do (
  if exist "%%~F" git add -f "%%~F" >> "%PUBLISH_LOG%" 2>&1
)
if errorlevel 1 (
  echo [%DATE% %TIME%] git add failed with errorlevel %ERRORLEVEL% >> "%PUBLISH_LOG%"
  exit /b %ERRORLEVEL%
)

echo [%DATE% %TIME%] Creating results commit >> "%PUBLISH_LOG%"
git commit -m "Add two-body sweep outputs and solver equations PDF" >> "%PUBLISH_LOG%" 2>&1
set COMMIT_RC=%ERRORLEVEL%
if not "%COMMIT_RC%"=="0" (
  echo [%DATE% %TIME%] git commit returned %COMMIT_RC%; continuing to push in case there was nothing new >> "%PUBLISH_LOG%"
)

echo [%DATE% %TIME%] Pushing branch >> "%PUBLISH_LOG%"
git push >> "%PUBLISH_LOG%" 2>&1
set PUSH_RC=%ERRORLEVEL%
echo [%DATE% %TIME%] Publisher exited with errorlevel %PUSH_RC% >> "%PUBLISH_LOG%"
exit /b %PUSH_RC%
