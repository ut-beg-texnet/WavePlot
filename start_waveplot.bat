@echo off
rem Windows launcher: finds Python and runs start_waveplot.py with the given options.
setlocal
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (where python >nul 2>nul && set "PY=python")
if not defined PY (
    echo Python was not found. Install Python 3.13 from https://www.python.org/downloads/
    echo ^(tick "Add python.exe to PATH"^) or run:  winget install Python.Python.3.13
    pause
    exit /b 1
)
%PY% "%~dp0start_waveplot.py" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" pause
exit /b %RC%
