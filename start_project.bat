@echo off
setlocal

set "ROOT=%~dp0"
set "OLLAMA_EXE=%USERPROFILE%\AppData\Local\Programs\Ollama\ollama.exe"
set "OLLAMA_HOST=127.0.0.1:11435"
set "MAS_OLLAMA_BASE_URL=http://127.0.0.1:11435"
if defined MAS_OLLAMA_MODELS_DIR set "OLLAMA_MODELS=%MAS_OLLAMA_MODELS_DIR%"
set "BACKEND_URL=http://127.0.0.1:8000"
set "FRONTEND_URL=http://localhost:5173"
set "PYTHON_EXE=%ROOT%backend\.venv\Scripts\python.exe"

title MAS Analytics - One-Tap Launcher
echo ==========================================
echo   MAS Analytics - One-Tap Launcher
echo ==========================================
echo.

if not exist "%PYTHON_EXE%" (
  echo [ERROR] Project Python environment is missing:
  echo %PYTHON_EXE%
  echo.
  echo Create it and install backend\requirements.txt before launching.
  pause
  exit /b 1
)
if not exist "%ROOT%frontend\node_modules" (
  echo [ERROR] frontend\node_modules is missing. Run npm install in frontend first.
  pause
  exit /b 1
)
where.exe npm.cmd >nul 2>&1
if errorlevel 1 (
  echo [ERROR] npm.cmd was not found. Install Node.js, then reopen this launcher.
  pause
  exit /b 1
)

echo [1/3] Checking Ollama...
call :wait_for_url "http://127.0.0.1:11435/api/tags" 2
if not errorlevel 1 goto backend
if not exist "%OLLAMA_EXE%" (
  echo [ERROR] Ollama is not responding and its executable was not found:
  echo %OLLAMA_EXE%
  pause
  exit /b 1
)
if defined MAS_OLLAMA_MODELS_DIR if not exist "%MAS_OLLAMA_MODELS_DIR%" (
  echo [ERROR] Configured Ollama model directory was not found:
  echo %MAS_OLLAMA_MODELS_DIR%
  pause
  exit /b 1
)
echo Starting Ollama at %OLLAMA_HOST%...
start "MAS Ollama" /D "%ROOT%" "%ComSpec%" /k ""%OLLAMA_EXE%" serve"
call :wait_for_url "http://127.0.0.1:11435/api/tags" 45
if errorlevel 1 (
  echo [ERROR] Ollama did not become ready on port 11435.
  echo Check the MAS Ollama terminal window for its startup error.
  pause
  exit /b 1
)

:backend
echo [2/3] Checking FastAPI backend...
call :wait_for_url "%BACKEND_URL%/health" 2
if not errorlevel 1 goto frontend
echo Starting FastAPI backend...
start "MAS Backend" cmd /k "cd /d ""%ROOT%backend"" ^&^& ""%PYTHON_EXE%"" -m uvicorn app.main:app --host 127.0.0.1 --port 8000"
call :wait_for_url "%BACKEND_URL%/health" 45
if errorlevel 1 (
  echo [ERROR] FastAPI did not become ready on port 8000.
  echo Check the MAS Backend terminal window for its startup error.
  pause
  exit /b 1
)

:frontend
echo [3/3] Checking React frontend...
call :wait_for_url "%FRONTEND_URL%/" 2
if not errorlevel 1 goto ready
echo Starting React frontend...
start "MAS Frontend" cmd /k "cd /d ""%ROOT%frontend"" ^&^& call npm.cmd run dev -- --host 0.0.0.0 --port 5173 --strictPort"
call :wait_for_url "%FRONTEND_URL%/" 45
if errorlevel 1 (
  echo [ERROR] Vite did not become ready on port 5173.
  echo Check the MAS Frontend terminal window for its startup error.
  pause
  exit /b 1
)

:ready
echo.
echo All project services are ready.
echo Frontend: %FRONTEND_URL%
echo Backend : %BACKEND_URL%
echo Ollama  : http://%OLLAMA_HOST%
echo Opening the dashboard...
if /I "%MAS_OPEN_BROWSER%"=="0" goto launcher_done
start "" "%FRONTEND_URL%/"

:launcher_done
echo.
pause
exit /b 0

:wait_for_url
set "CHECK_URL=%~1"
set "WAIT_SECONDS=%~2"
for /L %%I in (1,1,%WAIT_SECONDS%) do (
  curl.exe --silent --show-error --fail --max-time 2 "%CHECK_URL%" >nul 2>&1
  if not errorlevel 1 exit /b 0
  timeout /t 1 /nobreak >nul
)
exit /b 1
