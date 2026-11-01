@echo off
cd /d "%~dp0moteur\examples\soccer" 2>nul
if errorlevel 1 goto :pas_installe
if not exist "..\..\venv\Scripts\activate.bat" goto :pas_installe
call ..\..\venv\Scripts\activate.bat
python -m streamlit run app.py --server.maxUploadSize 4000 --server.fileWatcherType none --browser.gatherUsageStats false
pause
exit /b 0

:pas_installe
echo L'outil n'est pas encore installe. Lance d'abord installer.bat.
pause
exit /b 1
