@echo off
setlocal EnableExtensions
if "%~1"=="" goto :usage
if "%~2"=="" goto :usage
set "VIDEO=%~f1"
set "SORTIE=%~f2"
set "PAS=%~3"
if "%PAS%"=="" set "PAS=1"

cd /d "%~dp0pnlcalib\PnLCalib" 2>nul
if errorlevel 1 goto :pas_installe
if not exist "..\venv\Scripts\activate.bat" goto :pas_installe
if not exist "weights\SV_kp" goto :pas_poids
if not exist "weights\SV_lines" goto :pas_poids

call ..\venv\Scripts\activate.bat
python pnl_export.py --weights_kp weights\SV_kp --weights_line weights\SV_lines --input_path "%VIDEO%" --output_csv "%SORTIE%" --pnl_refine --stride %PAS%
pause
exit /b 0

:usage
echo Utilisation : calibrer_video.bat chemin_video.mp4 chemin_sortie.csv [pas]
echo   pas : 1 = toutes les images (precis, lent) ; 5 = une image sur 5 (environ 5 fois plus rapide)
pause
exit /b 1

:pas_installe
echo PnLCalib n'est pas installe : lance d'abord installer_pnlcalib.bat.
pause
exit /b 1

:pas_poids
echo Poids introuvables : place SV_kp et SV_lines dans outils\pnlcalib\PnLCalib\weights (voir GUIDE_PNLCALIB.md).
pause
exit /b 1
