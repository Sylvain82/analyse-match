@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Installation de PnLCalib (facultatif)

echo ============================================================
echo   Installation de PnLCalib : calibration plus precise du terrain
echo   Environnement SEPARE : il ne modifie pas l'application.
echo   Prevoir environ 5 Go de telechargements.
echo ============================================================
echo.

py -3.12 --version >nul 2>&1
if errorlevel 1 goto :pas_python
git --version >nul 2>&1
if errorlevel 1 goto :pas_git

if not exist pnlcalib mkdir pnlcalib
cd pnlcalib

if exist PnLCalib\inference.py goto :deja_clone
echo [1/4] Telechargement du depot PnLCalib...
git clone https://github.com/mguti97/PnLCalib.git
if errorlevel 1 goto :erreur
:deja_clone

if exist venv\Scripts\activate.bat goto :venv_ok
echo [2/4] Creation de l'environnement Python...
py -3.12 -m venv venv
if errorlevel 1 goto :erreur
:venv_ok
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 goto :erreur

echo [3/4] Installation des bibliotheques (long, patience)...
nvidia-smi >nul 2>&1
if errorlevel 1 goto :torch_cpu
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 goto :erreur
goto :torch_fin
:torch_cpu
echo Aucune carte Nvidia detectee : PyTorch sans carte graphique (tres lent).
pip install torch torchvision
if errorlevel 1 goto :erreur
:torch_fin
pip install opencv-python pyyaml scipy matplotlib tqdm yacs pillow scikit-learn
if errorlevel 1 goto :erreur

echo [4/4] Copie du script d'export...
copy /Y "..\pnl_export.py" "PnLCalib\" >nul
if not exist PnLCalib\weights mkdir PnLCalib\weights

echo.
echo ============================================================
echo   Installation terminee. Il reste a TELECHARGER LES POIDS :
echo   1. Ouvre PnLCalib\README.md et repere la section Weights.
echo   2. Telecharge les deux fichiers SV_kp et SV_lines.
echo   3. Place-les dans le dossier : %CD%\PnLCalib\weights
echo   Puis utilise calibrer_video.bat (voir GUIDE_PNLCALIB.md).
echo ============================================================
pause
exit /b 0

:pas_python
echo Python 3.12 est introuvable. Installe-le depuis python.org (coche "Add python.exe to PATH"), puis relance.
pause
exit /b 1

:pas_git
echo Git est introuvable. Installe-le depuis git-scm.com, puis relance ce script.
echo (Sinon : telecharge PnLCalib en ZIP depuis GitHub et decompresse-le dans outils\pnlcalib\PnLCalib.)
pause
exit /b 1

:erreur
echo.
echo Une erreur est survenue : regarde les dernieres lignes ci-dessus (voir GUIDE_PNLCALIB.md, section Depannage).
pause
exit /b 1
