@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Installation - Analyse de match

echo ============================================================
echo   Installation de l'outil d'analyse de match
echo   Prevoir environ 10 Go de telechargements et 10 a 30 minutes
echo ============================================================
echo.

rem ---- 1. Python 3.12 -------------------------------------------------
py -3.12 --version >nul 2>&1
if not errorlevel 1 goto :python_ok
echo Python 3.12 est introuvable sur ce PC.
winget --version >nul 2>&1
if errorlevel 1 goto :python_manuel
choice /c ON /m "Installer Python 3.12 avec winget maintenant (O = oui, N = non)"
if errorlevel 2 goto :python_manuel
winget install -e --id Python.Python.3.12
echo.
echo Python 3.12 est installe. FERME cette fenetre, puis relance installer.bat.
pause
exit /b 0

:python_manuel
echo Installe Python 3.12 depuis https://www.python.org/downloads/windows/
echo en cochant "Add python.exe to PATH", puis relance installer.bat.
pause
exit /b 1

:python_ok
echo [1/6] Python 3.12 trouve.

rem ---- 2. Moteur roboflow/sports ---------------------------------------
if exist "moteur\examples\soccer\main.py" goto :moteur_ok
echo [2/6] Telechargement du moteur roboflow/sports...
if exist sports-main rmdir /s /q sports-main
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri 'https://github.com/roboflow/sports/archive/refs/heads/main.zip' -OutFile 'sports.zip'"
if errorlevel 1 goto :erreur
powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Path 'sports.zip' -DestinationPath '.' -Force"
if errorlevel 1 goto :erreur
if exist moteur rmdir /s /q moteur
rename sports-main moteur
del sports.zip
:moteur_ok
echo [2/6] Moteur present.

rem ---- 3. Environnement Python -----------------------------------------
if exist "moteur\venv\Scripts\activate.bat" goto :venv_ok
echo [3/6] Creation de l'environnement Python...
py -3.12 -m venv moteur\venv
if errorlevel 1 goto :erreur
:venv_ok
call moteur\venv\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 goto :erreur

rem ---- 4. Bibliotheques ------------------------------------------------
echo [4/6] Installation des bibliotheques (long, patience)...
pip install -e moteur
if errorlevel 1 goto :erreur
pip install -r moteur\examples\soccer\requirements.txt
if errorlevel 1 goto :erreur
pip install -r requirements-extra.txt
if errorlevel 1 goto :erreur

rem ---- 5. Carte graphique ----------------------------------------------
nvidia-smi >nul 2>&1
if errorlevel 1 goto :sans_gpu
echo [5/6] Carte Nvidia detectee : installation de PyTorch avec CUDA...
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 goto :erreur
goto :gpu_fin
:sans_gpu
echo [5/6] Aucune carte Nvidia detectee : l'analyse se fera sur le processeur (lente).
:gpu_fin

rem ---- 6. Scripts, modeles et video d'exemple ----------------------------
echo [6/6] Copie des scripts et telechargement des modeles...
set "DEST=moteur\examples\soccer"
for %%F in (app.py radar_csv.py pitch_projection.py analyse_positions.py verifier_terrain.py affinage_lignes.py preparer_dataset.py annotateur_terrain.html GUIDE_REENTRAINEMENT.md) do copy /Y "%%F" "%DEST%\" >nul
if not exist "%DEST%\data" mkdir "%DEST%\data"
call :telecharger football-ball-detection.pt 1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V
call :telecharger football-player-detection.pt 17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q
call :telecharger football-pitch-detection.pt 1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf
call :telecharger 08fd33_0.mp4 1OG8K6wqUw9t7lp9ms1M48DxRhwTYciK-

rem Evite la question de Streamlit au premier lancement
if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
if exist "%USERPROFILE%\.streamlit\credentials.toml" goto :creds_ok
>"%USERPROFILE%\.streamlit\credentials.toml" echo [general]
>>"%USERPROFILE%\.streamlit\credentials.toml" echo email = ""
:creds_ok

echo.
if not exist "%DEST%\data\football-player-detection.pt" goto :modeles_manquants
if not exist "%DEST%\data\football-pitch-detection.pt" goto :modeles_manquants
echo ============================================================
echo   Installation terminee.
echo   Double-clique sur lancer_app.bat pour demarrer l'application.
echo   (Facultatif : installer_pnlcalib.bat pour une calibration plus precise.)
echo ============================================================
pause
exit /b 0

:modeles_manquants
echo ATTENTION : les modeles n'ont pas pu etre telecharges (limite de
echo Google Drive atteinte ?). Relance installer.bat plus tard : les
echo fichiers deja telecharges ne seront pas refaits.
pause
exit /b 1

:erreur
echo.
echo Une erreur est survenue. Regarde les dernieres lignes ci-dessus
echo (voir la section Depannage du README) puis relance installer.bat.
pause
exit /b 1

:telecharger
if exist "%DEST%\data\%~1" exit /b 0
gdown -O "%DEST%\data\%~1" "https://drive.google.com/uc?id=%~2"
exit /b 0
