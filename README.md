# Analyse de match de football

Application locale pour analyser une vidéo de match : détection des joueurs, position sur le terrain, nettoyage des trajectoires, puis statistiques (largeur et profondeur des équipes, distances, vitesses, cartes de chaleur) avec export Excel.

**Tout se calcule sur ton ordinateur.** Aucune vidéo et aucune donnée n'est envoyée sur internet.

Cet outil s'appuie sur [roboflow/sports](https://github.com/roboflow/sports) (détection, suivi, classification des équipes) et ajoute : projection fiabilisée sur le terrain, nettoyage et statistiques, interface Streamlit, page web d'analyse de CSV, calibration PnLCalib optionnelle et outils de réentraînement.

## Comment ça marche

1. L'**application** (Windows, navigateur) détecte les joueurs sur la vidéo et les place sur le terrain : fichier `positions.csv`.
2. Elle crée l'**Excel des données brutes**, puis nettoie les trajectoires et calcule les statistiques.
3. Elle **affiche les données nettoyées** : tableaux filtrables, radar image par image, cartes de chaleur, statistiques par équipe et par joueur.

La **page web** `index.html` (GitHub Pages) est une version allégée : elle ne sait pas analyser de vidéo, seulement traiter un `positions.csv` déjà produit, directement dans le navigateur.

## Ce qu'il faut

- Windows 10 ou 11
- Environ 10 Go d'espace disque libre
- Une **carte graphique Nvidia** : très fortement conseillée, car sans elle l'analyse d'une vidéo est très lente
- Une connexion internet pour l'installation (pas pour l'utilisation)

## Installation

1. Télécharge ce dépôt : bouton vert **Code**, puis **Download ZIP**, et décompresse-le.
2. Ouvre le dossier `outils` et double-clique sur **`installer.bat`**.
3. Attends la fin (10 à 30 minutes). Le script installe Python 3.12 si besoin (avec winget), télécharge le moteur roboflow/sports, crée un environnement isolé, installe les bibliothèques (PyTorch avec CUDA si une carte Nvidia est détectée), télécharge les modèles d'IA et une vidéo d'exemple, puis copie les scripts.

Tout est installé dans `outils\moteur`, sans toucher au reste du système.

## Utilisation

1. Double-clique sur **`outils\lancer_app.bat`** : l'application s'ouvre dans ton navigateur.
2. Choisis « Une vidéo de match » et dépose ta vidéo. Pour un premier essai, prends la vidéo d'exemple téléchargée par l'installeur (voir « Vidéos d'exemple » plus bas), et règle « Analyser seulement les N premières secondes » sur 30.
3. Clique sur « Lancer l'analyse complète » et ne touche à rien pendant le calcul.
4. Les résultats s'affichent dans des onglets : données nettoyées, équipes, joueurs, radar image par image, cartes, données brutes, méthode et limites, téléchargements.
5. Télécharge les Excel (`donnees_brutes.xlsx` et `analyse.xlsx`), les CSV ou la vidéo avec le radar. Tout est aussi enregistré dans le dossier `analyses`, à côté des scripts.

Tu peux aussi déposer directement un `positions.csv` déjà produit : l'application ne refait alors que les statistiques.

## Bien choisir ses vidéos

- Utilise de préférence une **vidéo de diffusion TV**, en plan large, avec le terrain bien visible.
- Commence par un **extrait court** (10 à 30 secondes), sans gros plan ni rediffusion.
- Résolution de 720p ou plus, au format MP4.

## Vidéos d'exemple

**Aucune vidéo n'est incluse dans ce dépôt**, volontairement : les extraits de matchs appartiennent à leurs ayants droit, les fichiers vidéo sont lourds pour GitHub, et le dépôt est fait pour ne contenir aucune donnée.

- **`installer.bat` télécharge pour toi un extrait de 30 s** (`08fd33_0.mp4`, fourni par Roboflow) dans `outils\moteur\examples\soccer\data`. C'est un bon premier test : dépose-le dans l'application.
- Pour récupérer les quatre autres extraits du projet roboflow/sports, lance dans `outils\moteur\examples\soccer`, avec l'environnement actif (`..\..\venv\Scripts\activate`) :
  ```
  gdown -O data\0bfacc_0.mp4 "https://drive.google.com/uc?id=12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF"
  gdown -O data\2e57b9_0.mp4 "https://drive.google.com/uc?id=19PGw55V8aA6GZu5-Aac5_9mCy3fNxmEf"
  gdown -O data\573e61_0.mp4 "https://drive.google.com/uc?id=1yYPKuXbHsCxqjA9G-S6aeR2Kcnos8RPU"
  gdown -O data\121364_0.mp4 "https://drive.google.com/uc?id=1vVwjW1dE1drIdd4ZSILfbCGPD4weoNiu"
  ```
  Google Drive peut limiter temporairement un fichier (« Too many users… ») : réessaie plus tard.
- Ces extraits viennent de la compétition Kaggle « DFL - Bundesliga Data Shootout », où l'on trouve aussi des matchs entiers (compte Kaggle gratuit et règlement à lire). Vérifie les conditions d'utilisation avant tout usage au-delà de tes tests.
- **Sans aucune vidéo** : sur la page web, le bouton « Essayer avec des données d'exemple (simulées) » génère un extrait fictif directement dans le navigateur, avec du bruit et des erreurs ajoutés pour montrer le nettoyage. Il n'a aucun lien avec un vrai match.

## Qualité des positions : ce qu'il faut savoir

- Les **statistiques d'équipe** (largeur, profondeur, compacité) sont les plus fiables.
- Les **distances et vitesses** sont des estimations. La vitesse maximale est la mesure la moins fiable.
- La projection sur le terrain peut se tromper quand la caméra zoome ou change de plan, surtout **sur le bord proche de la caméra**, où il y a très peu de repères.
- Le gabarit de terrain des points du terrain de Roboflow est de 120 × 70 m, alors que les terrains standard font 105 × 68 m : les distances le long du terrain peuvent être légèrement surestimées.

Pour améliorer la justesse, trois outils sont fournis (tous dans `outils`) :

| Outil | À quoi il sert |
|---|---|
| `verifier_terrain.py` | Superpose à la vidéo les lignes du terrain reconstituées, et compare plusieurs réglages (`--compare`). À utiliser pour juger la projection |
| **PnLCalib** (voir `GUIDE_PNLCALIB.md`) | Modèle de calibration plus précis sur le bord proche de la caméra, utilisable en option. Terrain de 105 × 68 m |
| Réentraînement (voir `GUIDE_REENTRAINEMENT.md`) | Affiner le modèle de points du terrain sur tes propres vidéos. Long (annotation), avec un gain incertain |

`affinage_lignes.py` (recalage sur les lignes blanches) est **expérimental** et s'est avéré peu fiable sur des vidéos réelles (il peut s'accrocher aux panneaux publicitaires ou aux lignes pointillées). Ne l'active pas sans vérification.

## Page web (GitHub Pages)

Pour mettre `index.html` en ligne : dans le dépôt GitHub, « Settings » → « Pages » → « Deploy from a branch », branche `main`, dossier `/ (root)`. La page est ensuite disponible à `https://<compte>.github.io/<depot>/`.

Elle lit un `positions.csv` ou un `positions_nettoyees.csv`, et adapte automatiquement le terrain aux dimensions indiquées dans le fichier (colonnes `pitch_l_cm` et `pitch_w_cm`). Seule la bibliothèque d'export Excel (SheetJS) est chargée depuis internet : c'est du code, jamais tes données.

## Confidentialité

- Les calculs de la vidéo se font localement.
- Le fichier `.gitignore` empêche d'envoyer par erreur des CSV, vidéos, images, modèles ou classeurs Excel sur GitHub. **Ne publie jamais de vidéo ni de `positions.csv` dans ce dépôt.**
- Pour des matchs de club ou amateurs, vérifie l'accord des personnes filmées avant de partager des résultats.

## Dépannage

| Problème | Solution |
|---|---|
| `installer.bat` ne trouve pas Python 3.12 | Installe-le depuis python.org en cochant « Add python.exe to PATH », ferme la fenêtre et relance |
| Erreur « Microsoft Visual C++ 14.0 or greater is required » | Mauvaise version de Python : l'installeur utilise la 3.12, qui évite ce problème. Supprime le dossier `outils\moteur\venv` et relance |
| Le téléchargement d'un modèle échoue (« Too many users… ») | Google Drive limite temporairement le fichier : relance `installer.bat` plus tard (les fichiers déjà téléchargés ne sont pas refaits) |
| `module 'supervision' has no attribute 'ByteTrack'` | Dans l'environnement : `pip install "supervision==0.25.1"` |
| `No module named ...` | L'environnement virtuel n'est pas activé : utilise `lancer_app.bat`, ou active-le avec `outils\moteur\venv\Scripts\activate` |
| `outils\moteur\venv` introuvable ou « Permission denied » | Ferme tous les terminaux et l'application, supprime `outils\moteur\venv` (redémarre le PC si besoin) et relance `installer.bat` |
| L'analyse est très lente | Sans carte Nvidia, c'est normal. Teste sur des extraits courts |
| L'analyse n'utilise pas la carte graphique | Dans l'environnement : `python -c "import torch; print(torch.cuda.is_available())"` doit afficher `True`. Sinon, réinstalle PyTorch avec CUDA : `pip uninstall -y torch torchvision` puis `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126` |
| La page web affiche « Colonnes manquantes » | Le fichier n'est pas un `positions.csv` produit par l'outil |

## Contenu du dépôt

```
index.html                  page web d'analyse de CSV (GitHub Pages)
README.md
.gitignore
outils/
  installer.bat             installation complète de l'application
  lancer_app.bat            lance l'application
  requirements-extra.txt    bibliothèques supplémentaires
  app.py                    interface de l'application (Streamlit)
  radar_csv.py              vidéo -> positions.csv + vidéo radar
  pitch_projection.py       projection image -> terrain fiabilisée
  analyse_positions.py      positions.csv -> statistiques, graphiques, Excel
  verifier_terrain.py       contrôle visuel et comparaison de réglages de projection
  affinage_lignes.py        recalage sur les lignes blanches (expérimental)
  installer_pnlcalib.bat    installation facultative de PnLCalib (environnement séparé)
  calibrer_video.bat        lance la calibration PnLCalib d'une vidéo
  pnl_export.py             export de la calibration PnLCalib (copié dans le dossier PnLCalib)
  preparer_dataset.py       réentraînement du modèle des points du terrain
  annotateur_terrain.html   outil d'annotation des points du terrain (navigateur)
  GUIDE_PNLCALIB.md         guide de la calibration PnLCalib
  GUIDE_REENTRAINEMENT.md   guide du réentraînement
```

## Limites connues

- Les tests ont été faits sur peu d'extraits (principalement un clip de 30 s de diffusion). Aucune garantie sur d'autres caméras, stades ou conditions.
- Il n'y a pas de positions de référence pour mesurer l'exactitude absolue des distances.
- Le traitement d'une vidéo depuis l'interface n'a pas été testé sur toutes les configurations.

## Licences et crédits

- Moteur de détection : [roboflow/sports](https://github.com/roboflow/sports) (licence MIT), qui s'appuie sur [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (licence **AGPL-3.0**) et sur [supervision](https://github.com/roboflow/supervision).
- Les modèles de détection et les vidéos d'exemple sont fournis par Roboflow : vérifie leurs conditions d'utilisation avant tout usage commercial.
- La licence AGPL-3.0 d'Ultralytics impose des obligations si tu proposes l'outil comme service à d'autres personnes. Pour un usage personnel, elle ne pose pas de difficulté.
- Calibration optionnelle : [PnLCalib](https://github.com/mguti97/PnLCalib) (code et poids sous leurs propres licences, à vérifier).
- Page web : export Excel avec [SheetJS](https://sheetjs.com/).
- **À toi d'ajouter ici la licence de ton propre code** (par exemple MIT) et un fichier `LICENSE`.
