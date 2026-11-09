# Calibration plus précise avec PnLCalib (facultatif)

Par défaut, l'application place les joueurs sur le terrain à partir des points du terrain détectés par le modèle de Roboflow. Sur le bord proche de la caméra, il n'existe presque aucun point reconnaissable (la ligne de touche ne croise que la ligne médiane, et les coins sont souvent hors champ) : la ligne du bas peut alors pencher.

**PnLCalib** est un modèle de recherche entraîné sur des matchs de diffusion. Il utilise à la fois les points et les lignes du terrain, et estime une caméra complète. Sur l'extrait de test de 30 s, la ligne de touche du bas était tracée à environ 5 px de la vraie ligne (contre 16 à 72 px avec les points seuls).

Il s'installe **à part** (dossier et environnement Python séparés) et n'est utilisé qu'en option.

## Avantages et limites

| | |
|---|---|
| Plus juste sur le bord proche de la caméra | oui, sur l'extrait mesuré |
| Terrain standard de 105 × 68 m | oui (le gabarit des points du terrain est de 120 × 70 m) |
| Rapidité | environ 3 images par seconde : 4 minutes pour 30 s de vidéo |
| Stabilité d'une image à l'autre | bruitée : le programme lisse la calibration et écarte les sauts |
| Début de plan large | les premières images (environ 1,6 s sur l'extrait de test) peuvent être mal calibrées et sont écartées |
| Licence | à vérifier sur le dépôt de PnLCalib avant tout usage hors test personnel |

Ces mesures viennent d'un seul extrait : il n'y a aucune garantie sur d'autres caméras ou d'autres stades.

## Installation

1. Double-clique sur **`outils\installer_pnlcalib.bat`**. Il télécharge le dépôt PnLCalib (Git est nécessaire, sinon télécharge-le en ZIP et décompresse-le dans `outils\pnlcalib\PnLCalib`), crée un environnement Python 3.12 séparé et installe les bibliothèques. Prévoir environ 5 Go.
2. **Télécharge les poids à la main** : ouvre `outils\pnlcalib\PnLCalib\README.md`, repère la section *Weights* et télécharge les deux fichiers **`SV_kp`** (points) et **`SV_lines`** (lignes). Place-les dans `outils\pnlcalib\PnLCalib\weights`.

Le script d'installation n'a pas pu être testé de bout en bout. Les mêmes étapes, faites à la main, ont fonctionné : `git clone`, environnement séparé, PyTorch avec CUDA, puis `pip install opencv-python pyyaml scipy matplotlib tqdm yacs pillow scikit-learn`. Le fichier `requirements.txt` du projet **ne s'installe pas** sous Python 3.12 (versions trop anciennes) : ne l'utilise pas.

## Utilisation

### 1. Calibrer la vidéo

Double-clique sur `calibrer_video.bat` depuis une invite de commandes, avec des chemins complets :

```
outils\calibrer_video.bat C:\chemin\ma_video.mp4 C:\chemin\calibration.csv
```

Un troisième argument facultatif calibre une image sur N, pour aller N fois plus vite (par exemple `5`). Les images intermédiaires sont comblées par interpolation. Sur l'extrait de test, calibrer une image sur 5 donne un écart médian de 15 cm avec la calibration complète (76 cm pour 5 % des cas).

Le fichier produit contient une homographie par image (colonnes `frame`, `ok`, `h00` à `h22`, `focale`). Les images sans calibration valable ont `ok = 0`.

### 2. Analyser avec cette calibration

**Dans l'application** : dans le panneau « Réglages de la détection », charge le fichier `calibration.csv` dans « Calibration PnLCalib », puis lance l'analyse comme d'habitude.

**En ligne de commande** (depuis `examples\soccer`, avec l'environnement de l'application actif) :

```
python radar_csv.py --source_video_path data\ma_video.mp4 --target_video_path data\radar.mp4 --csv_path data\positions.csv --device cuda --no_preview --calibration data\calibration.csv
python analyse_positions.py --csv_path data\positions.csv --output_dir data\analyse
```

Les positions sont alors exprimées sur un terrain de 105 × 68 m. Les dimensions sont enregistrées dans le CSV (colonnes `pitch_l_cm` et `pitch_w_cm`), et l'analyse comme la page web s'y adaptent.

## Ce que fait le nettoyage de la calibration

Avant utilisation, `radar_csv.py` :
- écarte les images dont la projection est invraisemblable ;
- comble les trous de 10 images ou moins ;
- coupe aux sauts de caméra et abandonne les états de moins de 15 images ;
- lisse la calibration dans le temps, sans jamais moyenner deux états différents ;
- rejette ensuite toute image dont la calibration saute de plus de 3 m d'un coup.

Options : `--lissage_calibration` (largeur du lissage, 0 pour le désactiver) et `--ignorer_debut_s` (ignorer les N premières secondes).

## Dépannage

| Problème | Solution |
|---|---|
| `installer_pnlcalib.bat` : Git introuvable | Installe Git (git-scm.com), ou décompresse le ZIP du dépôt dans `outils\pnlcalib\PnLCalib` |
| `No module named 'cv2'` (ou un autre module) | Dans l'environnement de PnLCalib : `pip install opencv-python` (ou le module manquant) |
| `pip install -r requirements.txt` échoue | Normal : n'utilise pas ce fichier, installe les bibliothèques une à une (voir plus haut) |
| Poids introuvables | Vérifie que `SV_kp` et `SV_lines` sont dans `outils\pnlcalib\PnLCalib\weights` |
| Résultat très lent | Utilise le troisième argument de `calibrer_video.bat` (par exemple `5`) |
| Positions décalées dans les premières secondes | Normal : la calibration est instable au début d'un plan large. Utilise `--ignorer_debut_s 2` |
