# Analyse de match de football

Outil pour analyser une vidéo de match : détection des joueurs, projection sur le terrain, nettoyage des trajectoires, puis statistiques (largeur et profondeur des équipes, distances, vitesses, cartes de chaleur) avec export Excel.

Tout se calcule sur ton ordinateur. Aucune vidéo et aucune donnée n'est envoyée sur internet.

## Comment ça marche

L'outil se compose de deux parties :

1. **Sur un PC Windows** : la vidéo est analysée par l'IA et transformée en fichier `positions.csv` (une ligne par joueur et par image). Cette partie demande une installation (voir plus bas) et, idéalement, une carte graphique Nvidia.
2. **Sur la page web** : le fichier `positions.csv` est déposé sur la page, qui calcule les statistiques directement dans le navigateur. Aucune installation n'est nécessaire.

Page web : `https://sylvain82.github.io/analyse-match/)` (adresse de ce dépôt sur GitHub Pages)

Si tu as déjà un `positions.csv`, la page web suffit.

## Ce qu'il faut

- Windows 10 ou 11
- Environ 10 Go d'espace disque libre
- Une carte graphique Nvidia est fortement conseillée : sans elle, l'analyse d'une vidéo est très lente
- Une connexion internet pour l'installation (pas pour l'utilisation)

## Installation

1. Télécharge ce dépôt : bouton vert **Code**, puis **Download ZIP**, et décompresse-le.
2. Ouvre le dossier `outils` et double-clique sur **`installer.bat`**.
3. Attends la fin (10 à 30 minutes). Le script installe Python 3.12 si besoin (avec winget), télécharge le moteur [roboflow/sports](https://github.com/roboflow/sports), crée un environnement isolé, installe les bibliothèques (avec PyTorch CUDA si une carte Nvidia est détectée) et télécharge les modèles d'IA ainsi qu'une vidéo d'exemple.

L'installation ne touche pas au reste de ton système : tout est dans `outils\moteur`.

## Utilisation

1. Double-clique sur **`outils\lancer_app.bat`**. Ton navigateur s'ouvre sur l'application.
2. Choisis « Une vidéo de match » et dépose ta vidéo. Pour un premier essai, règle « Analyser seulement les N premières secondes » sur 30.
3. Lance l'analyse et ne touche à rien pendant le calcul.
4. Télécharge l'Excel, les CSV ou la vidéo avec le radar.

Tu peux aussi déposer directement un `positions.csv` déjà produit : l'application ne refait alors que les statistiques.

## Conseils pour de bons résultats

- Utilise de préférence une **vidéo de diffusion TV**, avec le terrain bien visible.
- Commence par un **extrait court**.
- Les statistiques d'équipe (largeur, profondeur, compacité) sont les plus fiables. Les **distances et vitesses sont des estimations** : la projection sur le terrain peut se tromper quand la caméra zoome ou change de plan, et la vitesse maximale est la mesure la moins fiable.

## Confidentialité

- Les calculs de la vidéo se font localement.
- La page web traite le fichier dans ton navigateur. Seule la bibliothèque d'export Excel (SheetJS) est chargée depuis internet : c'est du code, pas tes données.
- Le fichier `.gitignore` empêche d'envoyer par erreur des CSV, vidéos, modèles ou classeurs Excel sur GitHub. **Ne publie jamais de vidéo ni de `positions.csv` dans ce dépôt.**

## Dépannage

| Problème | Solution |
|---|---|
| `installer.bat` ne trouve pas Python 3.12 | Installe-le depuis python.org en cochant « Add python.exe to PATH », ferme la fenêtre et relance |
| Erreur « Microsoft Visual C++ 14.0 or greater is required » | Mauvaise version de Python : l'installeur utilise la 3.12, qui évite ce problème. Supprime le dossier `outils\moteur\venv` et relance |
| Le téléchargement d'un modèle échoue (« Too many users… ») | Google Drive limite temporairement le fichier : relance `installer.bat` plus tard |
| `module 'supervision' has no attribute 'ByteTrack'` | Lance dans l'environnement : `pip install "supervision==0.25.1"` |
| L'analyse est très lente | Sans carte Nvidia, c'est normal. Teste sur des extraits courts |
| La page web affiche « Colonnes manquantes » | Le fichier n'est pas un `positions.csv` produit par l'outil |

## Contenu du dépôt

```
index.html              page web d'analyse (GitHub Pages)
outils/
  installer.bat         installation complète
  lancer_app.bat        lance l'application locale
  app.py                interface de l'application (Streamlit)
  radar_csv.py          vidéo -> positions.csv + vidéo radar
  analyse_positions.py  positions.csv -> statistiques, graphiques, Excel
  requirements-extra.txt
```

## Licences et crédits

- Le moteur de détection vient de [roboflow/sports](https://github.com/roboflow/sports) (licence MIT), qui s'appuie sur [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (licence AGPL-3.0) et sur la bibliothèque [supervision](https://github.com/roboflow/supervision).
- Les modèles de détection et les vidéos d'exemple sont fournis par Roboflow : vérifie leurs conditions d'utilisation avant tout usage commercial.
- La licence AGPL-3.0 d'Ultralytics impose des obligations si tu proposes l'outil comme service à d'autres personnes. Pour un usage personnel, elle ne pose pas de difficulté.
- À toi d'ajouter ici la licence de ton propre code (par exemple MIT).
