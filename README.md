# Analyse de match de football

Outil pour analyser une vidéo de match : détection des joueurs, projection sur le terrain, nettoyage des trajectoires, puis statistiques (largeur et profondeur des équipes, distances, vitesses, cartes de chaleur) avec export Excel.

Tout se calcule sur ton ordinateur. Aucune vidéo et aucune donnée n'est envoyée sur internet.

## Comment ça marche

L'outil principal est une application qui tourne sur un PC Windows, dans le navigateur. En **un seul clic**, elle :

1. détecte les joueurs sur la vidéo et les projette sur le terrain (fichier `positions.csv`) ;
2. crée l'**Excel des données brutes** (`donnees_brutes.xlsx`), avant tout traitement ;
3. nettoie les trajectoires et calcule les statistiques ;
4. **affiche les données nettoyées** dans la page, avec le radar image par image, les cartes de chaleur et les tableaux par équipe et par joueur.

Une carte graphique Nvidia est idéale pour cette partie.

La page web `index.html` (GitHub Pages) est une version allégée, facultative : elle ne sait pas analyser de vidéo, seulement traiter un `positions.csv` déjà produit, directement dans le navigateur. Adresse : `https://<compte>.github.io/<depot>/`

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
3. Clique sur « Lancer l'analyse complète » et ne touche à rien pendant le calcul.
4. Les résultats s'affichent dans des onglets : **données nettoyées** (avec filtres par rôle, équipe et joueur), statistiques par équipe et par joueur, **radar image par image**, cartes de chaleur, données brutes et liste des fichiers produits.
5. Télécharge les Excel (`donnees_brutes.xlsx` et `analyse.xlsx`), les CSV ou la vidéo avec le radar. Tout est aussi enregistré dans le dossier `analyses`, à côté des scripts.

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
