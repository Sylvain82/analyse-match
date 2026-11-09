# Réentraîner le modèle des points du terrain

Ce guide explique comment affiner le modèle `football-pitch-detection.pt` sur **tes propres vidéos**, pour qu'il repère mieux les points du terrain (notamment du côté proche de la caméra) et que le radar soit plus juste.

Tout se passe sur ton PC : aucune image n'est envoyée en ligne.

**À savoir avant de commencer**
- Compte 5 à 8 heures d'annotation pour 300 à 500 images, répartissables sur plusieurs séances.
- L'entraînement dure de quelques dizaines de minutes à quelques heures sur une carte graphique Nvidia.
- Le gain n'est pas garanti : on le mesure à la fin, et l'ancien modèle reste disponible en secours.
- Un réentraînement ne supprime pas la distorsion optique des caméras TV, il réduit surtout les points absents ou mal placés.

Tous les scripts se lancent depuis `C:\Users\Sylvain\sports\examples\soccer`, avec l'environnement virtuel actif (`(venv)` au début de la ligne).

---

## Étape 0 : choisir les vidéos

- Prends **plusieurs matchs ou stades différents** si possible. Un modèle entraîné sur un seul stade apprend ce stade et pas le football en général.
- Garde de côté **une ou deux vidéos pour la validation** : elles ne serviront pas à l'entraînement, mais à vérifier honnêtement que le nouveau modèle est meilleur.
- Préfère les plans larges où le terrain est bien visible, y compris le côté proche de la caméra.

## Étape 1 : extraire des images

```
python preparer_dataset.py extraire --videos data\match1.mp4 data\match2.mp4 data\match3.mp4 --dossier dataset
```

Une image par seconde est extraite de chaque vidéo, dans `dataset\images`. Pour limiter le nombre d'images par vidéo (par exemple 120) : ajoute `--max_par_video 120`.

## Étape 2 : pré-annoter avec le modèle actuel

```
python preparer_dataset.py preannoter --dossier dataset --device cuda
```

Le modèle propose les points sur chaque image (fichiers dans `dataset\preannotations`). C'est un point de départ : tu corrigeras ses erreurs à l'étape suivante.

## Étape 3 : annoter dans le navigateur

1. Ouvre le fichier **`annotateur_terrain.html`** avec **Chrome ou Edge** (clic droit puis « Ouvrir avec »). Firefox ne permet pas d'écrire dans un dossier.
2. Clique sur **« Ouvrir le dossier du jeu de données »**, choisis le dossier `dataset` et autorise l'écriture.
3. Pour chaque image :
   - Les **points verts** sont cohérents entre eux. Un **cercle rouge** signale un point suspect : déplace-le (glisse-le) ou retire-le (clique dessus, puis touche `X`).
   - Les **cercles orange en pointillés** sont des points absents dont la projection prédit la position. La touche **`A`** les place tous d'un coup ; un clic sur l'un d'eux ne place que celui-là.
   - Pour placer un point à la main : clique sur son numéro dans le **schéma** à droite, puis clique dans l'image à son emplacement exact.
   - Les **lignes bleu clair** montrent le terrain reconstitué : elles doivent coller aux lignes blanches de l'image.
   - Quand l'image est correcte, appuie sur **`Entrée`** : elle est enregistrée et la suivante s'ouvre.
4. Tu peux arrêter à tout moment : à la réouverture, l'outil reprend à la première image non validée.

**Règles d'annotation (importantes)**
- Ne place que les points **réellement visibles** dans l'image. Un point caché derrière un panneau, un joueur ou une tribune n'est pas visible : laisse-le absent.
- Mieux vaut **oublier** un point incertain que le placer à peu près : une erreur d'annotation apprend de fausses habitudes au modèle.
- **Passe les images inutilisables** (gros plans, rediffusions, plans de tribune) avec la flèche `→`, sans les valider.
- Sois **régulier** : place chaque point au même repère exact (intersection de lignes, centre du point de penalty, etc.).
- Il faut au moins 4 points pour valider, idéalement 6 ou plus.

## Étape 4 : construire le jeu d'entraînement

```
python preparer_dataset.py construire --dossier dataset --val_videos match3
```

`match3` est le nom d'une vidéo (sans `.mp4`, avec les `_` remplacés par `-`) réservée à la validation. Sans cette option, la fin de chaque vidéo sert de validation, ce qui est moins rigoureux. Le script indique le nombre d'images d'entraînement et de validation, et avertit s'il y en a moins de 50.

## Étape 5 : entraîner

```
python preparer_dataset.py entrainer --dossier dataset --device 0
```

Le modèle actuel est affiné pendant 50 passes au maximum (arrêt anticipé si la progression stagne). Le meilleur modèle est enregistré dans `dataset\runs\affinage\weights\best.pt`.

Quelques réglages utiles : `--epochs 80`, `--batch 4` (si la mémoire de la carte graphique est saturée), `--lr0 0.001` (affinage plus prudent). Les retournements en miroir sont désactivés volontairement : ils confondraient les côtés gauche et droit du terrain.

## Étape 6 : comparer avant et après

```
python preparer_dataset.py evaluer --dossier dataset --device cuda --modeles data\football-pitch-detection.pt dataset\runs\affinage\weights\best.pt
```

Le tableau compare les deux modèles sur les images de validation :
- **rappel** : part des points visibles retrouvés à moins de 15 pixels. Plus il est élevé, mieux c'est.
- **erreur** : écart médian et du 90e centile, en pixels.
- **faux points** : points annoncés alors qu'ils sont absents ou très éloignés.

Le nouveau modèle est meilleur s'il améliore le rappel et réduit l'erreur sans augmenter les faux points. S'il est moins bon, ne l'utilise pas (voir « Si les résultats sont décevants »).

## Étape 7 : utiliser le nouveau modèle

1. Garde l'ancien : renomme `data\football-pitch-detection.pt` en `data\football-pitch-detection-original.pt`.
2. Copie `best.pt` dans `data\` sous le nom `football-pitch-detection.pt`.
3. Refais le contrôle visuel : `python verifier_terrain.py --source_video_path data\ta_video.mp4 --device cuda --compare`.

Pour revenir en arrière, il suffit de remettre l'ancien fichier.

---

## Si les résultats sont décevants

- **Plus d'images, plus variées** : c'est presque toujours le levier principal.
- **Vérifie la cohérence des annotations** : un point placé tantôt sur la ligne, tantôt à côté, brouille l'apprentissage.
- **Affine plus doucement** : `--lr0 0.001` et moins de passes.
- **Compare sur une vidéo absente de l'entraînement** : un bon score sur les images d'entraînement ne prouve rien.

## Dépannage

| Problème | Solution |
|---|---|
| `Impossible de charger main.py` | Lance le script depuis `examples\soccer`, avec `(venv)` actif |
| `CUDA out of memory` | `--batch 4`, puis `--batch 2` |
| L'annotateur ne s'ouvre pas sur le dossier | Utilise Chrome ou Edge, et choisis le dossier qui contient `points_terrain.json` |
| Le modèle ne retrouve plus certains points | Vérifie que les points visibles sont tous annotés : un point visible non annoté apprend à l'ignorer |
| Résultat pire qu'avant | Reviens à l'ancien modèle et reprends avec plus d'images |

## Licence

L'entraînement utilise Ultralytics (licence AGPL-3.0). Un usage personnel ne pose pas de difficulté ; si tu proposes l'outil à d'autres, lis les conditions de cette licence.
