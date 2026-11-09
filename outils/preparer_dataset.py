"""
Preparation du reentrainement du modele des points du terrain.

Etapes, dans l'ordre (voir GUIDE_REENTRAINEMENT.md) :

  1. extraire    : sort des images fixes de tes videos
  2. preannoter  : le modele actuel propose les points sur chaque image
  3. (annotation a la main avec annotateur_terrain.html, dans le navigateur)
  4. construire  : prepare les dossiers d'entrainement et de validation
  5. entrainer   : affine le modele sur tes images (carte graphique conseillee)
  6. evaluer     : compare l'ancien et le nouveau modele sur des images de validation

Tout se passe en local. A placer dans examples/soccer, avec main.py.

Exemples :
    python preparer_dataset.py extraire --videos data\\a.mp4 data\\b.mp4 --dossier dataset
    python preparer_dataset.py preannoter --dossier dataset --device cuda
    python preparer_dataset.py construire --dossier dataset
    python preparer_dataset.py entrainer --dossier dataset --device 0
    python preparer_dataset.py evaluer --dossier dataset ^
        --modeles data\\football-pitch-detection.pt dataset\\runs\\affinage\\weights\\best.pt
"""
import argparse
import json
import math
import os
import shutil
import sys

import cv2
import numpy as np

IMAGE_EXT = ('.jpg', '.jpeg', '.png')


# ----------------------------------------------------------------------------
# Utilitaires
# ----------------------------------------------------------------------------
def to_np(x):
    """Tenseur PyTorch ou tableau -> tableau NumPy."""
    if hasattr(x, 'cpu'):
        x = x.cpu().numpy()
    return np.asarray(x)


def list_images(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXT))


def stem(name):
    return os.path.splitext(name)[0]


def video_of(image_name):
    """Nom de la video d'origine : 'match1_000120.jpg' -> 'match1'."""
    return stem(image_name).rsplit('_', 1)[0]


def format_label(box_xywhn, kpts_px, vis, width, height):
    """Ligne de label YOLO pose : classe, boite, puis (x, y, visibilite) par point."""
    cx, cy, bw, bh = box_xywhn
    parts = ['0', f'{cx:.6f}', f'{cy:.6f}', f'{bw:.6f}', f'{bh:.6f}']
    for (x, y), v in zip(kpts_px, vis):
        if v:
            parts += [f'{x / width:.6f}', f'{y / height:.6f}', str(int(v))]
        else:
            parts += ['0.000000', '0.000000', '0']
    return ' '.join(parts)


def parse_label(path, n_points):
    """Lit un label YOLO pose. Retourne (boite xywhn, tableau (n,3) normalise) ou None."""
    with open(path, encoding='utf-8') as f:
        line = f.readline().split()
    if len(line) != 5 + 3 * n_points:
        return None
    vals = np.array([float(v) for v in line[1:]], dtype=np.float64)
    return vals[:4], vals[4:].reshape(n_points, 3)


def tight_box(kpts_px, vis, width, height, pad=0.03):
    """Boite (cx, cy, w, h) normalisee autour des points visibles, ou image entiere."""
    pts = np.asarray(kpts_px)[np.asarray(vis) > 0]
    if len(pts) == 0:
        return np.array([0.5, 0.5, 1.0, 1.0])
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    x0, x1 = max(0.0, x0 / width - pad), min(1.0, x1 / width + pad)
    y0, y1 = max(0.0, y0 / height - pad), min(1.0, y1 / height + pad)
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0])


def union_box(a, b):
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    x0, y0, x1, y1 = min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1)
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0])


def load_main():
    try:
        from main import CONFIG, PITCH_DETECTION_MODEL_PATH
    except ImportError as err:
        sys.exit(f"Impossible de charger main.py ({err}). Lance ce script depuis "
                 "examples\\soccer, avec l'environnement virtuel actif.")
    return CONFIG, PITCH_DETECTION_MODEL_PATH


# ----------------------------------------------------------------------------
# 1. Extraction des images
# ----------------------------------------------------------------------------
def cmd_extraire(args):
    CONFIG, model_path = load_main()
    images_dir = os.path.join(args.dossier, 'images')
    for sub in ('images', 'preannotations', 'labels'):
        os.makedirs(os.path.join(args.dossier, sub), exist_ok=True)

    vertices = np.asarray(CONFIG.vertices, dtype=float)
    with open(os.path.join(args.dossier, 'points_terrain.json'), 'w', encoding='utf-8') as f:
        json.dump({'points_cm': vertices.round(1).tolist(), 'longueur_cm': 12000,
                   'largeur_cm': 7000}, f)

    total = 0
    for video in args.videos:
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            print(f'Video introuvable ou illisible : {video}')
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(int(round(args.pas * fps)), 1)
        name = stem(os.path.basename(video)).replace('_', '-')
        index, saved = 0, 0
        while True:
            if not cap.grab():
                break
            if index % step == 0:
                path = os.path.join(images_dir, f'{name}_{index:06d}.jpg')
                if not os.path.exists(path):
                    ok, frame = cap.retrieve()
                    if ok:
                        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved += 1
                if args.max_par_video and saved >= args.max_par_video:
                    break
            index += 1
        cap.release()
        print(f'{video} : {saved} images (une toutes les {args.pas} s)')
        total += saved
    print(f'\n{total} images dans {images_dir}')
    print('Etape suivante : preannoter (le modele propose les points).')


# ----------------------------------------------------------------------------
# 2. Pre-annotation avec le modele actuel
# ----------------------------------------------------------------------------
def cmd_preannoter(args):
    CONFIG, model_path = load_main()
    from ultralytics import YOLO
    n_points = len(CONFIG.vertices)
    model = YOLO(args.modele or model_path).to(device=args.device)

    images_dir = os.path.join(args.dossier, 'images')
    pre_dir = os.path.join(args.dossier, 'preannotations')
    os.makedirs(pre_dir, exist_ok=True)
    names = list_images(images_dir)
    if not names:
        sys.exit(f'Aucune image dans {images_dir} : lance d\'abord "extraire".')

    done = 0
    for name in names:
        target = os.path.join(pre_dir, stem(name) + '.txt')
        if os.path.exists(target) and not args.refaire:
            continue
        image = cv2.imread(os.path.join(images_dir, name))
        if image is None:
            continue
        h, w = image.shape[:2]
        result = model(image, verbose=False)[0]

        kpts = np.zeros((n_points, 2))
        vis = np.zeros(n_points, dtype=int)
        box = None
        if result.keypoints is not None and len(result.keypoints.xy):
            xy = to_np(result.keypoints.xy)[0]
            conf = to_np(result.keypoints.conf)[0] if result.keypoints.conf is not None else np.ones(len(xy))
            if len(xy) == n_points:
                inside = (xy[:, 0] > 1) & (xy[:, 0] < w - 1) & (xy[:, 1] > 1) & (xy[:, 1] < h - 1)
                vis = ((conf >= args.conf) & inside).astype(int) * 2
                kpts = xy
            if result.boxes is not None:
                boxes = to_np(result.boxes.xyxyn)
                if boxes.ndim == 2 and len(boxes):
                    x0, y0, x1, y1 = boxes[0]
                    box = np.array([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0])
        tight = tight_box(kpts, vis, w, h)
        box = tight if box is None else union_box(box, tight)
        with open(target, 'w', encoding='utf-8') as f:
            f.write(format_label(box, kpts, vis, w, h) + '\n')
        done += 1
    print(f'{done} pre-annotations ecrites dans {pre_dir}')
    print('Etape suivante : ouvre annotateur_terrain.html pour verifier et corriger les points.')


# ----------------------------------------------------------------------------
# 4. Construction du jeu d'entrainement / validation
# ----------------------------------------------------------------------------
def cmd_construire(args):
    CONFIG, _ = load_main()
    n_points = len(CONFIG.vertices)
    root = os.path.abspath(args.dossier)
    images_dir, labels_dir = os.path.join(root, 'images'), os.path.join(root, 'labels')

    pairs = []
    for name in list_images(images_dir):
        label = os.path.join(labels_dir, stem(name) + '.txt')
        if os.path.exists(label) and parse_label(label, n_points) is not None:
            pairs.append(name)
    if not pairs:
        sys.exit(f'Aucune image annotee dans {labels_dir}. Valide des images avec annotateur_terrain.html.')

    # separation : de preference par video entiere, sinon fin chronologique de chaque video
    by_video = {}
    for name in pairs:
        by_video.setdefault(video_of(name), []).append(name)
    val_videos = set(args.val_videos or [])
    train, val = [], []
    for video, names in by_video.items():
        names.sort()
        if video in val_videos:
            val += names
        elif val_videos:
            train += names
        else:
            n_val = int(round(len(names) * args.val)) if len(names) >= 6 else 0
            train += names[:len(names) - n_val]
            val += names[len(names) - n_val:]
    if not val:
        print('ATTENTION : aucune image de validation (trop peu d\'images annotees). '
              'L\'evaluation ne sera pas fiable.')
        val = train[-1:]

    out = os.path.join(root, 'yolo')
    if os.path.isdir(out):
        shutil.rmtree(out)
    for split, names in (('train', train), ('val', val)):
        os.makedirs(os.path.join(out, 'images', split))
        os.makedirs(os.path.join(out, 'labels', split))
        for name in names:
            shutil.copy2(os.path.join(images_dir, name), os.path.join(out, 'images', split, name))
            shutil.copy2(os.path.join(labels_dir, stem(name) + '.txt'),
                         os.path.join(out, 'labels', split, stem(name) + '.txt'))

    yaml_path = os.path.join(root, 'pitch.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(f'path: {out.replace(chr(92), "/")}\n')
        f.write('train: images/train\nval: images/val\n')
        f.write(f'kpt_shape: [{n_points}, 3]\n')
        f.write(f'flip_idx: {list(range(n_points))}\n')  # inutilise : on interdit le retournement
        f.write('names:\n  0: pitch\n')
    print(f'{len(train)} images d\'entrainement, {len(val)} de validation -> {out}')
    print(f'Configuration : {yaml_path}')
    if len(train) < 50:
        print('ATTENTION : moins de 50 images d\'entrainement. Le gain sera limite et le '
              'risque de surapprentissage eleve : annote plus d\'images si possible.')


# ----------------------------------------------------------------------------
# 5. Entrainement
# ----------------------------------------------------------------------------
def cmd_entrainer(args):
    CONFIG, model_path = load_main()
    from ultralytics import YOLO
    base = args.modele or model_path
    model = YOLO(base)
    imgsz = args.imgsz
    if imgsz is None:
        try:
            imgsz = int(model.ckpt.get('train_args', {}).get('imgsz', 640))
        except Exception:
            imgsz = 640
    yaml_path = os.path.join(os.path.abspath(args.dossier), 'pitch.yaml')
    if not os.path.exists(yaml_path):
        sys.exit('pitch.yaml introuvable : lance d\'abord "construire".')
    print(f'Affinage de {base} (taille d\'image {imgsz}, {args.epochs} passes maximum)')
    model.train(
        data=yaml_path, epochs=args.epochs, imgsz=imgsz, batch=args.batch,
        device=args.device, lr0=args.lr0, lrf=0.1, patience=args.patience,
        warmup_epochs=1,
        fliplr=0.0, flipud=0.0,        # jamais de miroir : gauche et droite seraient confondus
        mosaic=0.0, degrees=0.0,       # une seule vue du terrain par image, sans rotation
        scale=0.25, translate=0.08,
        project=os.path.join(os.path.abspath(args.dossier), 'runs'), name=args.nom,
        exist_ok=False, workers=args.workers, plots=False)
    best = os.path.join(os.path.abspath(args.dossier), 'runs', args.nom, 'weights', 'best.pt')
    print(f'\nTermine. Meilleur modele : {best}')
    print('Etape suivante : evaluer, en comparant avec le modele d\'origine.')


# ----------------------------------------------------------------------------
# 6. Evaluation avant / apres
# ----------------------------------------------------------------------------
def evaluate_model(model, images_dir, labels_dir, n_points, conf_thr, px_thr):
    errors, n_gt, n_found, n_pred, n_bad = [], 0, 0, 0, 0
    for name in list_images(images_dir):
        label = parse_label(os.path.join(labels_dir, stem(name) + '.txt'), n_points)
        image = cv2.imread(os.path.join(images_dir, name))
        if label is None or image is None:
            continue
        h, w = image.shape[:2]
        gt = label[1]
        gt_xy = gt[:, :2] * [w, h]
        gt_vis = gt[:, 2] >= 2
        n_gt += int(gt_vis.sum())

        result = model(image, verbose=False)[0]
        if result.keypoints is None or not len(result.keypoints.xy):
            continue
        xy = to_np(result.keypoints.xy)[0]
        conf = to_np(result.keypoints.conf)[0] if result.keypoints.conf is not None else np.ones(len(xy))
        if len(xy) != n_points:
            continue
        inside = (xy[:, 0] > 1) & (xy[:, 0] < w - 1) & (xy[:, 1] > 1) & (xy[:, 1] < h - 1)
        pred = (conf >= conf_thr) & inside
        err = np.hypot(*(xy - gt_xy).T)
        n_pred += int(pred.sum())
        n_bad += int((pred & (~gt_vis | (err > 3 * px_thr))).sum())
        ok = pred & gt_vis
        errors += err[ok].tolist()
        n_found += int((ok & (err <= px_thr)).sum())
    e = np.array(errors) if errors else np.array([np.nan])
    return {
        'points_reference': n_gt,
        'rappel': 100 * n_found / max(n_gt, 1),
        'erreur_med': float(np.nanmedian(e)),
        'erreur_p90': float(np.nanpercentile(e, 90)),
        'faux': 100 * n_bad / max(n_pred, 1),
    }


def cmd_evaluer(args):
    CONFIG, model_path = load_main()
    from ultralytics import YOLO
    n_points = len(CONFIG.vertices)
    root = os.path.abspath(args.dossier)
    images_dir = os.path.join(root, 'yolo', 'images', 'val')
    labels_dir = os.path.join(root, 'yolo', 'labels', 'val')
    if not list_images(images_dir):
        sys.exit('Aucune image de validation : lance d\'abord "construire".')
    models = args.modeles or [model_path]

    print(f"{len(list_images(images_dir))} images de validation (jamais vues a l'entrainement)\n")
    header = (f"{'modele':46s} {'rappel':>8s} {'erreur med.':>12s} {'erreur p90':>11s} {'faux points':>12s}")
    print(header)
    print('-' * len(header))
    for path in models:
        model = YOLO(path).to(device=args.device)
        m = evaluate_model(model, images_dir, labels_dir, n_points, args.conf, args.px)
        label = path if len(path) <= 46 else '...' + path[-43:]
        print(f"{label:46s} {m['rappel']:7.1f}% {m['erreur_med']:10.1f}px {m['erreur_p90']:9.1f}px {m['faux']:11.1f}%")
    print(f"\nrappel : part des points visibles retrouves a moins de {args.px:g} px (confiance >= {args.conf:g})")
    print('faux points : part des points annonces qui sont absents de l\'annotation ou tres eloignes')
    print('Le meilleur modele combine un rappel eleve, une erreur faible et peu de faux points.')


# ----------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description='Reentrainement du modele des points du terrain')
    sub = p.add_subparsers(dest='commande', required=True)

    e = sub.add_parser('extraire', help='extrait des images des videos')
    e.add_argument('--videos', nargs='+', required=True)
    e.add_argument('--dossier', default='dataset')
    e.add_argument('--pas', type=float, default=1.0, help='une image toutes les N secondes (defaut 1)')
    e.add_argument('--max_par_video', type=int, default=None)
    e.set_defaults(func=cmd_extraire)

    a = sub.add_parser('preannoter', help='le modele actuel propose les points')
    a.add_argument('--dossier', default='dataset')
    a.add_argument('--device', default='cpu')
    a.add_argument('--conf', type=float, default=0.5, help='confiance minimale (defaut 0.5)')
    a.add_argument('--modele', default=None)
    a.add_argument('--refaire', action='store_true', help='ecrase les pre-annotations existantes')
    a.set_defaults(func=cmd_preannoter)

    c = sub.add_parser('construire', help='prepare les dossiers entrainement / validation')
    c.add_argument('--dossier', default='dataset')
    c.add_argument('--val', type=float, default=0.15, help='part de validation (defaut 0.15)')
    c.add_argument('--val_videos', nargs='*', default=None,
                   help='videos reservees entierement a la validation (recommande)')
    c.set_defaults(func=cmd_construire)

    t = sub.add_parser('entrainer', help='affine le modele sur tes images')
    t.add_argument('--dossier', default='dataset')
    t.add_argument('--modele', default=None, help='modele de depart (defaut : le modele actuel)')
    t.add_argument('--epochs', type=int, default=50)
    t.add_argument('--batch', type=int, default=8)
    t.add_argument('--imgsz', type=int, default=None)
    t.add_argument('--device', default='0')
    t.add_argument('--lr0', type=float, default=0.002)
    t.add_argument('--patience', type=int, default=15)
    t.add_argument('--workers', type=int, default=0)
    t.add_argument('--nom', default='affinage')
    t.set_defaults(func=cmd_entrainer)

    v = sub.add_parser('evaluer', help='compare des modeles sur les images de validation')
    v.add_argument('--dossier', default='dataset')
    v.add_argument('--modeles', nargs='+', default=None)
    v.add_argument('--device', default='cpu')
    v.add_argument('--conf', type=float, default=0.5)
    v.add_argument('--px', type=float, default=15.0, help='ecart tolere en pixels (defaut 15)')
    v.set_defaults(func=cmd_evaluer)
    return p


if __name__ == '__main__':
    args = build_parser().parse_args()
    args.func(args)
