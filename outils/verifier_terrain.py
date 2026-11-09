"""
Verification de la projection sur le terrain.

Pour chaque image, le script superpose a la video les lignes du terrain
reconstituees a partir des points detectes :
  - en ROUGE : methode d'origine (tous les points detectes, sans filtre)
  - en VERT  : methode fiabilisee (points confiants + RANSAC + rejet des sauts)

Option --lignes : en plus, la projection est affinee sur les lignes blanches visibles
dans l'image (voir affinage_lignes.py) et tracee en JAUNE. Utile quand peu de points sont
detectes sur le bord proche de la camera.

Les points detectes sont dessines avec un code couleur :
  vert = retenu, rouge = ecarte comme aberrant (RANSAC), gris = confiance trop faible.

Un bilan chiffre est affiche a la fin, avec l'erreur de reprojection : l'ecart
moyen, en pixels, entre les points detectes et leur position theorique d'apres
la projection. Plus il est faible, mieux la projection est alignee.

Mode --compare : sans produire de video, essaie plusieurs reglages sur la meme
video (le modele n'est execute qu'une fois) et affiche un tableau comparatif.

A placer dans examples/soccer, avec main.py et pitch_projection.py.

    python verifier_terrain.py --source_video_path data\\08fd33_0.mp4 ^
        --target_video_path data\\verif_terrain.mp4 --device cuda
    python verifier_terrain.py --source_video_path data\\08fd33_0.mp4 ^
        --device cuda --compare
"""
import argparse

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from main import CONFIG, PITCH_DETECTION_MODEL_PATH
from pitch_projection import PitchProjector

# Dimensions du terrain (cm), identiques a la configuration du depot roboflow
L, W = 12000, 7000
PEN_W, PEN_L = 4100, 2015
GOAL_W, GOAL_L = 1832, 550
CIRCLE_R = 915

GREEN, RED, GREY, WHITE = (60, 200, 60), (50, 50, 230), (150, 150, 150), (255, 255, 255)
YELLOW = (0, 215, 255)

# Reglages essayes par --compare : (nom, confiance min, tolerance RANSAC en cm, lissage)
COMPARE_CONFIGS = [
    ('actuel (0,5 / 200 cm / lissage 0,6)', 0.5, 200.0, 0.6),
    ('sans lissage (0,5 / 200 cm / 1,0)', 0.5, 200.0, 1.0),
    ('tolerance serree (0,5 / 100 cm / 1,0)', 0.5, 100.0, 1.0),
    ('plus de points (0,3 / 100 cm / 1,0)', 0.3, 100.0, 1.0),
    ('plus de points, tol. 150 (0,3 / 150 / 1,0)', 0.3, 150.0, 1.0),
    ('encore plus de points (0,2 / 100 / 1,0)', 0.2, 100.0, 1.0),
    ('aucun filtre de confiance (0 / 100 / 1,0)', 0.0, 100.0, 1.0),
]


def pitch_lines():
    """Lignes du terrain sous forme de listes de points (en cm)."""
    def rect(x0, y0, x1, y1):
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], np.float32)
    ang = np.linspace(0, 2 * np.pi, 64)
    circle = np.stack([L / 2 + CIRCLE_R * np.cos(ang), W / 2 + CIRCLE_R * np.sin(ang)], 1)
    py0, py1 = (W - PEN_W) / 2, (W + PEN_W) / 2
    gy0, gy1 = (W - GOAL_W) / 2, (W + GOAL_W) / 2
    return [
        rect(0, 0, L, W),
        np.array([[L / 2, 0], [L / 2, W]], np.float32),
        circle.astype(np.float32),
        rect(0, py0, PEN_L, py1), rect(L - PEN_L, py0, L, py1),
        rect(0, gy0, GOAL_L, gy1), rect(L - GOAL_L, gy0, L, gy1),
    ]


def draw_lines(frame, H_image_to_pitch, color, thickness):
    """Dessine les lignes du terrain sur l'image, a partir de l'homographie image -> terrain."""
    try:
        H_inv = np.linalg.inv(H_image_to_pitch)
    except np.linalg.LinAlgError:
        return
    for pts in pitch_lines():
        out = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H_inv)
        out = np.clip(out.reshape(-1, 2), -20000, 20000).astype(np.int32)
        cv2.polylines(frame, [out.reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA)


def first_keypoints(keypoints):
    if keypoints.xy is None or len(keypoints.xy) == 0:
        return None, None
    conf = None
    if getattr(keypoints, 'confidence', None) is not None and len(keypoints.confidence):
        conf = keypoints.confidence[0]
    return keypoints.xy[0], conf


def inlier_indices(xy, conf, H, vertices, kp_conf, ransac_cm):
    """Indices des points retenus : confiants, et coherents avec la projection H."""
    present = (xy[:, 0] > 1) & (xy[:, 1] > 1)
    confident = present if (conf is None or kp_conf <= 0) else present & (conf > kp_conf)
    idx = np.flatnonzero(confident)
    if H is None or len(idx) == 0:
        return np.array([], dtype=int), confident
    proj = PitchProjector.transform(xy[idx], H)
    err = np.hypot(proj[:, 0] - vertices[idx, 0], proj[:, 1] - vertices[idx, 1])
    return idx[err <= ransac_cm], confident


def reproj_error_px(xy, vertices, H, idx):
    """Ecart (pixels) entre les points detectes et leur position theorique d'apres H."""
    if H is None or len(idx) == 0:
        return np.array([])
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.array([])
    img = PitchProjector.transform(vertices[idx], H_inv)
    return np.hypot(img[:, 0] - xy[idx, 0], img[:, 1] - xy[idx, 1])


def summarize(counts, n_frames, points_used, errors):
    for key, label in [('ok', 'projection utilisable'), ('rejetee', 'projection rejetee (saut)'),
                       ('absente', 'sans assez de points')]:
        print(f'  {label:28s}: {counts[key]:5d}  ({100 * counts[key] / max(n_frames, 1):.1f} %)')
    if points_used:
        pts = np.array(points_used)
        print(f'Points retenus par image : mediane {np.median(pts):.0f}, minimum {pts.min()}, '
              f'au moins 6 points sur {100 * (pts >= 6).mean():.0f} % des images utilisables')
    if errors:
        e = np.array(errors)
        print(f'Erreur de reprojection : mediane {np.median(e):.1f} px, '
              f'90 % des images sous {np.percentile(e, 90):.1f} px, maximum {e.max():.1f} px')


def run_video(source_video_path, target_video_path, device, kp_conf, ransac_cm, smoothing, preview,
              refine_lines=False):
    model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    vertices = np.array(CONFIG.vertices, dtype=np.float32)
    projector = PitchProjector(vertices, kp_conf=kp_conf, ransac_cm=ransac_cm, smoothing=smoothing)
    info = sv.VideoInfo.from_video_path(source_video_path)

    counts = {'ok': 0, 'rejetee': 0, 'absente': 0}
    points_used, errors, n_frames = [], [], 0
    refiner = None
    refined = {'tentees': 0, 'acceptees': 0, 'decalage': []}
    if refine_lines:
        from affinage_lignes import LineRefiner
        refiner = LineRefiner()
    with sv.VideoSink(target_video_path, info) as sink:
        for frame in sv.get_video_frames_generator(source_path=source_video_path):
            n_frames += 1
            h, w = frame.shape[:2]
            result = model(frame, verbose=False)[0]
            xy, conf = first_keypoints(sv.KeyPoints.from_ultralytics(result))
            out = frame.copy()

            if xy is None or len(xy) != len(vertices):
                counts['absente'] += 1
                cv2.putText(out, 'Aucun point du terrain detecte', (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED, 2, cv2.LINE_AA)
            else:
                present = (xy[:, 0] > 1) & (xy[:, 1] > 1)
                if present.sum() >= 4:  # methode d'origine : tous les points, moindres carres
                    H_old, _ = cv2.findHomography(xy[present].astype(np.float32), vertices[present])
                    if H_old is not None:
                        draw_lines(out, H_old, RED, 2)

                H_new, n_in, state = projector.update(xy, conf, (w, h))
                counts[state] += 1
                idx, confident = inlier_indices(xy, conf, H_new, vertices, kp_conf, ransac_cm)
                if H_new is not None:
                    draw_lines(out, H_new, GREEN, 3)
                    if refiner is not None:
                        H_ref, ref_info = refiner.refine(frame, H_new, xy[idx], vertices[idx])
                        refined['tentees'] += 1
                        if ref_info['ok']:
                            refined['acceptees'] += 1
                            refined['decalage'].append(ref_info['deplacement_px'])
                            draw_lines(out, H_ref, YELLOW, 3)
                    points_used.append(n_in)
                    err = reproj_error_px(xy, vertices, H_new, idx)
                    if len(err):
                        errors.append(float(np.median(err)))

                inlier = np.zeros(len(xy), dtype=bool)
                inlier[idx] = True
                for i in np.flatnonzero(present):
                    color = GREEN if inlier[i] else (RED if confident[i] else GREY)
                    cv2.circle(out, (int(xy[i][0]), int(xy[i][1])), 6, color, -1, cv2.LINE_AA)
                    cv2.circle(out, (int(xy[i][0]), int(xy[i][1])), 6, WHITE, 1, cv2.LINE_AA)
                label = {'ok': f'Projection OK ({n_in} points retenus)',
                         'rejetee': 'Projection REJETEE (saut)',
                         'absente': 'Pas assez de points'}[state]
                cv2.putText(out, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            GREEN if state == 'ok' else RED, 2, cv2.LINE_AA)

            legend = "Rouge = origine   Vert = fiabilisee" + ("   Jaune = affinee sur les lignes" if refiner is not None else "")
            cv2.putText(out, legend,
                        (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)
            sink.write_frame(out)
            if preview:
                cv2.imshow('verification terrain', out)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        if preview:
            cv2.destroyAllWindows()

    print(f'\nVideo enregistree : {target_video_path}')
    print(f'{n_frames} images analysees')
    summarize(counts, n_frames, points_used, errors)
    if refiner is not None and refined['tentees']:
        d = np.array(refined['decalage']) if refined['decalage'] else np.array([0.0])
        print(f"Affinage sur les lignes blanches : accepte sur {100 * refined['acceptees'] / refined['tentees']:.1f} % "
              f"des images ; correction maximale mediane {np.median(d):.1f} px (maximum {d.max():.1f} px)")


def detect_all(source_video_path, device):
    """Execute le modele du terrain une seule fois sur toute la video."""
    model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    detections = []
    for frame in sv.get_video_frames_generator(source_path=source_video_path):
        h, w = frame.shape[:2]
        result = model(frame, verbose=False)[0]
        xy, conf = first_keypoints(sv.KeyPoints.from_ultralytics(result))
        detections.append((
            None if xy is None else np.array(xy, dtype=np.float32),
            None if conf is None else np.array(conf, dtype=np.float32),
            (w, h)))
    return detections


def evaluate(detections, vertices, kp_conf, ransac_cm, smoothing):
    projector = PitchProjector(vertices, kp_conf=kp_conf, ransac_cm=ransac_cm, smoothing=smoothing)
    counts = {'ok': 0, 'rejetee': 0, 'absente': 0}
    points_used, errors = [], []
    for xy, conf, wh in detections:
        if xy is None or len(xy) != len(vertices):
            counts['absente'] += 1
            continue
        H, n_in, state = projector.update(xy, conf, wh)
        counts[state] += 1
        if H is None:
            continue
        points_used.append(n_in)
        idx, _ = inlier_indices(xy, conf, H, vertices, kp_conf, ransac_cm)
        err = reproj_error_px(xy, vertices, H, idx)
        if len(err):
            errors.append(float(np.median(err)))
    return counts, points_used, errors


def run_compare(source_video_path, device):
    vertices = np.array(CONFIG.vertices, dtype=np.float32)
    print('Detection des points du terrain sur toute la video (une seule fois)...')
    detections = detect_all(source_video_path, device)
    n = len(detections)
    print(f'{n} images. Comparaison des reglages :\n')
    header = (f"{'Reglage':46s} {'utilisables':>11s} {'rejetees':>9s} {'points':>7s} "
              f"{'erreur med.':>11s} {'erreur p90':>10s}")
    print(header)
    print('-' * len(header))
    for name, kp_conf, ransac_cm, smoothing in COMPARE_CONFIGS:
        counts, pts, errs = evaluate(detections, vertices, kp_conf, ransac_cm, smoothing)
        e = np.array(errs) if errs else np.array([np.nan])
        print(f"{name:46s} {100 * counts['ok'] / n:10.1f}% {100 * counts['rejetee'] / n:8.1f}% "
              f"{(np.median(pts) if pts else 0):7.0f} {np.nanmedian(e):9.1f}px {np.nanpercentile(e, 90):8.1f}px")
    print('\nLecture : "points" = points du terrain retenus par image (mediane) ; '
          '"erreur" = ecart en pixels entre les points retenus et leur position theorique.')
    print('Attention : une tolerance serree donne mecaniquement une erreur plus faible '
          'car elle retient moins de points. Cherche le meilleur compromis, puis verifie '
          'sur la video avec --kp_conf, --ransac_cm et --smoothing.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verification de la projection sur le terrain')
    parser.add_argument('--source_video_path', required=True)
    parser.add_argument('--target_video_path', default=None)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--kp_conf', type=float, default=0.5,
                        help='confiance minimale des points du terrain (defaut 0.5)')
    parser.add_argument('--ransac_cm', type=float, default=200.0,
                        help='ecart tolere des points, en cm (defaut 200)')
    parser.add_argument('--smoothing', type=float, default=0.6,
                        help='lissage de la projection, 1 = aucun (defaut 0.6)')
    parser.add_argument('--lignes', action='store_true',
                        help='affine aussi la projection sur les lignes blanches (trace jaune)')
    parser.add_argument('--compare', action='store_true',
                        help='compare plusieurs reglages sans produire de video')
    parser.add_argument('--no_preview', action='store_true')
    args = parser.parse_args()
    if args.compare:
        run_compare(args.source_video_path, args.device)
    else:
        if not args.target_video_path:
            parser.error('--target_video_path est requis (sauf avec --compare)')
        run_video(args.source_video_path, args.target_video_path, args.device,
                  args.kp_conf, args.ransac_cm, args.smoothing, not args.no_preview, args.lignes)
